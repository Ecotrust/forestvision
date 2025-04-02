import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchmetrics import (
    MeanSquaredError,
    MetricCollection,
    MeanAbsoluteError,
    R2Score,
)
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torchvision.transforms.functional as tvF
from torchgeo.trainers import BaseTask
from kornia.enhance import Denormalize

from ..models import UNet
from ..datasets import minmax_scaling
from ..losses import L1SSIMComboLoss


class RegressionUNet(BaseTask):

    target_key = "mask"
    input_stats: dict = None
    target_stats: dict = None

    def __init__(
        self,
        in_channels: int = 3,
        num_outputs: int = 1,
        loss: str = "mae",
        lr: float = 1e-4,
        dropout: float = 0.5,
        ignore_index=None,
    ):
        super().__init__()

    def compute_loss(self, y_hat, y, mask=None):
        if self.hparams["loss"] == "l1ssim":
            return self.criterion(y_hat, y, mask)

        loss = self.criterion(y_hat, y)
        if loss.dim() < 4:
            loss = loss.unsqueeze(1)
        if mask is not None:
            loss = loss[~mask]
        return loss.mean()

    def configure_losses(self) -> None:
        """Initialize the loss criterion.

        Raises:
            ValueError: If *loss* is invalid.
        """
        loss: str = self.hparams["loss"]
        if loss == "mse":
            self.criterion: nn.Module = nn.MSELoss(reduction="none")
        elif loss == "mae":
            self.criterion: nn.Module = nn.L1Loss(reduction="none")
        elif loss == "l1ssim":
            self.criterion: nn.Module = L1SSIMComboLoss()
        else:
            raise ValueError(
                f"Loss type '{loss}' is not valid. "
                "Currently, supports 'mse' or 'mae' loss."
            )

    def configure_metrics(self) -> None:
        """Initialize the performance metrics.

        * :class:`~torchmetrics.MeanSquaredError`: The average of the squared
          differences between the predicted and actual values (MSE) and its
          square root (RMSE). Lower values are better.
        * :class:`~torchmetrics.MeanAbsoluteError`: The average of the absolute
          differences between the predicted and actual values (MAE).
          Lower values are better.
        """
        metrics = MetricCollection(
            {
                "rmse": MeanSquaredError(squared=False),
                "mse": MeanSquaredError(squared=True),
                "mae": MeanAbsoluteError(),
                "r2": R2Score(),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")
        self.ssim = StructuralSimilarityIndexMeasure()

    def configure_models(self):
        self.model = UNet(
            in_channels=self.hparams["in_channels"],
            out_channels=self.hparams["num_outputs"],
            dropout=self.hparams["dropout"],
        )

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["mask"].float()
        y_hat = self(x)

        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(dim=1)

        mask = torch.zeros_like(y, dtype=torch.bool)
        if self.hparams["ignore_index"] is not None:
            mask = y == self.hparams["ignore_index"]

        loss = self.compute_loss(y_hat, y, mask)
        self.log("train_loss", loss, on_epoch=True, sync_dist=True)

        metrics = self.train_metrics(y_hat[~mask].flatten(), y[~mask].flatten())
        metrics.update(train_ssim=self.ssim(y_hat, y))
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        batch["prediction"] = y_hat
        self.training_step_outputs = batch
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["mask"].float()
        y_hat = self(x)

        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(dim=1)

        mask = torch.zeros_like(y, dtype=torch.bool)
        if self.hparams["ignore_index"] is not None:
            mask = y == self.hparams["ignore_index"]

        loss = self.compute_loss(y_hat, y, mask)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        metrics = self.val_metrics(y_hat[~mask].flatten(), y[~mask].flatten())
        metrics.update(val_ssim=self.ssim(y_hat, y))
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        batch["prediction"] = y_hat
        self.validation_step_outputs = batch

    def test_step(self, batch, batch_idx):
        x, y = batch["image"], batch["mask"].float()
        y_hat = self(x)

        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(dim=1)

        mask = torch.zeros_like(y, dtype=torch.bool)
        if self.hparams["ignore_index"] is not None:
            mask = y == self.hparams["ignore_index"]

        loss = self.compute_loss(y_hat, y, mask)
        self.log("test_loss", loss, on_epoch=True, sync_dist=True)

        metrics = self.test_metrics(y_hat[~mask].flatten(), y[~mask].flatten())
        metrics.update(test_ssim=self.ssim(y_hat, y))
        self.log_dict(metrics, sync_dist=True)

        batch["prediction"] = y_hat
        self.test_step_outputs = batch

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch["image"], batch["mask"].float()
        y_hat = self(x)

        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(dim=1)

        if self.hparams["ignore_index"] is not None:
            mask = y == self.hparams["ignore_index"]
            y_hat[mask] = self.hparams["ignore_index"]

        return y_hat

    def on_train_epoch_end(self):
        fig = self.plot_batch(self.training_step_outputs)
        self.logger.experiment.add_figure("train_images", fig, self.global_step)

    def on_validation_epoch_end(self):
        fig = self.plot_batch(self.validation_step_outputs)
        self.logger.experiment.add_figure("val_images", fig, self.global_step)

    def on_test_epoch_end(self):
        fig = self.plot_batch(self.test_step_outputs)
        self.logger.experiment.add_figure("test_images", fig, self.global_step)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.model.parameters(), lr=self.hparams["lr"], weight_decay=1e-5
        )

    def plot_batch(self, batch, n=7, rgb_bands=[3, 2, 1]):
        """Plot a sample of n images from batch."""
        plt.rcParams["savefig.bbox"] = "tight"
        plt.close("all")  # clear previous plots if any

        try:
            input_stats = self.trainer.datamodule.input_stats
            target_stats = self.trainer.datamodule.target_stats
        except AttributeError:
            input_stats = self.input_stats
            target_stats = self.target_stats

        def revert(tensor, stats):
            if stats is not None:
                return Denormalize(mean=stats["mean"], std=stats["std"])(tensor)
            return tensor

        x, y = batch["image"], batch["mask"].float()
        mask = y == self.hparams["ignore_index"]
        predictions = batch.get("prediction", None)
        x = revert(x, input_stats)
        y = revert(y, target_stats)
        sample_dict = {
            "x": x[:n],
            "y": y[:n],
        }
        if predictions is not None:
            sample_dict["y_hat"] = revert(predictions[:n], target_stats)

        num_rows = len(sample_dict)
        num_cols = len(sample_dict["x"])
        fig, axs = plt.subplots(
            figsize=(12, 7), nrows=num_rows, ncols=num_cols, squeeze=False
        )
        row_idx = 0
        for k, item in sample_dict.items():
            if k == "x":
                item = item[:, rgb_bands]
                for i, img in enumerate(item):
                    img = minmax_scaling(img, self.hparams["ignore_index"])
                    img = tvF.to_pil_image(img)
                    axs[0, i].imshow(np.asarray(img))
                    axs[0, i].set_title("Input", fontsize="small")
                    axs[0, i].get_xaxis().set_ticks([])
                    axs[0, i].get_yaxis().set_ticks([])

            else:
                for i, img in enumerate(item):
                    img = img.squeeze().clone().detach().cpu()
                    msk = mask[i].squeeze().detach().cpu()
                    img[msk == True] = np.nan
                    axs[row_idx, i].imshow(np.asarray(img), cmap="viridis")
                    axs[row_idx, i].set_title(k, fontsize="small")
                    axs[row_idx, i].get_xaxis().set_ticks([])
                    axs[row_idx, i].get_yaxis().set_ticks([])
                    axs[row_idx, i].set_xlabel(
                        f"min:{img.min().item():.2f} max:{img.max().item():.2f} mean:{img.mean().item():.2f}",
                        fontsize="xx-small",
                    )
            row_idx += 1

        plt.tight_layout()
        return fig

    def forward(self, x):
        return self.model(x)
