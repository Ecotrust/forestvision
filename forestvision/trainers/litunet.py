import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import (
    MetricCollection,
    Accuracy,
    CohenKappa,
    JaccardIndex,
    ConfusionMatrix,
    MeanSquaredError,
    MeanAbsoluteError,
    R2Score,
)
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torchvision.transforms.functional as tvF
from torchgeo.trainers import BaseTask
from kornia.enhance import Denormalize
from segmentation_models_pytorch.losses import FocalLoss

from forestvision.models.unet import MTUNet

from ..models import UNet
from ..datasets import minmax_scaling
from ..losses import L1SSIMComboLoss, MultiTaskLossWrapper, SharpLoss, HomoscedasticUncertaintyLoss


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
        self.save_hyperparameters()

    def compute_loss(self, y_hat, y, mask=None):
        if self.hparams["loss"] == "l1ssim":
            return self.criterion(y_hat, y, mask)

        loss = self.criterion(y_hat, y)
        if loss.dim() < 4:
            loss = loss.unsqueeze(1)
        if mask is not None:
            loss = loss[~mask]

        # Guard against fully masked batch
        return (
            loss.mean()
            if loss.numel() > 0
            else torch.tensor(0.0, device=y_hat.device, requires_grad=True)
        )

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

        # Sanitize target: remap all negative values to ignore_index
        ignore_idx = self.hparams.get("ignore_index", -1)
        if ignore_idx is not None:
            y[y < 0] = ignore_idx

        y_hat = self(x)

        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(dim=1)

        mask = torch.zeros_like(y, dtype=torch.bool)
        if ignore_idx is not None:
            mask = y == ignore_idx

        loss = self.compute_loss(y_hat, y, mask)
        self.log("train_loss", loss, on_epoch=True, sync_dist=True)

        # Sanitize y_hat for metrics and plotting
        if ignore_idx is not None:
            y_hat = y_hat.clone()
            y_hat[mask] = ignore_idx

        metrics = self.train_metrics(y_hat[~mask].flatten(), y[~mask].flatten())
        metrics.update(train_ssim=self.ssim(y_hat, y))
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        batch["prediction"] = y_hat
        self.training_step_outputs = batch
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["mask"].float()

        # Sanitize target: remap all negative values to ignore_index
        ignore_idx = self.hparams.get("ignore_index", -1)
        if ignore_idx is not None:
            y[y < 0] = ignore_idx

        y_hat = self(x)

        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(dim=1)

        mask = torch.zeros_like(y, dtype=torch.bool)
        if ignore_idx is not None:
            mask = y == ignore_idx

        loss = self.compute_loss(y_hat, y, mask)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        # Sanitize y_hat for metrics and plotting
        if ignore_idx is not None:
            y_hat = y_hat.clone()
            y_hat[mask] = ignore_idx

        metrics = self.val_metrics(y_hat[~mask].flatten(), y[~mask].flatten())
        metrics.update(val_ssim=self.ssim(y_hat, y))
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        batch["prediction"] = y_hat
        self.validation_step_outputs = batch

    def test_step(self, batch, batch_idx):
        x, y = batch["image"], batch["mask"].float()

        # Sanitize target: remap all negative values to ignore_index
        ignore_idx = self.hparams.get("ignore_index", -1)
        if ignore_idx is not None:
            y[y < 0] = ignore_idx

        y_hat = self(x)

        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(dim=1)

        mask = torch.zeros_like(y, dtype=torch.bool)
        if ignore_idx is not None:
            mask = y == ignore_idx
            # Sanitize predictions
            y_hat = y_hat.clone()
            y_hat[mask] = ignore_idx

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
        if self.logger is not None:
            self.logger.experiment.add_figure("train_images", fig, self.global_step)

    def on_validation_epoch_end(self):
        fig = self.plot_batch(self.validation_step_outputs)
        if self.logger is not None:
            self.logger.experiment.add_figure("val_images", fig, self.global_step)

    def on_test_epoch_end(self):
        fig = self.plot_batch(self.test_step_outputs)
        if self.logger is not None:
            self.logger.experiment.add_figure("test_images", fig, self.global_step)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.hparams["lr"], weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def plot_batch(self, batch, n=7, rgb_bands=[3, 2, 1]):
        """Plot a sample of n images from batch."""
        plt.rcParams["savefig.bbox"] = "tight"
        plt.close("all")  # clear previous plots if any

        # Robust stats lookup
        input_stats = self.hparams.get("input_stats")
        target_stats = self.hparams.get("target_stats")

        if input_stats is None:
            input_stats = (
                getattr(self.trainer.datamodule, "input_stats", None)
                if hasattr(self, "trainer")
                else None
            )
        if input_stats is None:
            input_stats = getattr(self, "input_stats", None)

        if target_stats is None:
            target_stats = (
                getattr(self.trainer.datamodule, "target_stats", None)
                if hasattr(self, "trainer")
                else None
            )
        if target_stats is None:
            target_stats = getattr(self, "target_stats", None)

        def revert(tensor, stats):
            if stats is not None:
                m, s = stats["mean"], stats["std"]

                # Convert back to tensor if they were serialized to lists in hparams
                if isinstance(m, list):
                    m = torch.tensor(m).clone()
                if isinstance(s, list):
                    s = torch.tensor(s).clone()

                if isinstance(m, torch.Tensor):
                    m = m.clone()
                if isinstance(s, torch.Tensor):
                    s = s.clone()

                # Ensure device match
                m = m.to(tensor.device)
                s = s.to(tensor.device)

                # Defensive slicing: ensure channel count matches tensor if stats are longer
                num_tensor_channels = tensor.shape[-3]
                if len(m) > num_tensor_channels:
                    m = m[:num_tensor_channels]
                    s = s[:num_tensor_channels]

                return Denormalize(mean=m, std=s)(tensor.float())
            return tensor

        x, y, y_hat = batch["image"], batch["mask"].float(), batch.get("prediction")

        # Sanitize y and y_hat before revert to handle large negative NoData values
        ignore_idx = self.hparams.get("ignore_index", -1)
        y = y.clone()
        y[y < 0] = ignore_idx

        # Create persistent boolean mask for visualization (before denormalization)
        mask_nodata = (y == ignore_idx).detach().cpu()  # [B, C, H, W]

        if y_hat is not None:
            y_hat = y_hat.clone()
            y_hat[y == ignore_idx] = ignore_idx

        x = revert(x, input_stats)
        y = revert(y, target_stats)
        if y_hat is not None:
            y_hat = revert(y_hat, target_stats)

        sample_dict = {
            "x": x[:n],
            "y": y[:n],
        }
        if y_hat is not None:
            sample_dict["y_hat"] = y_hat[:n]

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
                    img = minmax_scaling(img, ignore_idx)
                    if ignore_idx is not None:
                        # Ensure nodata pixels are true black (0) after scaling for visualization
                        img[img == ignore_idx] = 0.0
                    img = tvF.to_pil_image(img)
                    axs[0, i].imshow(np.asarray(img))
                    axs[0, i].set_title("Input", fontsize="small")
                    axs[0, i].get_xaxis().set_ticks([])
                    axs[0, i].get_yaxis().set_ticks([])

            else:
                for i, img in enumerate(item):
                    img = img.squeeze().clone().detach().cpu()
                    msk = mask_nodata[i].squeeze()
                    img[msk == True] = np.nan

                    # Create a colormap that shows NaN as black
                    cmap = plt.get_cmap("viridis").copy()
                    cmap.set_bad(color="black")

                    axs[row_idx, i].imshow(np.asarray(img), cmap=cmap)
                    axs[row_idx, i].set_title(k, fontsize="small")
                    axs[row_idx, i].get_xaxis().set_ticks([])
                    axs[row_idx, i].get_yaxis().set_ticks([])
                    axs[row_idx, i].set_xlabel(
                        f"min:{img.nanmin().item():.2f} max:{img.nanmax().item():.2f} mean:{img.nanmean().item():.2f}",
                        fontsize="xx-small",
                    )
            row_idx += 1

        plt.tight_layout()
        return fig

    def forward(self, x):
        return self.model(x)


class SegmentationUNet(BaseTask):

    target_key = "mask"
    input_stats: dict = None
    target_stats: dict = None

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        loss: str = "ce",
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        ignore_index: int = None,
        dropout: float = 0.0,
        focal_alpha: float = None,
        focal_gamma: float = 2.0,
        labels: dict = None,
        colormap: dict = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.labels = labels or {}
        self.colormap = colormap or {}
        self.validation_step_outputs = []

    def configure_models(self):
        """Initialize the UNet model for classification."""
        self.model = UNet(
            in_channels=self.hparams["in_channels"],
            out_channels=self.hparams["num_classes"],
            dropout=self.hparams["dropout"],
        )

    def configure_losses(self) -> None:
        """Initialize the loss criterion.

        Raises:
            ValueError: If *loss* is invalid.
        """
        loss: str = self.hparams["loss"]
        if loss == "ce":
            self.criterion: nn.Module = nn.CrossEntropyLoss(
                reduction="mean", ignore_index=self.hparams["ignore_index"]
            )
        elif loss == "focal":
            self.criterion: nn.Module = FocalLoss(
                mode="multiclass",
                alpha=self.hparams.get("focal_alpha"),
                gamma=self.hparams.get("focal_gamma", 2.0),
                reduction="mean",
                ignore_index=self.hparams["ignore_index"],
            )
        else:
            raise ValueError(
                f"Loss type '{loss}' is not valid. "
                "Currently, supports 'ce' or 'focal' loss."
            )

    def configure_metrics(self) -> None:
        """Initialize the performance metrics for classification."""
        metrics = MetricCollection(
            {
                "accuracy": Accuracy(
                    task="multiclass",
                    num_classes=self.hparams["num_classes"],  # Keep +1 for ignore_index
                    ignore_index=self.hparams["ignore_index"],
                ),
                "kappa": CohenKappa(
                    task="multiclass",
                    num_classes=self.hparams["num_classes"],
                    weights="quadratic",
                    ignore_index=self.hparams["ignore_index"],
                ),
                "jaccard": JaccardIndex(
                    task="multiclass",
                    num_classes=self.hparams["num_classes"],
                    ignore_index=self.hparams["ignore_index"],
                ),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

        # Confusion matrix for validation epoch end
        self.confusion_matrix = ConfusionMatrix(
            task="multiclass",
            num_classes=self.hparams["num_classes"],  # Keep +1 for ignore_index
            ignore_index=self.hparams["ignore_index"],
        )

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def training_step(self, batch, batch_idx):
        """Training step for classification."""
        x, y = batch["image"], batch["mask"].long()

        # Sanitize target: remap all negative values to ignore_index
        ignore_idx = self.hparams.get("ignore_index", -1)
        if ignore_idx is not None:
            y[y < 0] = ignore_idx

        y_logits = self(x)  # Model outputs logits

        # Compute loss using logits (FocalLoss expects logits and applies softmax internally)
        loss = self.criterion(y_logits, y.squeeze(1))
        self.log("train_loss", loss, on_epoch=True, sync_dist=True)

        # Compute predictions for metrics (apply softmax + argmax)
        y_probs = y_logits.softmax(dim=1)
        y_pred = torch.argmax(y_probs, dim=1)

        metrics = self.train_metrics(y_pred, y.squeeze(1))
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for classification."""
        x, y = batch["image"], batch["mask"].long()

        # Sanitize target: remap all negative values to ignore_index
        ignore_idx = self.hparams.get("ignore_index", -1)
        if ignore_idx is not None:
            y[y < 0] = ignore_idx

        y_logits = self(x)  # Model outputs logits

        loss = self.criterion(y_logits, y.squeeze(1))
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        y_probs = y_logits.softmax(dim=1)
        y_pred = torch.argmax(y_probs, dim=1)
        metrics = self.val_metrics(y_pred, y.squeeze(1))
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        self.confusion_matrix.update(y_pred, y.squeeze(1))

        # Store batch for visualization (use 2D predictions for plotting)
        # Sanitize y_pred using y
        y_pred_viz = y_pred.clone()
        if ignore_idx is not None:
            y_pred_viz[y.squeeze(1) == ignore_idx] = ignore_idx

        batch["prediction"] = y_pred_viz
        self.validation_step_outputs.append(batch)

    def test_step(self, batch, batch_idx):
        """Test step for classification."""
        x, y = batch["image"], batch["mask"].long()
        y_logits = self(x)  # Model outputs logits

        loss = self.criterion(y_logits, y.squeeze(1))
        self.log("test_loss", loss, on_epoch=True, sync_dist=True)

        y_probs = y_logits.softmax(dim=1)
        y_pred = torch.argmax(y_probs, dim=1)
        metrics = self.test_metrics(y_pred, y.squeeze(1))
        self.log_dict(metrics, sync_dist=True)

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        # Compute confusion matrix
        confmat = self.confusion_matrix.compute()
        self.confusion_matrix.reset()

        # Create and log confusion matrix plot
        confmat_fig = self.plot_confusion_matrix(confmat.cpu().numpy())
        if self.logger is not None:
            self.logger.experiment.add_figure(
                "confusion_matrix", confmat_fig, self.current_epoch
            )

        # Create and log sample batch plot
        if self.validation_step_outputs:
            batch_fig = self.plot_batch(self.validation_step_outputs[0])
            if self.logger is not None:
                self.logger.experiment.add_figure(
                    "val_images", batch_fig, self.current_epoch
                )

        self.validation_step_outputs.clear()

    def plot_batch(self, batch, n=5, rgb_bands=[6, 2, 1]):
        """Plot a sample of n images from batch for classification."""
        plt.rcParams["savefig.bbox"] = "tight"
        plt.close("all")  # clear previous plots if any

        # Robust stats lookup
        input_stats = self.hparams.get("input_stats")
        target_stats = self.hparams.get("target_stats")

        if input_stats is None:
            input_stats = (
                getattr(self.trainer.datamodule, "input_stats", None)
                if hasattr(self, "trainer")
                else None
            )
        if input_stats is None:
            input_stats = getattr(self, "input_stats", None)

        if target_stats is None:
            target_stats = (
                getattr(self.trainer.datamodule, "target_stats", None)
                if hasattr(self, "trainer")
                else None
            )
        if target_stats is None:
            target_stats = getattr(self, "target_stats", None)

        def revert(tensor, stats):
            if stats is not None:
                m, s = stats["mean"], stats["std"]

                # Convert back to tensor if they were serialized to lists in hparams
                if isinstance(m, list):
                    m = torch.tensor(m).clone()
                if isinstance(s, list):
                    s = torch.tensor(s).clone()

                # Ensure device match
                m = m.to(tensor.device)
                s = s.to(tensor.device)

                # Defensive slicing: ensure channel count matches tensor if stats are longer
                num_tensor_channels = tensor.shape[-3]
                if len(m) > num_tensor_channels:
                    m = m[:num_tensor_channels]
                    s = s[:num_tensor_channels]

                return Denormalize(mean=m, std=s)(tensor.float())
            return tensor

        x, y, y_hat = batch["image"], batch["mask"].float(), batch.get("prediction")

        # Sanitize y and y_hat
        ignore_idx = self.hparams.get("ignore_index", -1)
        y = y.clone()
        y[y < 0] = ignore_idx

        if y_hat is not None:
            y_hat = y_hat.float().clone()
            y_hat[y == ignore_idx] = ignore_idx

        x = revert(x, input_stats)

        # Determine actual number of samples to plot (min of n and available samples)
        actual_n = min(n, len(x))

        sample_dict = {
            "input": x[:actual_n],
            "ground_truth": y[:actual_n],
        }
        if y_hat is not None:
            sample_dict["prediction"] = y_hat[:actual_n]

        num_rows = len(sample_dict)
        num_cols = actual_n
        fig, axs = plt.subplots(
            figsize=(4 * num_cols, 3 * num_rows),
            nrows=num_rows,
            ncols=num_cols,
            squeeze=False,
        )

        for row_idx, (title, item) in enumerate(sample_dict.items()):
            for col_idx in range(num_cols):
                if title == "input":
                    # Handle different channel counts
                    img_tensor = item[col_idx]
                    num_channels = img_tensor.shape[0]

                    if num_channels >= 3:
                        # Use first 3 channels for RGB
                        img = img_tensor[:3]
                    elif num_channels == 2:
                        # For 2 channels, duplicate the first channel to create RGB
                        img = torch.stack([img_tensor[0], img_tensor[0], img_tensor[0]])
                    else:
                        # For 1 channel, create grayscale RGB
                        img = torch.stack([img_tensor[0], img_tensor[0], img_tensor[0]])

                    img = minmax_scaling(img, ignore_idx)
                    if ignore_idx is not None:
                        # Ensure nodata pixels are true black (0) after scaling for visualization
                        img[img == ignore_idx] = 0.0
                    img = tvF.to_pil_image(img)
                    axs[row_idx, col_idx].imshow(np.asarray(img))
                    axs[row_idx, col_idx].set_title(f"{title}", fontsize="small")
                else:
                    # Show categorical mask
                    mask = item[col_idx].squeeze().clone().detach().cpu().numpy()

                    # Ensure mask is integer for categorical comparison
                    mask = np.round(mask).astype(int)

                    # Create colored mask using colormap
                    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
                    for class_id, color in self.colormap.items():
                        # Ensure class_id is int for comparison with mask
                        try:
                            cid = int(class_id)
                        except (ValueError, TypeError):
                            cid = class_id

                        if isinstance(color, str):
                            # Convert hex to RGB
                            color = tuple(
                                int(color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)
                            )
                        mask_pixels = mask == cid
                        colored_mask[mask_pixels] = color

                    axs[row_idx, col_idx].imshow(colored_mask)
                    axs[row_idx, col_idx].set_title(f"{title}", fontsize="small")

                axs[row_idx, col_idx].get_xaxis().set_ticks([])
                axs[row_idx, col_idx].get_yaxis().set_ticks([])

        plt.tight_layout()
        return fig

    def plot_confusion_matrix(self, confmat, normalize=True):
        """Plot confusion matrix using matplotlib."""
        plt.rcParams["savefig.bbox"] = "tight"
        plt.close("all")  # clear previous plots if any

        fig, ax = plt.subplots(figsize=(10, 8))

        # Normalize by rows (percentage per true class)
        if normalize:
            # Calculate raw sums, avoiding division by zero
            row_sums = confmat.sum(axis=1, keepdims=True)
            # Replace zeros with 1 to avoid division by zero
            row_sums[row_sums == 0] = 1
            confmat_normalized = confmat / row_sums * 100  # Convert to percentage
            display_matrix = confmat_normalized
        else:
            display_matrix = confmat

        # Create heatmap using matplotlib imshow
        im = ax.imshow(display_matrix, cmap="Blues", aspect="auto")

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_label(
            "Percentage (%)" if normalize else "Count", rotation=270, labelpad=20
        )

        # Set labels
        label_names = [
            self.labels.get(i, f"Class {i}") for i in range(confmat.shape[0])
        ]
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(np.arange(len(label_names)))
        ax.set_yticks(np.arange(len(label_names)))
        ax.set_xticklabels(label_names, rotation=45, ha="right")
        ax.set_yticklabels(label_names, rotation=0)

        # Add text annotations
        for i in range(confmat.shape[0]):
            for j in range(confmat.shape[1]):
                if normalize:
                    # Display percentage with 1 decimal place
                    text_value = f"{display_matrix[i, j]:.1f}"
                else:
                    # Display raw count
                    text_value = f"{display_matrix[i, j]:d}"

                text = ax.text(
                    j,
                    i,
                    text_value,
                    ha="center",
                    va="center",
                    color=(
                        "black"
                        if display_matrix[i, j] < display_matrix.max() * 0.7
                        else "white"
                    ),
                )

        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        return fig

    def forward(self, x):
        return self.model(x)


class MultiTaskUNet(BaseTask):

    target_key = "mask"
    input_stats: dict = None
    target_stats: dict = None

    def crop_to_match(
        self, tensor: torch.Tensor, target_shape: tuple[int, int]
    ) -> torch.Tensor:
        """Center crop tensor to match target spatial dimensions.

        Args:
            tensor: Input tensor of shape [B, C, H, W]
            target_shape: Tuple of (target_h, target_w)

        Returns:
            Cropped tensor of shape [B, C, target_h, target_w]
        """
        _, _, h, w = tensor.shape
        target_h, target_w = target_shape

        if h == target_h and w == target_w:
            return tensor

        # Calculate center crop positions
        top = (h - target_h) // 2
        left = (w - target_w) // 2

        return tvF.crop(tensor, top=top, left=left, height=target_h, width=target_w)

    def __init__(
        self,
        in_channels: int = 3,
        num_seg_classes: int = 14,
        num_reg_targets: int = 1,
        loss: str = "ce",
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        ignore_index: int = None,
        dropout: float = 0.0,
        scheduler_patience: int = 10,
        scheduler_factor: float = 0.5,
        focal_alpha: float = None,
        focal_gamma: float = 2.0,
        labels: dict = None,
        colormap: dict = None,
        use_uncertainty_weighting: bool = True,
        init_log_vars: float = 0.0,
        use_loss_normalization: bool = True,
        loss_norm_momentum: float = 0.9,
        reg_loss: str = "mae",
        sharploss_alpha: float = 0.5,
        use_reg_tanh: bool = False,
    ):
        """Multi-task UNet for simultaneous segmentation and regression.

        Args:
            in_channels: Number of input channels.
            num_seg_classes: Number of segmentation classes (excluding ignore_index).
            num_reg_targets: Number of regression targets (e.g. 1 for single continuous variable).
            loss: Loss type for segmentation ("ce" or "focal").
            lr: Learning rate for optimizer.
            weight_decay: Weight decay for optimizer.
            ignore_index: Index to ignore in loss and metrics.
            dropout: Dropout rate for UNet.
            scheduler_patience: Patience for ReduceLROnPlateau scheduler.
            scheduler_factor: Factor for ReduceLROnPlateau scheduler.
            focal_alpha: Alpha parameter for focal loss (if used).
            focal_gamma: Gamma parameter for focal loss (if used).
            labels: Optional dict mapping class indices to human-readable labels for visualization.
            colormap: Optional dict mapping class indices to RGB color tuples for visualization. 
            use_uncertainty_weighting: Whether to use homoscedastic uncertainty-based weighting of losses.
            init_log_vars: Initial log variances for uncertainty weighting (if used).
            use_loss_normalization: Whether to apply running average normalization to losses before weighting.
            loss_norm_momentum: Momentum for updating running average of losses if normalization is used.
            reg_loss: Loss type for regression ("mae", "mse", or "l1ssim").
            sharploss_alpha: Alpha parameter for L1SSIMComboLoss if used for regression loss.  
            use_reg_tanh: Whether to apply a tanh activation to the regression output.
        """
        super().__init__()
        # Save hyperparameters, excluding visualization-only params
        self.save_hyperparameters(ignore=["labels", "colormap"])
        # Store as instance attributes (not hyperparameters)
        self.labels = labels or {}
        self.colormap = colormap or {}
        self.validation_step_outputs = []
        # Store for loss logging
        self._last_loss_logs = {}
        # Running average normalization buffers (initialized to 1.0)
        self.register_buffer("seg_loss_ema", torch.tensor(1.0))
        # Register EMA buffers for each regression channel
        for i in range(self.hparams["num_reg_targets"]):
            self.register_buffer(f"reg_loss_ema_{i}", torch.tensor(1.0))

    def configure_models(self):
        """Initialize the UNet model for classification."""
        # Compute total output channels from explicit parameters
        num_seg_classes = self.hparams["num_seg_classes"]
        num_reg_targets = self.hparams["num_reg_targets"]
        self.model = MTUNet(
            in_channels=self.hparams["in_channels"],
            seg_channels=num_seg_classes,
            reg_channels=num_reg_targets,
            dropout=self.hparams["dropout"],
            use_tanh=self.hparams.get("use_reg_tanh", False),
        )

    def compute_loss(self, y, seg_logits, reg_out, ignore_index=None):
        """Compute multi-task loss using homoscedastic uncertainty weighting.

        Calculates loss for each regression channel separately, then passes all
        task losses to the loss wrapper for uncertainty-based weighting.

        Uses learnable task weights based on homoscedastic uncertainty to balance
        segmentation and regression losses dynamically during training.

        Optionally applies running average normalization to handle highly different
        loss scales between tasks.

        Args:
            y: Target tensor of shape [B, C, H, W] where C includes seg + reg channels.
            seg_logits: Segmentation logits tensor of shape [B, num_seg_classes, H, W].
            reg_out: Regression output tensor of shape [B, num_reg_targets, H, W].
            ignore_index: Index to ignore in loss computation.

        Reference: Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh
        Losses for Scene Geometry and Semantics", CVPR 2018.
        """
        # Use hparams ignore_index if not provided
        if ignore_index is None:
            ignore_index = self.hparams.get("ignore_index")

        # Derive class counts from hparams
        num_reg = self.hparams["num_reg_targets"]  # Regression targets

        # Classification loss (FocalLoss returns scalar)
        seg_loss = self.focal_loss(seg_logits, y[:, 0].long())

        if num_reg > 0 and self.loss_wrapper is not None:
            # Regression loss with masking - calculate per-channel losses
            reg_target = y[:, 1 : num_reg + 1]  # [B, num_reg, H, W]
            reg_losses = []  # List to hold individual channel losses

            for i in range(num_reg):
                # Extract single channel
                reg_out_i = reg_out[:, i : i + 1]  # [B, 1, H, W]
                reg_target_i = reg_target[:, i : i + 1]  # [B, 1, H, W]

                # Handle different regression loss types
                if isinstance(self.reg_loss_fn, SharpLoss):
                    # SharpLoss handles masking internally and returns scalar
                    reg_mask = reg_target_i == ignore_index
                    reg_loss_i = self.reg_loss_fn(reg_out_i, reg_target_i, reg_mask)
                else:
                    # MAE loss returns per-element losses, needs manual masking
                    reg_loss_all = self.reg_loss_fn(
                        reg_out_i, reg_target_i
                    )  # [B, 1, H, W]
                    reg_mask = reg_target_i == ignore_index
                    reg_loss_valid = reg_loss_all[~reg_mask]

                    # Guard against fully masked batch
                    reg_loss_i = (
                        reg_loss_valid.mean()
                        if reg_loss_valid.numel() > 0
                        else torch.tensor(0.0, device=reg_out.device)
                    )

                reg_losses.append(reg_loss_i)

            # Store raw losses for logging
            raw_losses = [seg_loss.detach()] + [rl.detach() for rl in reg_losses]

            # Apply running average normalization if enabled
            if self.hparams.get("use_loss_normalization", False):
                # Normalize segmentation loss
                seg_loss_norm = seg_loss / (self.seg_loss_ema + 1e-8)

                # Normalize each regression channel loss
                reg_losses_norm = []
                for i, reg_loss_i in enumerate(reg_losses):
                    ema_buffer = getattr(self, f"reg_loss_ema_{i}")
                    reg_loss_norm = reg_loss_i / (ema_buffer + 1e-8)
                    reg_losses_norm.append(reg_loss_norm)

                # Update EMAs (in-place, no gradients)
                momentum = self.hparams.get("loss_norm_momentum", 0.9)
                self.seg_loss_ema = (
                    momentum * self.seg_loss_ema + (1 - momentum) * seg_loss.detach()
                )
                for i, reg_loss_i in enumerate(reg_losses):
                    ema_buffer = getattr(self, f"reg_loss_ema_{i}")
                    new_ema = momentum * ema_buffer + (1 - momentum) * reg_loss_i.detach()
                    setattr(self, f"reg_loss_ema_{i}", new_ema)

                task_losses = [seg_loss_norm] + reg_losses_norm
            else:
                task_losses = [seg_loss] + reg_losses

            # Apply homoscedastic uncertainty-based weighting
            total_loss, loss_logs = self.loss_wrapper(task_losses)

            # Add raw losses and EMA values to logs
            loss_logs["task_0_raw_loss"] = raw_losses[0]
            for i, raw_loss in enumerate(raw_losses[1:], start=1):
                loss_logs[f"task_{i}_raw_loss"] = raw_loss
            loss_logs["seg_loss_ema"] = self.seg_loss_ema.detach()
            for i in range(num_reg):
                loss_logs[f"reg_loss_ema_{i}"] = getattr(self, f"reg_loss_ema_{i}").detach()

            # Store logs for training_step to use
            self._last_loss_logs = loss_logs

            return total_loss

        return seg_loss

    def configure_losses(self) -> None:
        """Initialize the loss criterion and uncertainty weighting."""
        self.focal_loss = FocalLoss(
            mode="multiclass",
            alpha=self.hparams.get("focal_alpha"),
            gamma=self.hparams.get("focal_gamma", 2.0),
            reduction="mean",
            ignore_index=self.hparams.get("ignore_index", -1),
        )

        # Regression loss selection
        reg_loss_type = self.hparams.get("reg_loss", "mae")
        if reg_loss_type == "mae":
            self.reg_loss_fn = nn.L1Loss(reduction="none")
        elif reg_loss_type == "sharploss":
            self.reg_loss_fn = SharpLoss(alpha=self.hparams.get("sharploss_alpha", 0.5))
        else:
            raise ValueError(
                f"Regression loss type '{reg_loss_type}' is not valid. "
                "Currently, supports 'mae' or 'sharploss'."
            )

        # Initialize homoscedastic uncertainty-based loss weighting
        if self.hparams.get("use_uncertainty_weighting", True):
            num_reg = self.hparams["num_reg_targets"]
            task_types = ['classification'] + ['regression'] * num_reg
            self.loss_wrapper = HomoscedasticUncertaintyLoss(
                task_types=task_types,
                init_log_vars=self.hparams.get("init_log_vars", 0.0),
            )
        else:
            self.loss_wrapper = None

    def configure_metrics(self) -> None:
        """Initialize the performance metrics for classification."""
        seg_metrics = MetricCollection(
            {
                "accuracy": Accuracy(
                    task="multiclass",
                    num_classes=self.hparams["num_seg_classes"],
                    ignore_index=self.hparams["ignore_index"],
                ),
                "kappa": CohenKappa(
                    task="multiclass",
                    num_classes=self.hparams["num_seg_classes"],
                    weights="quadratic",
                    ignore_index=self.hparams["ignore_index"],
                ),
                "jaccard": JaccardIndex(
                    task="multiclass",
                    num_classes=self.hparams["num_seg_classes"],
                    ignore_index=self.hparams["ignore_index"],
                ),
            }
        )
        self.seg_train_metrics = seg_metrics.clone(prefix="seg_train_")
        self.seg_val_metrics = seg_metrics.clone(prefix="seg_val_")
        self.seg_test_metrics = seg_metrics.clone(prefix="seg_test_")

        # Confusion matrix for validation epoch end
        self.confusion_matrix = ConfusionMatrix(
            task="multiclass",
            num_classes=self.hparams["num_seg_classes"],
            ignore_index=self.hparams["ignore_index"],
        )

        reg_metrics = MetricCollection(
            {
                "rmse": MeanSquaredError(squared=False),
                "mse": MeanSquaredError(squared=True),
                "mae": MeanAbsoluteError(),
                "r2": R2Score(),
            }
        )
        self.reg_train_metrics = reg_metrics.clone(prefix="reg_train_")
        self.reg_val_metrics = reg_metrics.clone(prefix="reg_val_")
        self.reg_test_metrics = reg_metrics.clone(prefix="reg_test_")

    def configure_optimizers(self):
        """Configure optimizer including uncertainty weighting parameters."""
        # Collect all parameters
        params = list(self.model.parameters())

        # Add loss wrapper parameters if using uncertainty weighting
        if self.loss_wrapper is not None:
            params.extend(list(self.loss_wrapper.parameters()))

        optimizer = torch.optim.Adam(
            params,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.hparams["scheduler_factor"],
            patience=self.hparams["scheduler_patience"],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def training_step(self, batch, batch_idx):
        """Training step with uncertainty weighting logging."""
        x, y = batch["image"], batch["mask"].long()

        ignore_idx = self.hparams.get("ignore_index")

        seg_logits, reg_out = self(x)
        # Crop target to match logits shape (avoids interpolation artifacts on predictions)
        if y.shape[2:] != seg_logits.shape[2:]:
            y = self.crop_to_match(y, seg_logits.shape[2:])

        loss = self.compute_loss(y, seg_logits, reg_out, ignore_idx)
        self.log("train_loss", loss, on_epoch=True, sync_dist=True)

        # Log uncertainty weights and raw losses
        if hasattr(self, "_last_loss_logs") and self._last_loss_logs:
            for key, value in self._last_loss_logs.items():
                self.log(f"train_{key}", value, on_epoch=True, sync_dist=True)

        ft_probs = seg_logits.softmax(dim=1)
        ft_pred = torch.argmax(ft_probs, dim=1)

        seg_metrics = self.seg_train_metrics(ft_pred, y[:, 0, :, :])
        self.log_dict(seg_metrics, on_epoch=True, sync_dist=True)

        # Handle variable regression targets for metrics
        num_regression_targets = y.shape[1] - 1
        if num_regression_targets > 0:
            mask = torch.zeros_like(y, dtype=torch.bool)
            if self.hparams["ignore_index"] is not None:
                mask = y == self.hparams["ignore_index"]

            # Only calculate regression metrics for available channels
            y_reg = y[:, 1:, :]
            y_hat_reg = reg_out[:, :num_regression_targets, :]
            mask_reg = mask[:, 1:, :]

            reg_metrics = self.reg_train_metrics(
                y_hat_reg[~mask_reg].flatten(),
                y_reg[~mask_reg].float().flatten(),
            )
            self.log_dict(reg_metrics, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for multi-task classification and regression."""
        x, y = batch["image"], batch["mask"].long()

        ignore_idx = self.hparams.get("ignore_index")

        seg_logits, reg_out = self(x)  # Model outputs tuple
        # Crop target to match logits shape (avoids interpolation artifacts on predictions)
        if y.shape[2:] != seg_logits.shape[2:]:
            y = self.crop_to_match(y, seg_logits.shape[2:])

        loss = self.compute_loss(y, seg_logits, reg_out, ignore_idx)
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)

        ft_probs = seg_logits.softmax(dim=1)
        ft_pred = torch.argmax(ft_probs, dim=1)

        seg_metrics = self.seg_val_metrics(ft_pred, y[:, 0, :, :])
        self.log_dict(seg_metrics, on_epoch=True, sync_dist=True)

        # Handle variable regression targets for metrics
        num_regression_targets = y.shape[1] - 1
        if num_regression_targets > 0:
            mask = torch.zeros_like(y, dtype=torch.bool)
            if self.hparams["ignore_index"] is not None:
                mask = y == self.hparams["ignore_index"]

            # Only calculate regression metrics for available channels
            y_reg = y[:, 1:, :]
            y_hat_reg = reg_out[:, :num_regression_targets, :]
            # Clamp predictions to target min and max range
            # y_hat_reg = torch.clamp(y_hat_reg, -5, 5)
            mask_reg = mask[:, 1:, :]

            reg_metrics = self.reg_val_metrics(
                y_hat_reg[~mask_reg].flatten(),
                y_reg[~mask_reg].float().flatten(),
            )
            self.log_dict(reg_metrics, on_epoch=True, sync_dist=True)

        self.confusion_matrix.update(ft_pred, y[:, 0, :, :])

        # Ensure prediction concatenation matches available regression outputs
        preds = torch.cat([ft_pred.unsqueeze(1), y_hat_reg], dim=1)

        batch["prediction"] = preds
        self.validation_step_outputs.append(batch)

    def test_step(self, batch, batch_idx):
        """Test step for multi-task classification and regression."""
        x, y = batch["image"], batch["mask"].long()

        # Sanitize target: remap all negative values to ignore_index
        ignore_idx = self.hparams.get("ignore_index", -1)

        seg_logits, reg_out = self(x)  # Model outputs tuple
        # Crop target to match logits shape (avoids interpolation artifacts on predictions)
        if y.shape[2:] != seg_logits.shape[2:]:
            y = self.crop_to_match(y, seg_logits.shape[2:])

        loss = self.compute_loss(y, seg_logits, reg_out, ignore_idx)
        self.log("test_loss", loss, on_epoch=True, sync_dist=True)

        ft_probs = seg_logits.softmax(dim=1)
        ft_pred = torch.argmax(ft_probs, dim=1)

        seg_metrics = self.seg_test_metrics(ft_pred, y[:, 0, :, :])
        self.log_dict(seg_metrics, sync_dist=True)

        # Handle variable regression targets for metrics
        num_regression_targets = y.shape[1] - 1
        if num_regression_targets > 0:
            mask = torch.zeros_like(y, dtype=torch.bool)
            if self.hparams["ignore_index"] is not None:
                mask = y == self.hparams["ignore_index"]

            # Only calculate regression metrics for available channels
            y_reg = y[:, 1:, :]
            y_hat_reg = reg_out[:, :num_regression_targets, :]
            mask_reg = mask[:, 1:, :]

            reg_metrics = self.reg_test_metrics(
                y_hat_reg[~mask_reg].flatten(),
                y_reg[~mask_reg].float().flatten(),
            )
            self.log_dict(reg_metrics, sync_dist=True)

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        # Compute confusion matrix
        confmat = self.confusion_matrix.compute()
        self.confusion_matrix.reset()

        # Create and log confusion matrix plot
        confmat_fig = self.plot_confusion_matrix(confmat.cpu().numpy())
        if self.logger is not None:
            self.logger.experiment.add_figure(
                "confusion_matrix", confmat_fig, self.current_epoch
            )

        # Create and log sample batch plot
        if self.validation_step_outputs:
            batch_fig = self.plot_batch(self.validation_step_outputs[0])
            if self.logger is not None:
                self.logger.experiment.add_figure(
                    "val_images", batch_fig, self.current_epoch
                )

        self.validation_step_outputs.clear()

    def forward(self, x):
        """
        Forward pass through the multi-task U-Net.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple of (seg_logits, reg_out).
                - seg_logits: Segmentation logits of shape [B, num_seg_classes, H, W].
                - reg_out: Regression outputs of shape [B, num_reg_targets, H, W]
                  with Tanh activation applied (range [-1, 1]).
        """
        return self.model(x)

    def plot_batch(self, batch, n=10, rgb_bands=[2, 1, 0], max_null_ratio=0.7):
        """Plot a sample of n images from batch for classification.

        Args:
            batch: Batch dictionary containing images, masks, and predictions
            n: Maximum number of samples to plot
            rgb_bands: Bands to use for RGB visualization
            max_null_ratio: Maximum ratio of null pixels to allow in a sample (0.0-1.0)
        """
        plt.rcParams["savefig.bbox"] = "tight"
        plt.close("all")

        # Robust stats lookup
        input_stats = self.hparams.get("input_stats")
        target_stats = self.hparams.get("target_stats")

        if input_stats is None:
            input_stats = (
                getattr(self.trainer.datamodule, "input_stats", None)
                if hasattr(self, "trainer")
                else None
            )
        if input_stats is None:
            input_stats = getattr(self, "input_stats", None)

        if target_stats is None:
            target_stats = (
                getattr(self.trainer.datamodule, "target_stats", None)
                if hasattr(self, "trainer")
                else None
            )
        if target_stats is None:
            target_stats = getattr(self, "target_stats", None)

        def revert(tensor, stats, is_target=False):
            if stats is not None:
                m, s = stats["mean"], stats["std"]

                # Convert back to tensor if they were serialized to lists in hparams
                if isinstance(m, list):
                    m = torch.tensor(m).clone()
                if isinstance(s, list):
                    s = torch.tensor(s).clone()

                if isinstance(m, torch.Tensor):
                    m = m.clone()
                if isinstance(s, torch.Tensor):
                    s = s.clone()

                # Ensure device match
                m = m.to(tensor.device)
                s = s.to(tensor.device)

                num_stats_channels = len(m)
                num_tensor_channels = tensor.shape[-3]

                # STRICT CHECK: Stats must match tensor channels exactly
                if num_stats_channels != num_tensor_channels:
                    print(
                        f"Warning: Stats channel mismatch in MultiTaskUNet.revert(): stats={num_stats_channels}, tensor={num_tensor_channels}. Slicing."
                    )
                    m = m[:num_tensor_channels]
                    s = s[:num_tensor_channels]

                # IMPORTANT: For MultiTask targets, channel 0 is categorical and SHOULD NOT be denormalized
                if is_target and len(m) > 0:
                    m[0] = 0.0
                    s[0] = 1.0

                return Denormalize(mean=m, std=s)(tensor.float())
            return tensor

        x, y, y_hat = batch["image"], batch["mask"], batch["prediction"]

        # Sanitize y before revert to handle large negative NoData values
        ignore_idx = self.hparams.get("ignore_index")
        y = y.clone()
        # y[y < 0] = ignore_idx

        # Crop y to match y_hat shape if needed (handles shape mismatch from UNet output)
        if y_hat is not None and y.shape[2:] != y_hat.shape[2:]:
            y = self.crop_to_match(y, y_hat.shape[2:])

        # Create persistent boolean masks for visualization [B, C, H, W]
        # We track nodata per-channel to handle mismatched NoData patterns in MultiTask
        mask_nodata = (y == ignore_idx).detach().cpu().numpy()

        # Filter samples based on null pixel ratio
        # Calculate null ratio for each sample (considering all channels)
        batch_size = x.shape[0]
        valid_sample_indices = []

        for i in range(batch_size):
            # Calculate null ratio across all channels for this sample
            sample_null_mask = mask_nodata[i]  # [C, H, W]
            total_pixels = sample_null_mask.size  # C * H * W
            null_pixels = np.sum(sample_null_mask)
            null_ratio = null_pixels / total_pixels

            # Keep samples with null ratio below threshold
            if null_ratio <= max_null_ratio:
                valid_sample_indices.append(i)

        # If no samples meet the criteria, use the first 5
        if not valid_sample_indices:
            valid_sample_indices = list(range(min(5, batch_size)))
            print(
                f"Warning: No samples found with null ratio <= {max_null_ratio}. Using first 5 samples."
            )

        # Limit to n samples from valid indices
        if len(valid_sample_indices) > n:
            valid_sample_indices = valid_sample_indices[:n]

        # Sanitize y_hat for plotting if not already done
        if y_hat is not None:
            y_hat = y_hat.clone()
            # Mask y_hat using y's NoData mask (now shapes match)
            y_hat[y == ignore_idx] = float(ignore_idx)

        x = revert(x, input_stats)
        y = revert(y, target_stats, is_target=True)
        y_hat = revert(y_hat, target_stats, is_target=True)

        # Use only valid samples for plotting
        x = x[valid_sample_indices]
        y = y[valid_sample_indices]
        if y_hat is not None:
            y_hat = y_hat[valid_sample_indices]
        mask_nodata = mask_nodata[valid_sample_indices]

        # Determine actual number of samples to plot
        actual_n = len(valid_sample_indices)

        sample_dict = {
            "input": x[:actual_n],
            "fortypba": y[:actual_n, 0],
            "fortypba_pred": y_hat[:actual_n, 0],
        }

        # Dynamically add regression targets if available
        reg_names = ["cancov", "qmd_dom", "ba_ge_3"]
        num_reg_available = y.shape[1] - 1
        for i in range(num_reg_available):
            name = reg_names[i] if i < len(reg_names) else f"reg_{i}"
            sample_dict[name] = y[:actual_n, i + 1]
            sample_dict[f"{name}_pred"] = y_hat[:actual_n, i + 1]

        # Pre-calculate vmin/vmax for each regression target from target data
        # This ensures predictions use the same colormap range as targets
        reg_range = {}
        for i in range(num_reg_available):
            name = reg_names[i] if i < len(reg_names) else f"reg_{i}"
            target_data = y[:actual_n, i + 1]  # [actual_n, H, W]

            # Collect valid pixels across all samples for this target
            valid_pixels_list = []
            for col_idx in range(actual_n):
                ch_idx = 1 + i  # Regression channel index
                current_mask = mask_nodata[col_idx, ch_idx]
                img = target_data[col_idx].squeeze().cpu().numpy()
                valid_pixels = img[~current_mask]
                if len(valid_pixels) > 0:
                    valid_pixels_list.append(valid_pixels)

            if valid_pixels_list:
                all_valid = np.concatenate(valid_pixels_list)
                vmin, vmax = np.percentile(all_valid, [2, 98])
                if vmin == vmax:
                    vmin, vmax = all_valid.min(), all_valid.max()
                reg_range[name] = (vmin, vmax)
            else:
                reg_range[name] = (None, None)

        num_rows = len(sample_dict)
        num_cols = actual_n
        fig, axs = plt.subplots(
            figsize=(4 * num_cols, 3 * num_rows),
            nrows=num_rows,
            ncols=num_cols,
            squeeze=False,
        )

        # Prepare regression colormap once to show NaNs as black
        reg_cmap = plt.get_cmap("viridis").copy()
        reg_cmap.set_bad(color="black")

        for row_idx, (title, item) in enumerate(sample_dict.items()):
            for col_idx in range(num_cols):
                ax = axs[row_idx, col_idx]
                if title == "input":
                    # Handle different channel counts
                    img_tensor = item[col_idx]
                    num_channels = img_tensor.shape[0]

                    if num_channels >= 3:
                        # Use first 3 channels for RGB
                        img = img_tensor[rgb_bands, :]
                    elif num_channels == 2:
                        # For 2 channels, duplicate the first channel to create RGB
                        img = torch.stack([img_tensor[0], img_tensor[0], img_tensor[0]])
                    else:
                        # For 1 channel, create grayscale RGB
                        img = torch.stack([img_tensor[0], img_tensor[0], img_tensor[0]])

                    # Explicitly identify NoData areas for input images to show in black
                    # Use the combined target mask as a reference for NoData areas in visualization
                    # combined_nodata = mask_nodata[col_idx].any(axis=0)

                    img = minmax_scaling(img, ignore_idx)
                    img = img.clone()
                    # Ensure all channels are black where we have NoData (force to 0.0 after scaling)
                    # img[:, combined_nodata] = 0.0

                    img = tvF.to_pil_image(img)
                    img = tvF.adjust_contrast(img, 2)
                    img = tvF.adjust_brightness(img, 1)
                    ax.imshow(np.asarray(img))
                    ax.set_title(f"{title}", fontsize="small")
                elif title.startswith("fortypba"):
                    # Show categorical mask
                    mask_data = item[col_idx].squeeze().clone().detach().cpu().numpy()

                    # fortypba is at channel 0
                    ch_idx = 0
                    current_mask = mask_nodata[col_idx, ch_idx]

                    # Ensure mask_data is integer for categorical comparison
                    mask_data = np.round(mask_data).astype(int)

                    # Create colored mask using colormap
                    colored_mask = np.zeros((*mask_data.shape, 3), dtype=np.uint8)

                    if not self.colormap:
                        # Default colormapping if empty
                        valid_mask = ~current_mask
                        if valid_mask.any():
                            m_min, m_max = (
                                mask_data[valid_mask].min(),
                                mask_data[valid_mask].max(),
                            )
                            if m_max > m_min:
                                norm_mask = (mask_data - m_min) / (m_max - m_min)
                            else:
                                norm_mask = np.zeros_like(mask_data, dtype=float)

                            import matplotlib.cm as cm

                            cmap = cm.get_cmap("tab20")
                            colored_mask = (cmap(norm_mask)[..., :3] * 255).astype(
                                np.uint8
                            )
                    else:
                        # Use provided colormap
                        for class_id, color in self.colormap.items():
                            try:
                                cid = int(class_id)
                            except (ValueError, TypeError):
                                cid = class_id

                            if isinstance(color, str):
                                color = tuple(
                                    int(color.lstrip("#")[i : i + 2], 16)
                                    for i in (0, 2, 4)
                                )
                            mask_pixels = mask_data == cid
                            colored_mask[mask_pixels] = color

                        # Add fallback for classes not in colormap (red)
                        mapped_mask = np.zeros_like(mask_data, dtype=bool)
                        for class_id in self.colormap.keys():
                            try:
                                cid = int(class_id)
                            except (ValueError, TypeError):
                                cid = class_id
                            mapped_mask |= mask_data == cid

                        unmapped_mask = (~mapped_mask) & (~current_mask)
                        colored_mask[unmapped_mask] = (255, 0, 0)

                    # Always set ignore index to black
                    colored_mask[current_mask] = (0, 0, 0)

                    ax.imshow(colored_mask)
                    ax.set_title(f"{title}", fontsize="small")

                    # Add stats to xlabel for debugging
                    unique_vals = np.unique(mask_data[~current_mask])
                    ax.set_xlabel(
                        f"min:{mask_data.min()} max:{mask_data.max()} uniq:{len(unique_vals)}",
                        fontsize="xx-small",
                    )

                else:
                    # Show regression output
                    img = item[col_idx].squeeze().clone().detach().cpu().numpy()

                    # Calculate channel index to retrieve the correct NoData mask
                    reg_ch_offset = row_idx - 3
                    ch_idx = 1 + (reg_ch_offset // 2)
                    current_mask = mask_nodata[col_idx, ch_idx]

                    # Get the regression variable name (remove "_pred" suffix if present)
                    reg_name = title.replace("_pred", "")
                    # Use pre-computed vmin/vmax from target data for consistent colormap
                    vmin, vmax = reg_range.get(reg_name, (None, None))

                    img_masked = img.copy()
                    img_masked[current_mask] = np.nan

                    ax.imshow(img_masked, cmap=reg_cmap, vmin=vmin, vmax=vmax)
                    ax.set_title(f"{title}", fontsize="small")

                    valid_mask = ~current_mask
                    if valid_mask.any():
                        ax.set_xlabel(
                            f"min:{np.nanmin(img_masked):.1f} max:{np.nanmax(img_masked):.1f} mean:{np.nanmean(img_masked):.1f}",
                            fontsize="xx-small",
                        )

                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])

        plt.tight_layout()
        return fig

    def plot_confusion_matrix(self, confmat, normalize=True):
        """Plot confusion matrix using matplotlib."""
        plt.rcParams["savefig.bbox"] = "tight"
        plt.close("all")  # clear previous plots if any

        fig, ax = plt.subplots(figsize=(10, 8))

        # Normalize by rows (percentage per true class)
        if normalize:
            # Calculate raw sums, avoiding division by zero
            row_sums = confmat.sum(axis=1, keepdims=True)
            # Replace zeros with 1 to avoid division by zero
            row_sums[row_sums == 0] = 1
            confmat_normalized = confmat / row_sums * 100  # Convert to percentage
            display_matrix = confmat_normalized
        else:
            display_matrix = confmat

        # Create heatmap using matplotlib imshow
        im = ax.imshow(display_matrix, cmap="Blues", aspect="auto")

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_label(
            "Percentage (%)" if normalize else "Count", rotation=270, labelpad=20
        )

        # Set labels
        label_names = [
            self.labels.get(i, f"Class {i}") for i in range(confmat.shape[0])
        ]
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(np.arange(len(label_names)))
        ax.set_yticks(np.arange(len(label_names)))
        ax.set_xticklabels(label_names, rotation=45, ha="right")
        ax.set_yticklabels(label_names, rotation=0)

        # Add text annotations
        for i in range(confmat.shape[0]):
            for j in range(confmat.shape[1]):
                if normalize:
                    # Display percentage with 1 decimal place
                    text_value = f"{display_matrix[i, j]:.1f}"
                else:
                    # Display raw count
                    text_value = f"{display_matrix[i, j]:d}"

                text = ax.text(
                    j,
                    i,
                    text_value,
                    ha="center",
                    va="center",
                    color=(
                        "black"
                        if display_matrix[i, j] < display_matrix.max() * 0.7
                        else "white"
                    ),
                )

        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        return fig
