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


class SegmentationUNet(BaseTask):

    target_key = "mask"
    input_stats: dict = None

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        loss: str = "ce",
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        ignore_index: int = None,
        dropout: float = 0.0,
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
                reduction="none", ignore_index=self.hparams["ignore_index"]
            )
        elif loss == "focal":
            self.criterion: nn.Module = FocalLoss(
                mode="multiclass",
                gamma=2,
                reduction="none",
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
        return optimizer

    def training_step(self, batch, batch_idx):
        """Training step for classification."""
        x, y = batch["image"], batch["mask"].long()
        y_logits = self(x)  # Model outputs logits

        # Compute loss using logits (FocalLoss expects logits and applies softmax internally)
        loss = self.criterion(y_logits, y.squeeze(1))
        loss = loss.mean()
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
        y_logits = self(x)  # Model outputs logits

        loss = self.criterion(y_logits, y.squeeze(1))
        loss = loss.mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        y_probs = y_logits.softmax(dim=1)
        y_pred = torch.argmax(y_probs, dim=1)
        metrics = self.val_metrics(y_pred, y.squeeze(1))
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        self.confusion_matrix.update(y_pred, y.squeeze(1))

        # Store batch for visualization (use 2D predictions for plotting)
        batch["prediction"] = y_pred
        self.validation_step_outputs.append(batch)

    def test_step(self, batch, batch_idx):
        """Test step for classification."""
        x, y = batch["image"], batch["mask"].long()
        y_logits = self(x)  # Model outputs logits

        loss = self.criterion(y_logits, y.squeeze(1))
        loss = loss.mean()
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

        try:
            input_stats = self.trainer.datamodule.input_stats
        except AttributeError:
            input_stats = self.input_stats

        def revert(tensor, stats):
            if stats is not None:
                return Denormalize(mean=stats["mean"], std=stats["std"])(tensor)
            return tensor

        x, y = batch["image"], batch["mask"]
        predictions = batch.get("prediction", None)
        x = revert(x, input_stats)

        # Determine actual number of samples to plot (min of n and available samples)
        actual_n = min(n, len(x))

        sample_dict = {
            "input": x[:actual_n],
            "ground_truth": y[:actual_n],
        }
        if predictions is not None:
            sample_dict["prediction"] = predictions[:actual_n]

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

                    img = minmax_scaling(img, self.hparams["ignore_index"])
                    img = tvF.to_pil_image(img)
                    axs[row_idx, col_idx].imshow(np.asarray(img))
                    axs[row_idx, col_idx].set_title(f"{title}", fontsize="small")
                else:
                    # Show categorical mask
                    mask = item[col_idx].squeeze().clone().detach().cpu().numpy()

                    # Create colored mask using colormap
                    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
                    for class_id, color in self.colormap.items():
                        if isinstance(color, str):
                            # Convert hex to RGB
                            color = tuple(
                                int(color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)
                            )
                        mask_pixels = mask == class_id
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

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        loss: str = "ce",
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        ignore_index: int = None,
        dropout: float = 0.0,
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

    def compute_loss(self, y, logits, ignore_index=-1):
        # Ensure logits and y have matching spatial dimensions
        if logits.shape[2:] != y.shape[2:]:
            # Resize logits to match y's spatial dimensions
            target_h, target_w = y.shape[2:]
            logits = F.interpolate(
                logits, size=(target_h, target_w), mode="bilinear", align_corners=False
            )

        ndmask = y[:, 0, :] == ignore_index
        # ndmask = ndmask.expand(-1, 3, -1, -1)
        mae = nn.L1Loss(reduction="none")
        focal = FocalLoss(
            mode="multiclass",
            gamma=2,
            reduction="mean",
            ignore_index=ignore_index,
        )
        fty, fat = (logits[:, :14], logits[:, 14:])
        l1_loss1 = mae(fat[:, 0, :], y[:, 1, :])[~ndmask].mean()
        l1_loss2 = mae(fat[:, 1, :], y[:, 2, :])[~ndmask].mean()
        l1_loss3 = mae(fat[:, 2, :], y[:, 3, :])[~ndmask].mean()
        l1_loss = (l1_loss1 * 0.8 + l1_loss2 * 0.1 + l1_loss3 * 0.1).sum()

        focal_loss = focal(fty, y[:, 0, :, :])
        return focal_loss * 0.6 + l1_loss * 0.4

    def configure_losses(self) -> None:
        """Initialize the loss criterion.

        Raises:
            ValueError: If *loss* is invalid.
        """
        pass

    def configure_metrics(self) -> None:
        """Initialize the performance metrics for classification."""
        seg_metrics = MetricCollection(
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
        self.seg_train_metrics = seg_metrics.clone(prefix="seg_train_")
        self.seg_val_metrics = seg_metrics.clone(prefix="seg_val_")
        self.seg_test_metrics = seg_metrics.clone(prefix="seg_test_")

        # Confusion matrix for validation epoch end
        self.confusion_matrix = ConfusionMatrix(
            task="multiclass",
            num_classes=14,
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
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        """Training step for classification."""
        x, y = batch["image"], batch["mask"].long()
        y_logits = self(x)  # Model outputs logits

        loss = self.compute_loss(y, y_logits, self.hparams["ignore_index"])
        self.log("train_loss", loss, on_epoch=True, sync_dist=True)

        ft_logits, fa_logits = (y_logits[:, :14], y_logits[:, 14:])

        ft_probs = ft_logits.softmax(dim=1)
        ft_pred = torch.argmax(ft_probs, dim=1)

        # Resize predictions to match target spatial dimensions before metrics computation
        target_h, target_w = y[:, 0, :, :].shape[1:]
        if ft_pred.shape[1:] != (target_h, target_w):
            # Reshape for interpolation: [batch, 1, h, w]
            ft_pred_resized = (
                F.interpolate(
                    ft_pred.unsqueeze(1).float(),
                    size=(target_h, target_w),
                    mode="nearest",
                )
                .squeeze(1)
                .long()
            )
        else:
            ft_pred_resized = ft_pred

        if fa_logits.shape[2:] != (target_h, target_w):
            fa_logits_resized = F.interpolate(
                fa_logits,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
        else:
            fa_logits_resized = fa_logits

        seg_metrics = self.seg_train_metrics(ft_pred_resized, y[:, 0, :, :])
        self.log_dict(seg_metrics, on_epoch=True, sync_dist=True)

        mask = torch.zeros_like(y, dtype=torch.bool)
        if self.hparams["ignore_index"] is not None:
            mask = y == self.hparams["ignore_index"]

        reg_metrics = self.reg_train_metrics(
            fa_logits_resized[~mask[:, 1:, :]].flatten(),
            y[:, 1:, :][~mask[:, 1:, :]].flatten(),
        )
        self.log_dict(reg_metrics, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for classification."""
        x, y = batch["image"], batch["mask"].long()
        y_logits = self(x)  # Model outputs logits

        loss = self.compute_loss(y, y_logits, self.hparams["ignore_index"])
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)

        ft_logits, fa_logits = (y_logits[:, :14], y_logits[:, 14:])
        ft_probs = ft_logits.softmax(dim=1)
        ft_pred = torch.argmax(ft_probs, dim=1)

        # Resize predictions to match target spatial dimensions before metrics computation
        target_h, target_w = y[:, 0, :, :].shape[1:]
        if ft_pred.shape[1:] != (target_h, target_w):
            # Reshape for interpolation: [batch, 1, h, w]
            print(target_h, target_w)
            ft_pred_resized = (
                F.interpolate(
                    ft_pred.unsqueeze(1).float(),
                    size=(target_h, target_w),
                    mode="nearest",
                )
                .squeeze(1)
                .long()
            )
        else:
            ft_pred_resized = ft_pred

        if fa_logits.shape[2:] != (target_h, target_w):
            fa_logits_resized = F.interpolate(
                fa_logits,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
        else:
            fa_logits_resized = fa_logits

        seg_metrics = self.seg_val_metrics(ft_pred_resized, y[:, 0, :, :])
        self.log_dict(seg_metrics, on_epoch=True, sync_dist=True)
        if self.hparams["ignore_index"] is not None:
            mask = y == self.hparams["ignore_index"]
        reg_metrics = self.reg_val_metrics(
            fa_logits_resized[~mask[:, 1:, :]].flatten(),
            y[:, 1:, :][~mask[:, 1:, :]].flatten(),
        )
        self.log_dict(reg_metrics, on_epoch=True, sync_dist=True)

        self.confusion_matrix.update(ft_pred_resized, y[:, 0, :, :])

        batch["prediction"] = torch.cat(
            [ft_pred_resized.unsqueeze(1), fa_logits_resized], dim=1
        )
        self.validation_step_outputs.append(batch)

    def test_step(self, batch, batch_idx):
        """Test step for classification."""
        x, y = batch["image"], batch["mask"].long()
        y_logits = self(x)  # Model outputs logits

        loss = self.criterion(y_logits, y.squeeze(1))
        loss = loss.mean()
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

    def plot_batch(self, batch, n=5, rgb_bands=[3, 2, 1]):
        """Plot a sample of n images from batch for classification."""
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

        x, y, y_hat = batch["image"], batch["mask"], batch["prediction"]
        x = revert(x, input_stats)
        y = revert(y, target_stats)
        y_hat = revert(y_hat, target_stats)

        # Determine actual number of samples to plot (min of n and available samples)
        actual_n = min(n, len(x))

        sample_dict = {
            "input": x[:actual_n],
            "fortypba": y[:actual_n, 0],
            "fortypba_pred": y_hat[:actual_n, 0],
            "cancov": y[:actual_n, 1],
            "cancov_pred": y_hat[:actual_n, 1],
            "qmd_dom": y[:actual_n, 2],
            "qmd_dom_pred": y_hat[:actual_n, 2],
            "ba_ge_3": y[:actual_n, 3],
            "ba_ge_3_pred": y_hat[:actual_n, 3],
        }

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
                        img = img_tensor[rgb_bands, :]
                    elif num_channels == 2:
                        # For 2 channels, duplicate the first channel to create RGB
                        img = torch.stack([img_tensor[0], img_tensor[0], img_tensor[0]])
                    else:
                        # For 1 channel, create grayscale RGB
                        img = torch.stack([img_tensor[0], img_tensor[0], img_tensor[0]])

                    img = minmax_scaling(img, self.hparams["ignore_index"])
                    img = tvF.to_pil_image(img)
                    img = tvF.adjust_contrast(img, 2)
                    img = tvF.adjust_brightness(img, 1)
                    axs[row_idx, col_idx].imshow(np.asarray(img))
                    axs[row_idx, col_idx].set_title(f"{title}", fontsize="small")
                elif title.startswith("fortypba"):
                    # Show categorical mask
                    mask = item[col_idx].squeeze().clone().detach().cpu().numpy()

                    # Create colored mask using colormap
                    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
                    for class_id, color in self.colormap.items():
                        if isinstance(color, str):
                            # Convert hex to RGB
                            color = tuple(
                                int(color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)
                            )
                        mask_pixels = mask == class_id
                        colored_mask[mask_pixels] = color

                    axs[row_idx, col_idx].imshow(colored_mask)
                    axs[row_idx, col_idx].set_title(f"{title}", fontsize="small")

                else:
                    # Show regression output
                    img = item[col_idx].squeeze().clone().detach().cpu()
                    axs[row_idx, col_idx].imshow(np.asarray(img), cmap="viridis")
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
