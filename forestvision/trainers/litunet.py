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
            input_stats = getattr(self.trainer.datamodule, "input_stats", None) if hasattr(self, "trainer") else None
        if input_stats is None:
            input_stats = getattr(self, "input_stats", None)

        if target_stats is None:
            target_stats = getattr(self.trainer.datamodule, "target_stats", None) if hasattr(self, "trainer") else None
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
        mask_nodata = (y == ignore_idx).detach().cpu() # [B, C, H, W]

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
                    img = minmax_scaling(img, self.hparams["ignore_index"])
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
            input_stats = getattr(self.trainer.datamodule, "input_stats", None) if hasattr(self, "trainer") else None
        if input_stats is None:
            input_stats = getattr(self, "input_stats", None)

        if target_stats is None:
            target_stats = getattr(self.trainer.datamodule, "target_stats", None) if hasattr(self, "trainer") else None
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

                    img = minmax_scaling(img, self.hparams["ignore_index"])
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
        seg_loss_weight: float = 0.4,
        focal_alpha: float = None,
        focal_gamma: float = 2.0,
        labels: dict = None,
        colormap: dict = None,
    ):
        super().__init__()
        # Save hyperparameters, excluding visualization-only params
        self.save_hyperparameters(ignore=['labels', 'colormap'])
        # Store as instance attributes (not hyperparameters)
        self.labels = labels or {}
        self.colormap = colormap or {}
        self.validation_step_outputs = []

    def configure_models(self):
        """Initialize the UNet model for classification."""
        # Compute total output channels from explicit parameters
        num_classes = self.hparams["num_seg_classes"] + self.hparams["num_reg_targets"]
        self.model = UNet(
            in_channels=self.hparams["in_channels"],
            out_channels=num_classes,
            dropout=self.hparams["dropout"],
        )

    def compute_loss(self, y, logits, ignore_index=None):
        """Compute multi-task loss (Focal for classification + L1 for regression)."""
        # Use hparams ignore_index if not provided
        if ignore_index is None:
            ignore_index = self.hparams.get("ignore_index", -1)

        # Ensure logits and y have matching spatial dimensions
        if logits.shape[2:] != y.shape[2:]:
            logits = F.interpolate(
                logits, size=y.shape[2:], mode="bilinear", align_corners=False
            )

        # Derive class counts from hparams
        num_seg = self.hparams["num_seg_classes"]  # Segmentation classes
        num_reg = self.hparams["num_reg_targets"]  # Regression targets

        # Split outputs: [B, num_seg + num_reg, H, W]
        seg_logits = logits[:, :num_seg]  # [B, num_seg, H, W]
        reg_logits = logits[:, num_seg : num_seg + num_reg]  # [B, num_reg, H, W]

        # Classification loss
        focal_loss = self.focal_loss(seg_logits, y[:, 0].long())

        # Regression loss (vectorized)
        if num_reg > 0:
            reg_target = y[:, 1 : num_reg + 1]  # [B, num_reg, H, W]
            reg_loss_all = self.mae_loss(reg_logits, reg_target)  # [B, num_reg, H, W]

            # Use per-channel masking for regression targets to handle mismatched nodata
            reg_mask = reg_target == ignore_index
            reg_loss_valid = reg_loss_all[~reg_mask]

            # Guard against fully masked batch
            reg_loss = (
                reg_loss_valid.mean()
                if reg_loss_valid.numel() > 0
                else torch.tensor(0.0, device=logits.device)
            )

            # Combine losses using configurable weights
            seg_w = self.hparams["seg_loss_weight"]
            reg_w = 1 - seg_w  
            total_loss = focal_loss * seg_w + reg_loss * reg_w

            return total_loss

        return focal_loss

    def configure_losses(self) -> None:
        """Initialize the loss criterion."""
        self.focal_loss = FocalLoss(
            mode="multiclass",
            alpha=self.hparams.get("focal_alpha"),
            gamma=self.hparams.get("focal_gamma", 2.0),
            reduction="mean",
            ignore_index=self.hparams.get("ignore_index", -1),
        )
        self.mae_loss = nn.L1Loss(reduction="none")

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
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=self.hparams["scheduler_factor"], patience=self.hparams["scheduler_patience"]
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
        # This prevents large negative NoData values (e.g. -2147483648) from crashing torchmetrics
        ignore_idx = self.hparams.get("ignore_index", -1)
        if ignore_idx is not None:
            y[y < 0] = ignore_idx

        y_logits = self(x)  # Model outputs logits

        loss = self.compute_loss(y, y_logits, ignore_idx)
        self.log("train_loss", loss, on_epoch=True, sync_dist=True)

        # Use explicit parameter for segmentation classes
        num_seg = self.hparams["num_seg_classes"]
        ft_logits, fa_logits = (y_logits[:, :num_seg], y_logits[:, num_seg:])

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

        # Sanitize predictions at target resolution before metrics
        if ignore_idx is not None:
            ft_pred_resized = ft_pred_resized.clone()
            ft_pred_resized[y[:, 0] == ignore_idx] = ignore_idx

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

        # Handle variable regression targets for metrics
        num_regression_targets = y.shape[1] - 1
        if num_regression_targets > 0:
            mask = torch.zeros_like(y, dtype=torch.bool)
            if self.hparams["ignore_index"] is not None:
                mask = y == self.hparams["ignore_index"]

            # Only calculate regression metrics for available channels
            y_reg = y[:, 1:, :]
            y_hat_reg = fa_logits_resized[:, :num_regression_targets, :]
            mask_reg = mask[:, 1:, :]

            reg_metrics = self.reg_train_metrics(
                y_hat_reg[~mask_reg].flatten(),
                y_reg[~mask_reg].flatten(),
            )
            self.log_dict(reg_metrics, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for classification."""
        x, y = batch["image"], batch["mask"].long()

        # Sanitize target: remap all negative values to ignore_index
        ignore_idx = self.hparams.get("ignore_index", -1)
        if ignore_idx is not None:
            y[y < 0] = ignore_idx

        y_logits = self(x)  # Model outputs logits

        loss = self.compute_loss(y, y_logits, ignore_idx)
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)

        # Use explicit parameter for segmentation classes
        num_seg = self.hparams["num_seg_classes"]
        ft_logits, fa_logits = (y_logits[:, :num_seg], y_logits[:, num_seg:])
        ft_probs = ft_logits.softmax(dim=1)
        ft_pred = torch.argmax(ft_probs, dim=1)

        # Resize predictions to match target spatial dimensions before metrics computation
        target_h, target_w = y[:, 0, :, :].shape[1:]
        if ft_pred.shape[1:] != (target_h, target_w):
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

        # Sanitize predictions at target resolution before metrics
        if ignore_idx is not None:
            ft_pred_resized = ft_pred_resized.clone()
            ft_pred_resized[y[:, 0] == ignore_idx] = ignore_idx

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

        # Handle variable regression targets for metrics
        num_regression_targets = y.shape[1] - 1
        if num_regression_targets > 0:
            mask = torch.zeros_like(y, dtype=torch.bool)
            if self.hparams["ignore_index"] is not None:
                mask = y == self.hparams["ignore_index"]

            # Only calculate regression metrics for available channels
            y_reg = y[:, 1:, :]
            y_hat_reg = fa_logits_resized[:, :num_regression_targets, :]
            mask_reg = mask[:, 1:, :]

            reg_metrics = self.reg_val_metrics(
                y_hat_reg[~mask_reg].flatten(),
                y_reg[~mask_reg].flatten(),
            )
            self.log_dict(reg_metrics, on_epoch=True, sync_dist=True)

        self.confusion_matrix.update(ft_pred_resized, y[:, 0, :, :])

        # Ensure prediction concatenation matches available regression outputs
        preds = torch.cat(
            [ft_pred_resized.unsqueeze(1), fa_logits_resized[:, :num_regression_targets, :]], dim=1
        )
        
        # Sanitize batch predictions for plotting
        if ignore_idx is not None:
            mask_data = (y[:, 0:1] == ignore_idx)
            preds[mask_data.expand_as(preds)] = float(ignore_idx)
            
        batch["prediction"] = preds
        self.validation_step_outputs.append(batch)

    def test_step(self, batch, batch_idx):
        """Test step for multi-task."""
        x, y = batch["image"], batch["mask"].long()
        
        # Sanitize target: remap all negative values to ignore_index
        ignore_idx = self.hparams.get("ignore_index", -1)
        if ignore_idx is not None:
            y[y < 0] = ignore_idx
            
        y_logits = self(x)  # Model outputs logits

        loss = self.compute_loss(y, y_logits, ignore_idx)
        self.log("test_loss", loss, on_epoch=True, sync_dist=True)

        # Use explicit parameter for segmentation classes
        num_seg = self.hparams["num_seg_classes"]
        ft_logits, fa_logits = (y_logits[:, :num_seg], y_logits[:, num_seg:])
        ft_probs = ft_logits.softmax(dim=1)
        ft_pred = torch.argmax(ft_probs, dim=1)

        # Resize predictions to match target spatial dimensions before metrics computation
        target_h, target_w = y[:, 0, :, :].shape[1:]
        if ft_pred.shape[1:] != (target_h, target_w):
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

        # Sanitize predictions at target resolution before metrics
        if ignore_idx is not None:
            ft_pred_resized = ft_pred_resized.clone()
            ft_pred_resized[y[:, 0] == ignore_idx] = ignore_idx

        if fa_logits.shape[2:] != (target_h, target_w):
            fa_logits_resized = F.interpolate(
                fa_logits,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
        else:
            fa_logits_resized = fa_logits

        seg_metrics = self.seg_test_metrics(ft_pred_resized, y[:, 0, :, :])
        self.log_dict(seg_metrics, sync_dist=True)

        # Handle variable regression targets for metrics
        num_regression_targets = y.shape[1] - 1
        if num_regression_targets > 0:
            mask = torch.zeros_like(y, dtype=torch.bool)
            if self.hparams["ignore_index"] is not None:
                mask = y == self.hparams["ignore_index"]

            # Only calculate regression metrics for available channels
            y_reg = y[:, 1:, :]
            y_hat_reg = fa_logits_resized[:, :num_regression_targets, :]
            mask_reg = mask[:, 1:, :]

            reg_metrics = self.reg_test_metrics(
                y_hat_reg[~mask_reg].flatten(),
                y_reg[~mask_reg].flatten(),
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

    def plot_batch(self, batch, n=5, rgb_bands=[2, 1, 0]):
        """Plot a sample of n images from batch for classification."""
        plt.rcParams["savefig.bbox"] = "tight"
        plt.close("all")  # clear previous plots if any

        # Robust stats lookup
        input_stats = self.hparams.get("input_stats")
        target_stats = self.hparams.get("target_stats")

        if input_stats is None:
            input_stats = getattr(self.trainer.datamodule, "input_stats", None) if hasattr(self, "trainer") else None
        if input_stats is None:
            input_stats = getattr(self, "input_stats", None)

        if target_stats is None:
            target_stats = getattr(self.trainer.datamodule, "target_stats", None) if hasattr(self, "trainer") else None
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
                    print(f"Warning: Stats channel mismatch in MultiTaskUNet.revert(): stats={num_stats_channels}, tensor={num_tensor_channels}. Slicing.")
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
        ignore_idx = self.hparams.get("ignore_index", -1)
        y = y.clone()
        y[y < 0] = ignore_idx
        
        # Create persistent boolean mask for visualization (based on categorical channel)
        mask_nodata = (y[:, 0] == ignore_idx).detach().cpu().numpy() # [B, H, W]

        # Sanitize y_hat for plotting if not already done
        if y_hat is not None:
            y_hat = y_hat.clone()
            y_hat[y[:, 0:1].expand_as(y_hat) == ignore_idx] = float(ignore_idx)

        x = revert(x, input_stats)
        y = revert(y, target_stats, is_target=True)
        y_hat = revert(y_hat, target_stats, is_target=True)

        # Determine actual number of samples to plot (min of n and available samples)
        actual_n = min(n, len(x))

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
                    mask_data = item[col_idx].squeeze().clone().detach().cpu().numpy()
                    
                    # Ensure mask_data is integer for categorical comparison, especially after float reversion
                    mask_data = np.round(mask_data).astype(int)

                    # Create colored mask using colormap
                    colored_mask = np.zeros((*mask_data.shape, 3), dtype=np.uint8)
                    
                    # If colormap is empty, use a default Jet-like mapping for visibility
                    if not self.colormap:
                        # Normalize mask_data to [0, 1] for colormapping
                        valid_mask = ~mask_nodata[col_idx]
                        if valid_mask.any():
                            m_min, m_max = mask_data[valid_mask].min(), mask_data[valid_mask].max()
                            if m_max > m_min:
                                norm_mask = (mask_data - m_min) / (m_max - m_min)
                            else:
                                norm_mask = np.zeros_like(mask_data, dtype=float)
                            
                            # Use matplotlib to get a colored version
                            import matplotlib.cm as cm
                            cmap = cm.get_cmap('tab20')
                            colored_mask = (cmap(norm_mask)[..., :3] * 255).astype(np.uint8)
                    else:
                        # Use provided colormap
                        for class_id, color in self.colormap.items():
                            # Ensure class_id is int for comparison with mask_data
                            try:
                                cid = int(class_id)
                            except (ValueError, TypeError):
                                cid = class_id

                            if isinstance(color, str):
                                # Convert hex to RGB
                                color = tuple(
                                    int(color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)
                                )
                            mask_pixels = mask_data == cid
                            colored_mask[mask_pixels] = color
                            
                        # Add fallback for classes not in colormap (bright red)
                        mapped_mask = np.zeros_like(mask_data, dtype=bool)
                        for class_id in self.colormap.keys():
                            try:
                                cid = int(class_id)
                            except (ValueError, TypeError):
                                cid = class_id
                            mapped_mask |= (mask_data == cid)
                        
                        unmapped_mask = (~mapped_mask) & (~mask_nodata[col_idx])
                        colored_mask[unmapped_mask] = (255, 0, 0) # Red for unmapped classes

                    # Always set ignore index to dark gray
                    colored_mask[mask_nodata[col_idx]] = (50, 50, 50)

                    axs[row_idx, col_idx].imshow(colored_mask)
                    axs[row_idx, col_idx].set_title(f"{title}", fontsize="small")

                    # Add stats to xlabel for debugging
                    unique_vals = np.unique(mask_data[~mask_nodata[col_idx]])
                    axs[row_idx, col_idx].set_xlabel(
                        f"min:{mask_data.min()} max:{mask_data.max()} uniq:{len(unique_vals)}", 
                        fontsize="xx-small"
                    )

                else:
                    # Show regression output
                    img = item[col_idx].squeeze().clone().detach().cpu().numpy()
                    
                    # Handle masking for visualization using persistent mask
                    valid_mask = ~mask_nodata[col_idx]
                    # Calculate stats for better scaling
                    valid_pixels = img[valid_mask]
                    if len(valid_pixels) > 0:
                        vmin, vmax = np.percentile(valid_pixels, [2, 98])
                        if vmin == vmax:
                             vmin, vmax = valid_pixels.min(), valid_pixels.max()
                    else:
                        vmin, vmax = None, None
                    
                    # Set mask value to NaN for viridis colormap to handle correctly
                    img_masked = img.copy()
                    img_masked[mask_nodata[col_idx]] = np.nan
                    
                    axs[row_idx, col_idx].imshow(img_masked, cmap="viridis", vmin=vmin, vmax=vmax)
                    axs[row_idx, col_idx].set_title(f"{title}", fontsize="small")
                    
                    # Add stats to xlabel for debugging
                    if len(valid_pixels) > 0:
                         axs[row_idx, col_idx].set_xlabel(
                             f"min:{valid_pixels.min():.1f} max:{valid_pixels.max():.1f}", 
                             fontsize="xx-small"
                         )

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
