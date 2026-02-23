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

from forestvision.models.unet import MTUNet, ResMTUNet, OptimizedMTUNet

from ..models import UNet
from ..datasets import minmax_scaling
from ..losses import L1SSIMComboLoss, SharpLoss, HomoscedasticUncertaintyLoss


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
            self.criterion: nn.Module = L1SSIMComboLoss(
                w=self.hparams.get("l1ssim_w", [1, 1])
            )
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

        # DEBUG: Print batch stats
        print(f"\n[DEBUG training_step] Batch {batch_idx}")
        print(f"[DEBUG training_step] x shape: {x.shape}, dtype: {x.dtype}")
        print(f"[DEBUG training_step] x stats: min={x.min().item():.2f}, max={x.max().item():.2f}, mean={x.mean().item():.2f}")
        print(f"[DEBUG training_step] y shape: {y.shape}, dtype: {y.dtype}")
        print(f"[DEBUG training_step] y stats: min={y.min().item():.2f}, max={y.max().item():.2f}, mean={y.mean().item():.2f}")

        ignore_idx = self.hparams.get("ignore_index", -1)
        print(f"[DEBUG training_step] ignore_index: {ignore_idx}")

        y_hat = self(x)

        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(dim=1)

        mask = torch.zeros_like(y, dtype=torch.bool)
        if ignore_idx is not None:
            # Robust mask: exact match OR large negative value
            mask = (y == ignore_idx) | (y < -1e9)

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

        ignore_idx = self.hparams.get("ignore_index", -1)
        y_hat = self(x)

        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(dim=1)

        mask = torch.zeros_like(y, dtype=torch.bool)
        if ignore_idx is not None:
            mask = (y == ignore_idx) | (y < -1e9)

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

        ignore_idx = self.hparams.get("ignore_index", -1)
        y_hat = self(x)

        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(dim=1)

        mask = torch.zeros_like(y, dtype=torch.bool)
        if ignore_idx is not None:
            mask = (y == ignore_idx) | (y < -1e9)
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

        ignore_idx = self.hparams.get("ignore_index")
        if ignore_idx is not None:
            mask = (y == ignore_idx) | (y < -1e9)
            y_hat[mask] = ignore_idx

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

        # Create persistent boolean mask for visualization (before denormalization)
        # Use robust check for NoData
        mask_nodata = ((y == ignore_idx) | (y < -1e9)).detach().cpu()  # [B, C, H, W]

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
        # Only compute confusion matrix if there are classification tasks
        has_classification = any(t == "classification" for t in self.task_types)
        if has_classification and hasattr(self, "confusion_matrix"):
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
        task_types: list[str] = None,
        num_classes_per_task: list[int] = None,
        # Deprecated: kept for backward compatibility
        num_seg_classes: int = None,
        num_reg_targets: int = None,
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
        loss_weighting: str = "uncertainty",
        seg_loss_weight: float = 0.5,
        reg_loss_weights: list = None,
        init_log_vars: float = 0.0,
        use_loss_normalization: bool = False,
        loss_norm_momentum: float = 0.9,
        reg_loss: str = "mae",
        ssim_w: float = None,
        sharploss_alpha: float = 0.5,
        use_reg_tanh: bool = False,
        model: str = "MTUNet",
        backbone: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        task_band_names: list[str] = None,
    ):
        """Multi-task UNet for flexible task combinations.

        Supports any mix of classification and regression tasks.

        Args:
            in_channels: Number of input channels.
            task_types: List of task types, e.g., ["classification", "regression"].
                If None, inferred from num_seg_classes/num_reg_targets (deprecated).
            num_classes_per_task: List of class counts per task. For classification,
                this is the number of classes. For regression, this should be 1.
                If None, inferred from num_seg_classes/num_reg_targets (deprecated).
            num_seg_classes: (Deprecated) Number of segmentation classes.
            num_reg_targets: (Deprecated) Number of regression targets.
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
            loss_weighting: Loss weighting strategy ("uncertainty", "fixed", or "equal").
            seg_loss_weight: Weight for segmentation loss when using "fixed" weighting.
            reg_loss_weights: List of weights for each regression channel when using "fixed" weighting.
            init_log_vars: Initial log variances for uncertainty weighting (if used).
            use_loss_normalization: Whether to apply running average normalization to losses before weighting.
            loss_norm_momentum: Momentum for updating running average of losses if normalization is used.
            reg_loss: Loss type for regression ("mae", "sharploss", or "l1ssim").
            ssim_w: Weight for SSIM component in L1SSIMComboLoss if used for regression loss.
            sharploss_alpha: Alpha parameter for SharpLoss if used for regression loss.
            use_reg_tanh: Whether to apply a tanh activation to the regression output.
            model: Model architecture to use ("MTUNet", "ResMTUNet", or "OptimizedMTUNet").
            backbone: ResNet backbone variant for ResMTUNet ("resnet18", "resnet34", "resnet50", "resnet101").
            pretrained: Whether to use ImageNet pretrained weights for ResMTUNet.
            freeze_backbone: Whether to freeze the backbone parameters for transfer learning.
        """
        # Handle backward compatibility: convert deprecated params to new format first
        task_types, num_classes_per_task = self._normalize_task_config(
            task_types, num_classes_per_task, num_seg_classes, num_reg_targets
        )

        # Store the normalized values as the primary parameters
        self.task_types = task_types
        self.num_classes_per_task = num_classes_per_task

        super().__init__()

        # Save hyperparameters, excluding visualization-only params and deprecated ones
        # Note: task_types and num_classes_per_task are now instance attributes
        self.save_hyperparameters(
            ignore=[
                "labels",
                "colormap",
                "num_seg_classes",
                "num_reg_targets",
                "task_types",
                "num_classes_per_task",
                "task_band_names",
            ]
        )

        # Store as instance attributes (not hyperparameters)
        self.labels = labels or {}
        self.colormap = colormap or {}
        self.task_band_names = task_band_names or [f"task_{i}" for i in range(len(task_types))]
        self.validation_step_outputs = []
        # Store for loss logging
        self._last_loss_logs = {}

        # Running average normalization buffers (initialized to 1.0)
        num_tasks = len(task_types)
        for i in range(num_tasks):
            self.register_buffer(f"loss_ema_{i}", torch.tensor(1.0))

    @staticmethod
    def _normalize_task_config(
        task_types, num_classes_per_task, num_seg_classes, num_reg_targets
    ):
        """Normalize task configuration from new or deprecated parameters.

        Args:
            task_types: New parameter - list of task type strings.
            num_classes_per_task: New parameter - list of class counts.
            num_seg_classes: Deprecated parameter.
            num_reg_targets: Deprecated parameter.

        Returns:
            Tuple of (task_types, num_classes_per_task).
        """
        # If new format provided, use it
        if task_types is not None and num_classes_per_task is not None:
            if len(task_types) != len(num_classes_per_task):
                raise ValueError(
                    f"task_types ({len(task_types)} items) and num_classes_per_task "
                    f"({len(num_classes_per_task)} items) must have the same length"
                )
            return task_types, num_classes_per_task

        # If neither format provided, use defaults
        if task_types is None and num_classes_per_task is None:
            if num_seg_classes is None and num_reg_targets is None:
                # Default: single classification task with 14 classes
                return ["classification"], [14]
            # Convert deprecated format
            task_types = []
            num_classes_per_task = []
            if num_seg_classes is not None and num_seg_classes > 0:
                task_types.append("classification")
                num_classes_per_task.append(num_seg_classes)
            if num_reg_targets is not None and num_reg_targets > 0:
                task_types.extend(["regression"] * num_reg_targets)
                num_classes_per_task.extend([1] * num_reg_targets)
            if not task_types:
                raise ValueError("At least one task must be specified")
            return task_types, num_classes_per_task

        # Partial new format - error
        raise ValueError(
            "Must provide both task_types and num_classes_per_task, or use "
            "deprecated num_seg_classes/num_reg_targets (but not both)"
        )

    def _get_total_output_channels(self) -> int:
        """Calculate total number of output channels needed."""
        return sum(self.num_classes_per_task)

    def _get_classification_task_indices(self) -> list[int]:
        """Get indices of classification tasks."""
        return [i for i, t in enumerate(self.task_types) if t == "classification"]

    def _get_regression_task_indices(self) -> list[int]:
        """Get indices of regression tasks."""
        return [i for i, t in enumerate(self.task_types) if t == "regression"]

    def configure_models(self):
        """Initialize the model for multi-task learning.

        Supports MTUNet (standard U-Net), ResMTUNet (ResNet backbone), and
        OptimizedMTUNet (with bottleneck skip connections).
        """
        model_type = self.hparams.get("model", "MTUNet")

        # Calculate total channels needed for each head type
        # seg_channels = sum of all classification task classes
        # reg_channels = number of regression tasks (each has 1 channel)
        seg_channels = sum(
            num_classes
            for task_type, num_classes in zip(
                self.task_types, self.num_classes_per_task
            )
            if task_type == "classification"
        )
        reg_channels = sum(
            1 for task_type in self.task_types if task_type == "regression"
        )

        # MTUNet requires at least 1 seg_channel, even if we only have regression tasks
        # We'll allocate a dummy channel that won't be used
        if seg_channels == 0:
            seg_channels = 1

        if model_type == "MTUNet":
            self.model = MTUNet(
                in_channels=self.hparams["in_channels"],
                seg_channels=seg_channels,
                reg_channels=reg_channels,
                dropout=self.hparams["dropout"],
                use_tanh=False,  # Handle activation in forward
            )
        elif model_type == "ResMTUNet":
            self.model = ResMTUNet(
                in_channels=self.hparams["in_channels"],
                seg_channels=seg_channels,
                reg_channels=reg_channels,
                backbone=self.hparams.get("backbone", "resnet50"),
                pretrained=self.hparams.get("pretrained", True),
                freeze_backbone=self.hparams.get("freeze_backbone", False),
                dropout=self.hparams["dropout"],
                use_tanh=False,
            )
        elif model_type == "OptimizedMTUNet":
            self.model = OptimizedMTUNet(
                in_channels=self.hparams["in_channels"],
                seg_channels=seg_channels,
                reg_channels=reg_channels,
                dropout=self.hparams["dropout"],
                use_tanh=False,
                bottleneck_ratio=self.hparams.get("bottleneck_ratio", 0.5),
            )
        else:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                "Supported models: 'MTUNet', 'ResMTUNet', 'OptimizedMTUNet'"
            )

    def compute_loss(self, y_hat, y, ignore_index=None):
        """Compute multi-task loss with flexible task handling.

        Supports three weighting strategies:
        - "uncertainty": Homoscedastic uncertainty-based learnable weighting
        - "fixed": Fixed manual weights for each task
        - "equal": Equal weighting for all tasks

        Args:
            y_hat: Model predictions tensor of shape [B, total_channels, H, W].
            y: Target tensor of shape [B, num_tasks, H, W].
            ignore_index: Index to ignore in loss computation.

        Returns:
            Total loss tensor.
        """
        if ignore_index is None:
            ignore_index = self.hparams.get("ignore_index")

        task_losses = []
        channel_offset = 0

        # Compute loss for each task
        for task_idx, (task_type, num_classes) in enumerate(
            zip(self.task_types, self.num_classes_per_task)
        ):
            task_pred = y_hat[:, channel_offset : channel_offset + num_classes]
            task_target = y[:, task_idx]

            if task_type == "classification":
                # Classification loss using FocalLoss
                # For classification, we still use exact match for ignore_index as it's usually small positive or -1
                loss = self.focal_loss(task_pred, task_target.long())
            else:  # regression
                # Regression loss - handle single channel
                task_pred = task_pred[:, 0:1]  # [B, 1, H, W]
                task_target = task_target.unsqueeze(1).float()  # [B, 1, H, W]

                # Robust mask for regression
                reg_mask = (task_target == ignore_index) | (task_target < -1e9)

                if isinstance(self.reg_loss_fn, (SharpLoss, L1SSIMComboLoss)):
                    loss = self.reg_loss_fn(task_pred, task_target, reg_mask)
                else:
                    loss_all = self.reg_loss_fn(task_pred, task_target)
                    loss_valid = loss_all[~reg_mask]
                    loss = (
                        loss_valid.mean()
                        if loss_valid.numel() > 0
                        else torch.tensor(0.0, device=y_hat.device)
                    )

            task_losses.append(loss)
            channel_offset += num_classes

        # Store raw losses for logging
        raw_losses = [l.detach() for l in task_losses]

        # Apply running average normalization if enabled
        if self.hparams.get("use_loss_normalization", False):
            task_losses_norm = []
            for i, loss in enumerate(task_losses):
                ema_buffer = getattr(self, f"loss_ema_{i}")
                loss_norm = loss / (ema_buffer + 1e-8)
                task_losses_norm.append(loss_norm)

                # Update EMA
                momentum = self.hparams.get("loss_norm_momentum", 0.9)
                new_ema = momentum * ema_buffer + (1 - momentum) * loss.detach()
                setattr(self, f"loss_ema_{i}", new_ema)

            task_losses = task_losses_norm

        # Apply selected weighting strategy
        loss_weighting = self.hparams.get("loss_weighting", "uncertainty")

        if loss_weighting == "uncertainty":
            total_loss, loss_logs = self.loss_wrapper(task_losses)
        elif loss_weighting == "fixed":
            total_loss, loss_logs = self._compute_fixed_weighted_loss(task_losses)
        elif loss_weighting == "equal":
            total_loss, loss_logs = self._compute_equal_weighted_loss(task_losses)
        elif loss_weighting == "realtime":
            total_loss, loss_logs = self._compute_realtime_weighted_loss(task_losses)
        else:
            raise ValueError(f"Unknown loss_weighting strategy: {loss_weighting}")

        # Add raw losses and EMA values to logs
        for i, raw_loss in enumerate(raw_losses):
            loss_logs[f"task_{i}_raw_loss"] = raw_loss
            loss_logs[f"loss_ema_{i}"] = getattr(self, f"loss_ema_{i}").detach()

        self._last_loss_logs = loss_logs
        return total_loss

    def _compute_fixed_weighted_loss(self, task_losses):
        """Compute weighted loss using fixed manual weights.

        Args:
            task_losses: List of scalar loss tensors, one per task

        Returns:
            tuple containing:
                - total_loss: Weighted sum of all task losses
                - log_dict: Dictionary with individual losses and weights
        """
        num_tasks = len(task_losses)

        # Get fixed weights - if not provided, use equal weights
        fixed_weights = self.hparams.get("fixed_loss_weights", None)

        if fixed_weights is None:
            # Default to equal weighting
            weights = [1.0 / num_tasks] * num_tasks
        else:
            # Use provided weights
            if len(fixed_weights) != num_tasks:
                raise ValueError(
                    f"Number of fixed_loss_weights ({len(fixed_weights)}) must match "
                    f"number of tasks ({num_tasks})"
                )
            # Normalize weights to sum to 1
            total_weight = sum(fixed_weights)
            weights = [w / total_weight for w in fixed_weights]

        # Compute weighted loss
        weighted_losses = []
        log_dict = {}

        for i, (loss, weight) in enumerate(zip(task_losses, weights)):
            weighted_loss = weight * loss
            weighted_losses.append(weighted_loss)

            # Logging
            log_dict[f"task_{i}_raw_loss"] = loss.detach()
            log_dict[f"task_{i}_weight"] = weight

        total_loss = torch.stack(weighted_losses).sum()
        log_dict["total_loss"] = total_loss.detach()

        return total_loss, log_dict

    def _compute_equal_weighted_loss(self, task_losses):
        """Compute weighted loss using equal weights for all tasks.

        Args:
            task_losses: List of scalar loss tensors [seg_loss, reg_loss_0, ...]

        Returns:
            tuple containing:
                - total_loss: Equal-weighted sum of all task losses
                - log_dict: Dictionary with individual losses and weights
        """
        num_tasks = len(task_losses)
        weight = 1.0 / num_tasks

        weighted_losses = []
        log_dict = {}

        for i, loss in enumerate(task_losses):
            weighted_loss = weight * loss
            weighted_losses.append(weighted_loss)

            # Logging
            log_dict[f"task_{i}_raw_loss"] = loss.detach()
            log_dict[f"task_{i}_weight"] = weight

        total_loss = torch.stack(weighted_losses).sum()
        log_dict["total_loss"] = total_loss.detach()

        return total_loss, log_dict

    def _compute_realtime_weighted_loss(self, task_losses):
        """Compute loss using real-time reciprocal normalization.

        Each loss is divided by its own detached value, making the normalized
        loss equal to 1 while preserving gradients scaled by 1/loss_value.
        This provides dynamic gradient balancing without learnable parameters
        or running averages.

        Formula: combined_loss = sum(loss_i / loss_i.detach())

        Args:
            task_losses: List of scalar loss tensors, one per task

        Returns:
            tuple containing:
                - total_loss: Sum of normalized losses
                - log_dict: Dictionary with raw losses and effective weights
        """
        log_dict = {}
        normalized_losses = []

        for i, loss in enumerate(task_losses):
            # Compute effective weight (reciprocal of detached loss)
            effective_weight = 1.0 / (loss.detach() + 1e-8)

            # Normalize: loss / loss.detach() gives normalized loss ~1.0
            normalized_loss = loss / (loss.detach() + 1e-8)
            normalized_losses.append(normalized_loss)

            # Logging
            log_dict[f"task_{i}_raw_loss"] = loss.detach()
            log_dict[f"task_{i}_effective_weight"] = effective_weight

        total_loss = torch.stack(normalized_losses).sum()
        log_dict["total_loss"] = total_loss.detach()

        return total_loss, log_dict

    def configure_losses(self) -> None:
        """Initialize the loss criterion and weighting strategy."""
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
        elif reg_loss_type == "l1ssim":
            ssim_w = self.hparams.get("ssim_w", 0.5)
            l1_w = 1 - ssim_w
            self.reg_loss_fn = L1SSIMComboLoss(w=[l1_w, ssim_w])
        else:
            raise ValueError(
                f"Regression loss type '{reg_loss_type}' is not valid. "
                "Currently, supports 'mae', 'sharploss', or 'ssim'."
            )

        # Initialize loss weighting strategy
        loss_weighting = self.hparams.get("loss_weighting", "uncertainty")

        if loss_weighting == "uncertainty":
            # Homoscedastic uncertainty-based learnable weighting
            # Use actual task_types from config
            self.loss_wrapper = HomoscedasticUncertaintyLoss(
                task_types=self.task_types,
                init_log_vars=self.hparams.get("init_log_vars", 0.0),
            )
        elif loss_weighting in ["fixed", "equal", "realtime"]:
            # Simple weighted loss (fixed, equal, or realtime) - no learnable wrapper needed
            self.loss_wrapper = None
        else:
            raise ValueError(
                f"Unknown loss_weighting strategy: {loss_weighting}. "
                "Supported: 'uncertainty', 'fixed', 'equal', 'realtime'."
            )

    def configure_metrics(self) -> None:
        """Initialize metrics for all tasks dynamically."""
        # Get the maximum number of classes for any classification task
        max_num_classes = 0
        for task_type, num_classes in zip(self.task_types, self.num_classes_per_task):
            if task_type == "classification" and num_classes > max_num_classes:
                max_num_classes = num_classes

        # Classification metrics (shared across all classification tasks)
        if max_num_classes > 0:
            seg_metrics = MetricCollection(
                {
                    "accuracy": Accuracy(
                        task="multiclass",
                        num_classes=max_num_classes,
                        ignore_index=self.hparams["ignore_index"],
                    ),
                    "kappa": CohenKappa(
                        task="multiclass",
                        num_classes=max_num_classes,
                        weights="quadratic",
                        ignore_index=self.hparams["ignore_index"],
                    ),
                    "jaccard": JaccardIndex(
                        task="multiclass",
                        num_classes=max_num_classes,
                        ignore_index=self.hparams["ignore_index"],
                    ),
                }
            )
            self.seg_train_metrics = seg_metrics.clone(prefix="seg_train_")
            self.seg_val_metrics = seg_metrics.clone(prefix="seg_val_")
            self.seg_test_metrics = seg_metrics.clone(prefix="seg_test_")

            # Confusion matrix for validation epoch end (use first classification task)
            self.confusion_matrix = ConfusionMatrix(
                task="multiclass",
                num_classes=max_num_classes,
                ignore_index=self.hparams["ignore_index"],
            )

        # Regression metrics
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

        optimizer = torch.optim.AdamW(
            params,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        # Use ReduceLROnPlateau scheduler (doesn't require steps_per_epoch)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.hparams.get("scheduler_factor", 0.5),
            patience=self.hparams.get("scheduler_patience", 10),
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
        x, y = batch["image"], batch["mask"]
        ignore_idx = self.hparams.get("ignore_index")

        # Forward pass
        y_hat = self(x)

        # Crop target to match output shape
        if y.shape[2:] != y_hat.shape[2:]:
            y = self.crop_to_match(y, y_hat.shape[2:])

        # Compute loss
        loss = self.compute_loss(y_hat, y, ignore_idx)
        self.log("train_loss", loss, on_epoch=True, sync_dist=True)

        # Log uncertainty weights and raw losses
        if hasattr(self, "_last_loss_logs") and self._last_loss_logs:
            for key, value in self._last_loss_logs.items():
                self.log(f"train_{key}", value, on_epoch=True, sync_dist=True)

        # Compute metrics for each task
        self._compute_and_log_metrics(y_hat, y, "train")

        return loss

    def _compute_and_log_metrics(self, y_hat, y, stage):
        """Compute and log metrics for all tasks.

        Args:
            y_hat: Model predictions
            y: Ground truth targets
            stage: One of 'train', 'val', 'test'
        """
        ignore_idx = self.hparams.get("ignore_index")

        # Get metric collections based on stage
        # Segmentation metrics only exist if there are classification tasks
        has_classification = any(t == "classification" for t in self.task_types)
        if stage == "train":
            seg_metrics = getattr(self, "seg_train_metrics", None)
            reg_metrics = self.reg_train_metrics
        elif stage == "val":
            seg_metrics = getattr(self, "seg_val_metrics", None)
            reg_metrics = self.reg_val_metrics
        else:  # test
            seg_metrics = getattr(self, "seg_test_metrics", None)
            reg_metrics = self.reg_test_metrics

        # Process each task
        channel_offset = 0
        for task_idx, (task_type, num_classes) in enumerate(
            zip(self.task_types, self.num_classes_per_task)
        ):
            task_pred = y_hat[:, channel_offset : channel_offset + num_classes]
            task_target = y[:, task_idx]

            if task_type == "classification":
                # Classification metrics
                probs = task_pred.softmax(dim=1)
                pred = torch.argmax(probs, dim=1)

                # Handle ignore_index
                mask = torch.zeros_like(task_target, dtype=torch.bool)
                if ignore_idx is not None:
                    mask = task_target == ignore_idx

                if not mask.all():
                    metrics = seg_metrics(pred[~mask], task_target[~mask])
                    self.log_dict(metrics, on_epoch=True, sync_dist=True)

                # Update confusion matrix for validation
                if stage == "val":
                    self.confusion_matrix.update(pred, task_target)
            else:
                # Regression metrics
                mask = torch.zeros_like(task_target, dtype=torch.bool)
                if ignore_idx is not None:
                    mask = (task_target == ignore_idx) | (task_target < -1e9)

                if not mask.all():
                    # Squeeze channel dim for regression (task_pred is [B, 1, H, W])
                    metrics = reg_metrics(
                        task_pred[:, 0][~mask].flatten(),
                        task_target[~mask].float().flatten(),
                    )
                    self.log_dict(metrics, on_epoch=True, sync_dist=True)

            channel_offset += num_classes

    def validation_step(self, batch, batch_idx):
        """Validation step with flexible task handling."""
        x, y = batch["image"], batch["mask"]
        ignore_idx = self.hparams.get("ignore_index")

        # Forward pass
        y_hat = self(x)

        # Crop target to match output shape
        if y.shape[2:] != y_hat.shape[2:]:
            y = self.crop_to_match(y, y_hat.shape[2:])

        # Compute loss
        loss = self.compute_loss(y_hat, y, ignore_idx)
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)

        # Compute metrics for each task
        self._compute_and_log_metrics(y_hat, y, "val")

        # Convert y_hat from raw output [B, total_channels, H, W] to task format [B, num_tasks, H, W]
        # for visualization. For classification, take argmax. For regression, keep single channel.
        y_hat_tasks = []
        channel_offset = 0
        for task_type, num_classes in zip(self.task_types, self.num_classes_per_task):
            if task_type == "classification":
                # Take argmax to get class predictions [B, H, W] -> [B, 1, H, W]
                task_pred = y_hat[:, channel_offset : channel_offset + num_classes]
                pred_classes = torch.argmax(task_pred, dim=1, keepdim=True)
                y_hat_tasks.append(pred_classes)
            else:  # regression
                # Keep the single regression channel
                y_hat_tasks.append(y_hat[:, channel_offset : channel_offset + 1])
            channel_offset += num_classes

        y_hat_for_viz = torch.cat(y_hat_tasks, dim=1)  # [B, num_tasks, H, W]

        # Store predictions for visualization
        batch["prediction"] = y_hat_for_viz
        self.validation_step_outputs.append(batch)

    def test_step(self, batch, batch_idx):
        """Test step with flexible task handling."""
        x, y = batch["image"], batch["mask"]
        ignore_idx = self.hparams.get("ignore_index")

        # Forward pass
        y_hat = self(x)

        # Crop target to match output shape
        if y.shape[2:] != y_hat.shape[2:]:
            y = self.crop_to_match(y, y_hat.shape[2:])

        # Compute loss
        loss = self.compute_loss(y_hat, y, ignore_idx)
        self.log("test_loss", loss, on_epoch=True, sync_dist=True)

        # Compute metrics for each task
        self._compute_and_log_metrics(y_hat, y, "test")

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        # Only compute confusion matrix if there are classification tasks
        has_classification = any(t == "classification" for t in self.task_types)
        if has_classification and hasattr(self, "confusion_matrix"):
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
        Forward pass through the multi-task U-Net with flexible task handling.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: Output tensor with task-specific activations applied.
                Classification tasks: Raw logits (softmax applied in loss).
                Regression tasks: Tanh activation (if use_reg_tanh=True) or raw values.
        """
        # Get model output - model returns tuple of (seg_logits, reg_out)
        seg_logits, reg_out = self.model(x)

        # Combine into unified tensor based on task configuration
        outputs = []

        # Track which parts of model output we've consumed
        seg_channels_used = 0
        reg_channels_used = 0

        for task_type, num_classes in zip(self.task_types, self.num_classes_per_task):
            if task_type == "classification":
                # Take from segmentation logits
                task_output = seg_logits[
                    :, seg_channels_used : seg_channels_used + num_classes
                ]
                seg_channels_used += num_classes
                outputs.append(task_output)
            else:  # regression
                # Take from regression output
                task_output = reg_out[:, reg_channels_used : reg_channels_used + 1]
                reg_channels_used += 1

                if self.hparams.get("use_reg_tanh", False):
                    # Apply tanh for bounded regression outputs
                    outputs.append(torch.tanh(task_output))
                else:
                    # Raw regression values
                    outputs.append(task_output)

        # Concatenate all task outputs
        return torch.cat(outputs, dim=1)

    def _split_outputs_by_task(self, y_hat):
        """Split model output into task-specific tensors.

        Args:
            y_hat: Model output tensor of shape [B, total_channels, H, W].

        Returns:
            Tuple of (classification_outputs, regression_outputs) where each is
            a list of tensors, one per task of that type.
        """
        classification_outputs = []
        regression_outputs = []
        channel_offset = 0

        for task_type, num_classes in zip(self.task_types, self.num_classes_per_task):
            task_output = y_hat[:, channel_offset : channel_offset + num_classes]

            if task_type == "classification":
                classification_outputs.append(task_output)
            else:  # regression
                regression_outputs.append(task_output)

            channel_offset += num_classes

        return classification_outputs, regression_outputs

    def _plot_input_row(self, axs_row, x, rgb_bands, ignore_idx):
        """Plot input image row."""
        for col_idx, img_tensor in enumerate(x):
            num_channels = img_tensor.shape[0]
            if num_channels >= 3:
                bands = [b for b in rgb_bands if b < num_channels]
                if len(bands) < 3:
                    bands = list(range(min(3, num_channels)))
                img = img_tensor[bands]
                if img.shape[0] == 1:
                    img = torch.stack([img[0]] * 3)
                elif img.shape[0] == 2:
                    img = torch.stack([img[0], img[1], img[0]])
            else:
                img = torch.stack([img_tensor[0]] * 3)

            img = minmax_scaling(img, ignore_idx)
            img = tvF.to_pil_image(img)
            img = tvF.adjust_contrast(img, 2)
            img = tvF.adjust_brightness(img, 1)

            axs_row[col_idx].imshow(np.asarray(img))
            axs_row[col_idx].set_title("input", fontsize="small")
            axs_row[col_idx].axis("off")

    def _plot_segmentation(self, ax, mask, mask_nodata, title):
        """Plot segmentation mask."""
        mask_data = np.round(mask.detach().cpu().numpy()).astype(int)
        colored_mask = np.zeros((*mask_data.shape, 3), dtype=np.uint8)

        if not self.colormap:
            valid_mask = ~mask_nodata
            if valid_mask.any():
                m_min, m_max = mask_data[valid_mask].min(), mask_data[valid_mask].max()
                if m_max > m_min:
                    norm_mask = (mask_data - m_min) / (m_max - m_min)
                else:
                    norm_mask = np.zeros_like(mask_data, dtype=float)
                cmap = plt.get_cmap("tab20")
                colored_mask = (cmap(norm_mask)[..., :3] * 255).astype(np.uint8)
        else:
            for class_id, color in self.colormap.items():
                try:
                    cid = int(class_id)
                except (ValueError, TypeError):
                    cid = class_id
                if isinstance(color, str):
                    color = tuple(
                        int(color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)
                    )
                colored_mask[mask_data == cid] = color

            mapped = np.zeros_like(mask_data, dtype=bool)
            for cid in self.colormap.keys():
                try:
                    cid = int(cid)
                except:
                    pass
                mapped |= mask_data == cid
            colored_mask[(~mapped) & (~mask_nodata)] = (255, 0, 0)

        colored_mask[mask_nodata] = (0, 0, 0)
        ax.imshow(colored_mask)
        ax.set_title(title, fontsize="small")
        ax.axis("off")
        # Add min/mean/max stats for segmentation
        valid_mask = ~mask_nodata
        if valid_mask.any():
            valid_data = mask_data[valid_mask]
            stats_text = f"min:{valid_data.min()} mean:{valid_data.mean():.1f} max:{valid_data.max()} uniq:{len(np.unique(valid_data))}"
        else:
            stats_text = "nodata"
        ax.set_xlabel(stats_text, fontsize="xx-small")

    def _plot_regression(self, ax, img, mask_nodata, vmin, vmax, title):
        """Plot regression image."""
        img_data = img.detach().cpu().numpy()
        img_masked = img_data.copy()
        img_masked[mask_nodata] = np.nan
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(color="black")
        ax.imshow(img_masked, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize="small")
        ax.axis("off")
        if (~mask_nodata).any():
            # Calculate min/mean/max excluding NoData
            valid_data = img_masked[~np.isnan(img_masked)]
            stats_text = f"min:{valid_data.min():.1f} mean:{valid_data.mean():.1f} max:{valid_data.max():.1f}"
            ax.set_xlabel(stats_text, fontsize="xx-small")
        else:
            ax.set_xlabel("nodata", fontsize="xx-small")

    def plot_batch(self, batch, n=10, rgb_bands=[2, 1, 0], max_null_ratio=0.7):
        """Plot a sample of n images from batch for multi-task models.

        Works for any combination of classification and regression tasks.
        Layout: 1 row (input) + 2 rows per task (target + prediction).
        """
        plt.rcParams["savefig.bbox"] = "tight"
        plt.close("all")

        # Stats lookup
        input_stats = self.hparams.get("input_stats")
        target_stats = self.hparams.get("target_stats")
        if input_stats is None and hasattr(self, "trainer"):
            input_stats = getattr(self.trainer.datamodule, "input_stats", None)
        if input_stats is None:
            input_stats = getattr(self, "input_stats", None)
        if target_stats is None and hasattr(self, "trainer"):
            target_stats = getattr(self.trainer.datamodule, "target_stats", None)
        if target_stats is None:
            target_stats = getattr(self, "target_stats", None)

        def revert(tensor, stats, is_target=False):
            if stats is None:
                return tensor
            m, s = stats.get("mean"), stats.get("std")
            if m is None or s is None:
                return tensor
            if isinstance(m, list):
                m = torch.tensor(m)
            if isinstance(s, list):
                s = torch.tensor(s)
            m = m.to(device=tensor.device, dtype=tensor.dtype)
            s = s.to(device=tensor.device, dtype=tensor.dtype)
            num_c = tensor.shape[-3]
            m, s = m[:num_c].clone(), s[:num_c].clone()
            if is_target:
                for i, t in enumerate(self.task_types):
                    if t == "classification" and i < len(m):
                        m[i], s[i] = 0.0, 1.0
            view_shape = [1] * tensor.ndim
            view_shape[-3] = num_c
            return tensor * s.view(*view_shape) + m.view(*view_shape)

        x, y, y_hat = batch["image"], batch["mask"], batch.get("prediction")
        ignore_idx = self.hparams.get("ignore_index")

        if y_hat is not None and y.shape[2:] != y_hat.shape[2:]:
            y = self.crop_to_match(y, y_hat.shape[2:])

        mask_nodata = ((y == ignore_idx) | (y < -1e9)).detach().cpu().numpy()

        # Filter samples by null ratio
        batch_size = x.shape[0]
        valid_sample_indices = []
        for i in range(batch_size):
            sample_null_mask = mask_nodata[i]
            null_ratio = np.sum(sample_null_mask) / sample_null_mask.size
            if null_ratio <= max_null_ratio:
                valid_sample_indices.append(i)
        if not valid_sample_indices:
            valid_sample_indices = list(range(min(5, batch_size)))
        if len(valid_sample_indices) > n:
            valid_sample_indices = valid_sample_indices[:n]

        # Sanitize y_hat
        if y_hat is not None:
            y_hat = y_hat.clone()
            y_hat[(y == ignore_idx) | (y < -1e9)] = float(ignore_idx)

        # Revert normalization
        x_plot = revert(x[valid_sample_indices], input_stats)
        y_plot = revert(y[valid_sample_indices], target_stats, is_target=True)
        if y_hat is not None:
            y_hat_plot = revert(
                y_hat[valid_sample_indices], target_stats, is_target=True
            )

        mask_nodata_plot = mask_nodata[valid_sample_indices]
        actual_n = len(valid_sample_indices)
        num_tasks = len(self.task_types)
        n_rows = 1 + (2 if y_hat is not None else 1) * num_tasks

        fig, axs = plt.subplots(
            nrows=n_rows,
            ncols=actual_n,
            figsize=(4 * actual_n, 3 * n_rows),
            squeeze=False,
        )

        self._plot_input_row(axs[0], x_plot, rgb_bands, ignore_idx)

        for task_idx, task_type in enumerate(self.task_types):
            row_target = 1 + task_idx * (2 if y_hat is not None else 1)
            row_pred = row_target + 1 if y_hat is not None else None

            vmin = vmax = None
            if task_type == "regression" and target_stats:
                vmin = target_stats.get("min", [None] * num_tasks)[task_idx]
                vmax = target_stats.get("max", [None] * num_tasks)[task_idx]

            # Get band name for this task
            band_name = self.task_band_names[task_idx] if task_idx < len(self.task_band_names) else f"task_{task_idx}"

            for col_idx in range(actual_n):
                title_t = f"{band_name}_true"
                title_p = f"{band_name}_pred"

                if task_type == "classification":
                    self._plot_segmentation(
                        axs[row_target, col_idx],
                        y_plot[col_idx, task_idx],
                        mask_nodata_plot[col_idx, task_idx],
                        title_t,
                    )
                    if row_pred is not None:
                        self._plot_segmentation(
                            axs[row_pred, col_idx],
                            y_hat_plot[col_idx, task_idx],
                            mask_nodata_plot[col_idx, task_idx],
                            title_p,
                        )
                else:
                    self._plot_regression(
                        axs[row_target, col_idx],
                        y_plot[col_idx, task_idx],
                        mask_nodata_plot[col_idx, task_idx],
                        vmin,
                        vmax,
                        title_t,
                    )
                    if row_pred is not None:
                        self._plot_regression(
                            axs[row_pred, col_idx],
                            y_hat_plot[col_idx, task_idx],
                            mask_nodata_plot[col_idx, task_idx],
                            vmin,
                            vmax,
                            title_p,
                        )

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
