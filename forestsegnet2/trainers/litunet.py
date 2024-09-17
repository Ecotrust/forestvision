import torch
import torch.nn as nn
import torchvision.transforms.functional as tvF

from torchmetrics import R2Score
import lightning as pl
import numpy as np
import matplotlib.pyplot as plt

from ..models import UNet as UNet
from ..datasets import minmax_scaling


class LitUNet(pl.LightningModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        ignore_index=None,
        revert_transforms=None,
        mask_denormalize=None,
    ):
        super(LitUNet, self).__init__()
        self.model = UNet(in_channels, out_channels, dropout=True)
        self.ignore_index = ignore_index
        self.r_squared = R2Score()
        self.revert_transforms = revert_transforms
        self.mask_denormalize = mask_denormalize
        self.rgb_bands = torch.tensor([3, 2, 1])

    def loss_function(self, y_hat, y, mask=None):
        # loss = nn.MSELoss(reduction="none")(y_hat, y)
        loss = nn.L1Loss(reduction="none")(y_hat, y)
        if mask is not None:
            loss = loss[~mask]
        return loss.mean()

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["mask"].float()
        if y.dim() < 4:
            y = y.unsqueeze(1)

        mask = torch.zeros_like(y, dtype=torch.bool)
        if self.ignore_index is not None:
            mask = y == self.ignore_index
        y_hat = self(x)
        loss = self.loss_function(y_hat, y, mask)
        acc = self.r_squared(y_hat[~mask].flatten(), y[~mask].flatten())
        self.log_dict(
            {"loss": loss, "acc": acc},
            sync_dist=True,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
        )
        batch["prediction"] = y_hat
        self.training_step_outputs = batch
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["mask"].float()
        if y.dim() < 4:
            y = y.unsqueeze(1)
        mask = torch.zeros_like(y, dtype=torch.bool)
        if self.ignore_index is not None:
            mask = y == self.ignore_index

        y_hat = self(x)
        val_loss = self.loss_function(y_hat, y, mask)
        # acc = self.r_squared(y_hat[~mask].flatten(), y[~mask].flatten())
        # self.log_dict(
        #     {"val_loss": val_loss, "val_acc": acc},
        #     sync_dist=True,
        #     prog_bar=True,
        #     # on_epoch=True,
        # )
        self.log(
            "val_loss",
            val_loss,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        if batch_idx == 0:
            batch["prediction"] = y_hat
            fig = self.batch_grid(batch)
            self.logger.experiment.add_figure("val_images", fig, self.global_step)

    def on_train_epoch_end(self):
        fig = self.batch_grid(self.training_step_outputs)
        self.logger.experiment.add_figure("train_images", fig, self.global_step)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def batch_grid(self, batch, n=5):
        """Run AGB plot method."""
        plt.rcParams["savefig.bbox"] = "tight"

        x, y = batch["image"], batch["mask"].float()
        mask = y == self.ignore_index
        predictions = batch.get("prediction", None)
        if self.revert_transforms:
            x = self.revert_transforms(x)
        if self.mask_denormalize:
            y = self.mask_denormalize(y)

        sample_dict = {
            "x": x[:n],
            "y": y[:n],
        }

        if predictions is not None:
            if self.mask_denormalize:
                predictions = self.mask_denormalize(predictions)
            sample_dict["y_hat"] = predictions[:n]

        num_rows = len(sample_dict)
        num_cols = len(sample_dict["x"])
        fig, axs = plt.subplots(
            figsize=(7, 5), nrows=num_rows, ncols=num_cols, squeeze=False
        )
        row_idx = 0
        for k, item in sample_dict.items():
            if k == "x":
                item = item[:, self.rgb_bands]
                for i, img in enumerate(item):
                    img = minmax_scaling(img)
                    img = tvF.to_pil_image(img)
                    # img = tvF.adjust_contrast(img, 3)
                    # img = tvF.adjust_brightness(img, 3)
                    axs[0, i].imshow(np.asarray(img))
                    axs[0, i].set_title("Input")
                    axs[0, i].axis("off")

            else:
                for i, img in enumerate(item):
                    img = img.squeeze().clone().detach()
                    msk = mask[i].squeeze()
                    img[msk == True] = 0
                    # img = tvF.to_pil_image(img)
                    axs[row_idx, i].imshow(
                        np.asarray(img.detach().cpu()), cmap="viridis"
                    )
                    axs[row_idx, i].set_title(k)
                    axs[row_idx, i].axis("off")

            row_idx += 1

        plt.tight_layout()
        return fig

    def forward(self, x):
        return self.model(x)
