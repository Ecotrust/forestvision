import os
import hashlib
from pathlib import Path

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
from rasterio.profiles import DefaultGTiffProfile
from rasterio.windows import Window
from rasterio.crs import CRS
import rasterio

from forestvision.datasets.utils import save_cog
from forestvision.transforms import Denormalize


class AGBPredictionSaver(BasePredictionWriter):
    """Class to save AGB predictions using Lightning distributed inference."""

    def __init__(
        self,
        output_dir: str | Path,
        write_interval: str = "batch",
        crs: CRS = None,
        masked: bool = False,
        crop: int = 0,
        overwrite: bool = False,
    ):
        """
        Args:
            output_dir (str): Directory to save predictions.
            write_interval (int): Interval to save predictions.
            masked (bool): Apply mask to predictions
            crop (int): Distance to crop from the edges of the prediction in map units.

        Returns:
            None
        """
        super().__init__(write_interval)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.masked = masked
        self.crop = crop
        self.crs = crs
        self.overwrite = overwrite

    # def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
    #     profile = DefaultGTiffProfile(count=1, dtype="uint8")
    #     pass

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ) -> None:
        profile = DefaultGTiffProfile(count=1, dtype="uint16")
        masks = batch.get("mask", None)
        for idx, pred in enumerate(prediction):
            bbox = batch_indices[idx]
            minx, maxx, miny, maxy, _, _ = bbox
            pred_id = hashlib.md5(
                f"({minx}, {miny}, {maxx}, {maxy})".encode()
            ).hexdigest()

            pred = trainer.datamodule.revert_target(pred)
            pred[pred < 0] = 65535

            if (masks is not None) and self.masked:
                mask = masks[idx]
                if mask.ndim < 3:
                    mask = mask.squeeze(dim=0)
                pred = pred * mask

            filepath = os.path.join(
                self.output_dir, f"{pred_id}_{pl_module.__class__.__name__}.tif"
            )
            pred = pred.cpu().numpy()

            width = pred.shape[-1]
            height = pred.shape[-2]
            profile.update(
                width=width,
                height=height,
                transform=rasterio.transform.from_bounds(
                    bbox.minx,
                    bbox.miny,
                    bbox.maxx,
                    bbox.maxy,
                    width=width,
                    height=height,
                ),
                crs=self.crs,
                nodata=65535,
            )
            save_cog(
                pred,
                profile,
                filepath,
                overwrite=self.overwrite,
                window=Window(
                    self.crop, self.crop, width - self.crop * 2, height - self.crop * 2
                ),
            )
