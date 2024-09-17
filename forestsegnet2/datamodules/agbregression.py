"""
"""

import inspect
from typing import Any

import ee
import kornia as K
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from kornia.constants import DataKey, Resample
from kornia.enhance import Denormalize
from matplotlib.figure import Figure
from torchgeo.samplers import RandomGeoSampler
from torchgeo.datasets import BoundingBox, stack_samples
from torchgeo.samplers.utils import _to_tuple
from torchgeo.transforms import AugmentationSequential
from torchgeo.datamodules import GeoDataModule
from torchvision.transforms import v2
from torchvision.transforms import functional as tvF
from geopandas import GeoDataFrame
import numpy as np

# from ..models import UNet
from forestsegnet2.datasets import (
    eMapRAGB,
    GEELandsat8,
    GEESentinel2,
    DatasetStats,
    ReplaceNodataVal,
    Normalize,
    # Denormalize,
    minmax_scaling,
)
from forestsegnet2.samplers import TileGeoSampler

# Suppress rasterio errors while reading geotiffs
# see https://stackoverflow.com/a/74136171/1913361
from osgeo import gdal

gdal.PushErrorHandler("CPLQuietErrorHandler")


class AGBRegressionDataModule(GeoDataModule):
    """LightningDataModule implementation to predict AGB."""

    input_datasets = {
        "landsat8": GEELandsat8,
        "sentinel2": GEESentinel2,
    }

    mask_datasets = {"agb": eMapRAGB}

    def __init__(
        self,
        tiles: GeoDataFrame,
        batch_size: int = 30,
        patch_size: int | tuple[int, int] = 128,
        length: int | None = None,
        num_workers: int = 10,
        mask_args: dict[str, Any] = None,
        input_args: dict[str, Any] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a new Landsat8AGBDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            patch_size: Size of each patch, either ``size`` or ``(height, width)``.
            length: Length of each training epoch.
            num_workers: Number of workers for parallel data loading.
            mask_args: Keyword arguments passed to the AGB dataset passed as a dictionary
            {"dataset_key": {arg1:val1, arg2:val2, ...}}. Valid keys are "agb".
            input_args: Keyword arguments passed to the Images dataset passed as a dictionary
            {"dataset_key": {arg1:val1, arg2:val2, ...}}. Valid keys are "landsat8" and "sentinel2".
            **kwargs: Additional keyword arguments passed to ... ?
        """
        self.tiles = tiles
        minx, miny, maxx, maxy = self.tiles.total_bounds
        self.roi = BoundingBox(minx, maxx, miny, maxy, 0, 0)
        (self.train_tiles, self.val_tiles) = train_test_split(
            self.tiles, test_size=0.2, random_state=0
        )

        mask_dataset_key, mask_dataset_args = list(mask_args.items()).pop()
        input_dataset_key, input_dataset_args = list(input_args.items()).pop()

        input_class = self.input_datasets[input_dataset_key]
        mask_class = self.mask_datasets[mask_dataset_key]

        if ("roi" not in mask_dataset_args.keys()) and (
            "roi" in list(inspect.signature(mask_class).parameters)
        ):
            mask_dataset_args.update(roi=self.roi)
        if "tiles" in list(inspect.signature(mask_class).parameters):
            mask_dataset_args.update(tiles=None)
        if ("roi" not in input_dataset_args.keys()) and (
            "roi" in list(inspect.signature(input_class).parameters)
        ):
            input_dataset_args.update(roi=self.roi)
        if "tiles" in list(inspect.signature(input_class).parameters):
            input_dataset_args.update(tiles=None)

        input_args = input_dataset_args
        mask_args = mask_dataset_args

        self.input_dataset = input_class(**input_dataset_args)
        self.mask_dataset = mask_class(**mask_dataset_args)
        self.rgb_bands = torch.tensor([3, 2, 1])

        # pass arbitraty GeoDataset class to get around GeoDataModule's `dataset_class`
        # required argument. It wont be used.
        super().__init__(input_class, batch_size, patch_size, length, num_workers)

        # overwrite collate_fn
        self.collate_fn = self._collate_fn

        # Turn off augmentations for now
        # self.train_aug = AugmentationSequential(
        #     K.Normalize(mean=self.mean, std=self.std),
        #     K.RandomResizedCrop(_to_tuple(self.patch_size), scale=(0.6, 1.0)),
        #     K.RandomVerticalFlip(p=0.5),
        #     K.RandomHorizontalFlip(p=0.5),
        #     data_keys=["image", "mask"],
        #     extra_args={
        #         DataKey.MASK: {"resample": Resample.NEAREST, "align_corners": None}
        #     },
        # )

        # self.aug = AugmentationSequential(
        #     K.Normalize(mean=self.mean, std=self.std), data_keys=["image", "mask"]
        # )

    def _collate_fn(self, batch):
        new_batch = stack_samples(batch)
        # prep mask
        return {
            "mask": new_batch["mask"],
            "image": new_batch["image"],
            "crs": new_batch["crs"],
            "bbox": new_batch["bbox"],
        }

    def setup_transforms(self):
        self.mask_dataset.transforms = v2.Compose(
            [
                ReplaceNodataVal(self.mask_dataset.nodata, 0),
                # v2.ConvertImageDtype(torch.float32),
                Normalize(
                    mean=self.mask_stats["mean"],
                    std=self.mask_stats["std"],
                    on_key="mask",
                ),
                v2.Resize(self.patch_size),
            ]
        )
        self.input_dataset.transforms = v2.Compose(
            [
                v2.Normalize(
                    mean=self.input_stats["mean"], std=self.input_stats["std"]
                ),
                v2.Resize(self.patch_size),
            ]
        )
        self.denormalizer = Denormalize(
            mean=self.input_stats["mean"], std=self.input_stats["std"]
        )
        self.mask_denormalizer = Denormalize(
            mean=self.mask_stats["mean"], std=self.mask_stats["std"]
        )

    def prepare_data(self) -> None:
        """Prepare data.

        Method inherited from lightning DataHooks via LightningDataModule.
        """
        ee.Initialize()

        # Compute stats. If stats already exist, they will be loaded.
        sampler = TileGeoSampler(self.input_dataset, tiles=self.tiles)
        input_stats = DatasetStats(
            self.input_dataset,
            sampler,
            batch_size=1,
            num_workers=20,
            channels=len(self.input_dataset.bands) or 1,
            nodata=self.input_dataset.nodata,
        )

        sampler = TileGeoSampler(self.mask_dataset, tiles=self.tiles)
        mask_stats = DatasetStats(
            self.mask_dataset,
            sampler,
            batch_size=1,
            num_workers=30,  #
            channels=len(self.mask_dataset.bands) or 1,
            nodata=self.mask_dataset.nodata,
        )

        self.input_stats = input_stats.compute()
        self.mask_stats = mask_stats.compute()

    def setup(self, stage: str) -> None:
        """Set up datasets and samplers.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        self.setup_transforms()

        self.dataset = self.mask_dataset & self.input_dataset
        if stage in ["fit"]:
            self.train_sampler = TileGeoSampler(self.dataset, self.train_tiles)
        if stage in ["fit", "validate"]:
            self.val_sampler = TileGeoSampler(self.dataset, self.val_tiles)
        if stage in ["test"]:
            self.test_sampler = RandomGeoSampler(self.dataset, self.patch_size, 100)

    def plot(self, batch: dict, n=5) -> Figure:
        """Run AGB plot method."""
        plt.rcParams["savefig.bbox"] = "tight"

        x, y = batch["image"], batch["mask"].float()
        mask = y == self.mask_dataset.nodata
        predictions = batch.get("prediction", None)
        if self.denormalizer:
            x = self.denormalizer(x)
        if self.mask_denormalizer:
            y = self.mask_denormalizer(y)

        sample_dict = {
            "x": x[:n],
            "y": y[:n],
        }

        if predictions is not None:
            if self.mask_denormalizer:
                predictions = self.mask_denormalizer(predictions)
            sample_dict["y_hat"] = predictions[:n]

        num_rows = len(sample_dict)
        num_cols = len(sample_dict["x"])
        fig, axs = plt.subplots(
            figsize=(8, 5), nrows=num_rows, ncols=num_cols, squeeze=False
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
                    axs[row_idx, i].imshow(np.asarray(img), cmap="viridis")
                    axs[row_idx, i].set_title(k)
                    axs[row_idx, i].axis("off")

            row_idx += 1

        plt.tight_layout()
        return fig


if __name__ == "__main__":

    import ee
    import geopandas as gpd

    ee.Initialize()
    tiles = gpd.read_file("data/vector/tiles_80x80.geojson")
    input_args = {"landsat8": dict(year=2018, path="data/training/geelandsat8")}
    mask_args = {"agb": dict(year=2018)}
    dm = AGBRegressionDataModule(
        tiles,
        batch_size=10,
        patch_size=128,
        num_workers=0,
        mask_args=mask_args,
        input_args=input_args,
    )
    dm.prepare_data()
    dm.setup("fit")

    for batch in dm.train_dataloader():
        print(batch.keys())
        fig = dm.plot(batch)
        fig.savefig("test.png")
        break
