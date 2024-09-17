import os
from pathlib import Path
from typing import Any, Callable, Iterable
import re

import torch
from rasterio.crs import CRS
import torchvision.transforms.functional as tvF
from torchgeo.datasets import RasterDataset, BoundingBox
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class Sentinel2SR(RasterDataset):
    """Sentinel2 SR

    crs = EPSG:4326
    res = 8.983152841196101e-05 (10m)
    Downloaded from Earth Engine, collection COPERNICUS/S2_SR (not harmonized)
    """

    _res = 8.983152841196101e-05
    is_image = True
    # Using a single vrt file is faster when image data is stored in multiple files
    filename_glob = "*.vrt"
    nodata = 0
    instrument = "Sentinel 2 MSI"
    rgb_bands = ["B4", "B3", "B2"]

    # These are the bands available in the source images
    all_bands = [
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B11",
        "B12",
    ]

    def __init__(
        self,
        paths: Path | Iterable[Path] = "data/geodatasets/sentinel2sr",
        year: int | None = None,
        crs: CRS | None = CRS.from_epsg(5070),
        res: float | None = 10,
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        cache: bool = False,
    ) -> None:
        """Initialize edataset.

        Args:
            paths: str or list of str:
                A directory containing the dataset files or a list of paths.
            year: int
                Optional year to filter the dataset.
            crs: CRS
                Optional CRS to reproject the dataset.
            res: float
                An optional resolution to resample the dataset.
            transforms: Callable
                An optional function to apply to each sample.
            cache: bool
                Flag indicating whether to cache the dataset in memory.
        """
        self.paths = paths
        if res:
            self._res = res
        if year:
            self.filename_regex = rf".*{year}"
        # bands =
        super().__init__(paths, crs, res, transforms=transforms, cache=cache)

    def plot(
        self,
        sample: dict[str, Any],
        show_titles: bool = True,
        suptitle: str | None = None,
        contrast: float = 1,
        brightness: float = 1,
        denormalizer: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`RasterDataset.__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """

        k = "image" if self.is_image else "mask"
        image = sample[k].squeeze()
        if self.rgb_bands:
            if denormalizer:
                image = denormalizer(image)
            rgb_bands_idx = [self.all_bands.index(b) for b in self.rgb_bands]
            image = image[rgb_bands_idx]
            image = tvF.to_pil_image(image)
            image = tvF.adjust_contrast(image, contrast)
            image = tvF.adjust_brightness(image, brightness)

        ncols = 1

        showing_predictions = "prediction" in sample
        if showing_predictions:
            pred = sample["prediction"].squeeze()
            ncols = 2

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 4, 4))

        if showing_predictions:
            axs[0].imshow(image)
            axs[0].axis("off")
            axs[1].imshow(pred)
            axs[1].axis("off")
            if show_titles:
                axs[0].set_title(self.instrument)
                axs[1].set_title("Prediction")
        else:
            axs.imshow(image)
            axs.axis("off")
            if show_titles:
                axs.set_title(self.instrument)

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchgeo.datasets import stack_samples
    from torchgeo.samplers import PreChippedGeoSampler
    from torchgeo.datasets import BoundingBox

    # coords(minx, maxx, miny, maxy, mint, maxt)
    coords = (
        -122.76214976026239,
        -122.67528267228803,
        42.11589512843235,
        42.19656384094628,
        0,
        9.223372036854776e18,
    )
    coords = (-1849185, -1846785, 2434815, 2437215, 0, 9.223372036854776e18)
    bbox = BoundingBox(*coords)
    s2 = Sentinel2SR(year=2017, cache=False, res=10)
    fig = s2.plot(s2.__getitem__(bbox))
    fig.savefig("sentinel2sr_test.png")
