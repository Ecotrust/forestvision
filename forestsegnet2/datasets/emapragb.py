"""
"""

import os
from pathlib import Path
from typing import Any, Callable, Iterable, cast
import re

import torch
from torch import Tensor
from rasterio.crs import CRS
from torchgeo.datasets import RasterDataset, BoundingBox
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class eMapRAGB(RasterDataset):
    """eMapR Aboveground Biomass dataset."""

    _res = 30
    is_image = False
    filename_glob = "*_cog.tif"
    nodata = -32768

    def __init__(
        self,
        paths: Path | Iterable[Path] = "data/geodatasets/emapr/biomass",
        year: int | None = None,
        crs: CRS | None = None,
        res: float | None = 30,
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        cache: bool = False,
    ) -> None:
        """Initialize eMapR Biomass dataset.

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
        super().__init__(paths, crs, res, transforms=transforms, cache=cache)

    def plot(
        self,
        sample: dict[str, Any],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`RasterDataset.__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        mask = sample["mask"].squeeze()
        ncols = 1

        showing_predictions = "prediction" in sample
        if showing_predictions:
            pred = sample["prediction"].squeeze()
            ncols = 2

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 4, 4))

        if showing_predictions:
            axs[0].imshow(mask)
            axs[0].axis("off")
            axs[1].imshow(pred)
            axs[1].axis("off")
            if show_titles:
                axs[0].set_title("Mask")
                axs[1].set_title("Prediction")
        else:
            axs.imshow(mask)
            axs.axis("off")
            if show_titles:
                axs.set_title("Mask")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchgeo.datasets import stack_samples
    from torchgeo.samplers import PreChippedGeoSampler
    from torchgeo.datasets import BoundingBox

    # coords(minx, maxx, miny, maxy, mint, maxt)
    coords = (-1863585, -1861185, 3046815, 3049215, 0, 9.223372036854776e18)
    bbox = BoundingBox(*coords)
    biomass = eMapRAGB(year=2018)
    fig = biomass.plot(biomass.__getitem__(bbox))
    plt.show()
