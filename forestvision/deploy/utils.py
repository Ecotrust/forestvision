from pathlib import Path
from typing import Any, Callable, Iterable

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from rasterio.crs import CRS

import torch
from torchgeo.datasets import RasterDataset
import torchvision.transforms.functional as tvF

from ..datasets.utils import minmax_scaling


class AGBPredictions(RasterDataset):
    """eMapR Aboveground Biomass predictions."""

    _res = 30
    is_image = False
    filename_glob = "*.tif"
    nodata = None

    def __init__(
        self,
        paths: Path | Iterable[Path] = "data/predict/biomass",
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


class AnyRasterDataset(RasterDataset):
    """Load any raster file collection from disk."""

    _res = 30  # Default resolution

    def __init__(
        self,
        paths: Path | Iterable[Path] = None,
        glob: str = "*.tif",
        nodata: int = None,
        crs: CRS = None,
        res: float = None,
        bands: list[int] | None = None,
        is_image: bool = False,
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        cache: bool = False,
    ) -> None:
        """Initialize AnyRasterDataset instance.

        Args:
            paths: str or list of str:
                A directory containing the dataset files or a list of paths.
            glob: str
                A glob pattern to match files in the directory.
            nodata: int
                The nodata value for the dataset.
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
        self.filename_glob = glob
        self.nodata = nodata
        self.is_image = is_image
        if res:
            self._res = res
        super().__init__(
            paths, crs, res, bands=bands, transforms=transforms, cache=cache
        )
        # self.bands = bands

    @property
    def res(self) -> float:
        """Get the resolution of the dataset.

        Returns:
            float: Resolution in meters per pixel.
        """
        # Handle both single float and tuple cases
        if hasattr(self, "_res") and self._res is not None:
            if isinstance(self._res, tuple):
                # Return the x resolution (first element of tuple)
                return float(self._res[0])
            return float(self._res)
        return 30.0

    @res.setter
    def res(self, value: float) -> None:
        """Set the resolution of the dataset.

        Args:
            value (float): Resolution in meters per pixel.
        """
        self._res = value

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
            sample (dict[str, Any]): Sample returned by RasterDataset.__getitem__
            show_titles (bool): Whether to show titles above each panel
            suptitle (str | None): Optional text to use as a suptitle
            contrast (float): Contrast adjustment
            brightness (float): Brightness adjustment
            denormalizer (Callable[[torch.Tensor], torch.Tensor] | None): Optional function to denormalize the image

        Returns:
            Figure: Matplotlib Figure with the rendered sample
        """
        cmap = None
        norm = None
        if self._cmap:
            cmap, norm = self._get_cmap()

        k = "image" if self.is_image else "mask"
        image = sample[k].squeeze()
        # mask = image == self.nodata
        if self.rgb_bands and self.bands:
            if denormalizer:
                image = denormalizer(image)

            image = minmax_scaling(image, self.nodata)
            rgb_bands_idx = [self.bands.index(b) for b in self.rgb_bands]
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
        title = (
            f"{self.instrument}\nRGB: {', '.join([b[-1] for b in self.rgb_bands])}"
            if k == "image"
            else self.instrument
        )

        if showing_predictions:
            axs[0].imshow(image, cmap=cmap, norm=norm)
            axs[0].axis("off")
            axs[1].imshow(pred, cmap=cmap, norm=norm)
            axs[1].axis("off")
            if show_titles:
                axs[0].set_title(title)
                axs[1].set_title("Prediction")
        else:
            axs.imshow(image, cmap=cmap, norm=norm)
            axs.axis("off")
            if show_titles:
                axs.set_title(title)

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
