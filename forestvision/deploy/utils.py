from pathlib import Path
from typing import Any, Callable, Iterable

from rasterio.crs import CRS
from torchgeo.datasets import RasterDataset, BoundingBox
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


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
        super().__init__(paths, crs, res, transforms=transforms, cache=cache)
        self.bands = bands

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
