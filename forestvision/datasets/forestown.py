"""Forest Ownership dataset module.

This module provides the ForestOwnership class for accessing USFS Forest Ownership
data circa 2017, which depicts eight ownership categories of forest land across
the conterminous United States.
"""

from pathlib import Path
from typing import Any, Callable, Iterable

from rasterio.crs import CRS
from torchgeo.datasets import RasterDataset, BoundingBox
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class ForestOwnership(RasterDataset):
    """USFS Forest Ownership Circa 2017.

    This geospatial dataset depicts eight ownership categories of forest land
    across the conterminous United States. The data are modeled from Forest
    Inventory and Analysis (FIA) points from 2012-2017 and the most up-to-date
    publicly available boundaries of federal, state, and tribal lands.

    Attributes:
        _res (int): Internal resolution storage (30 meters).
        is_image (bool): Flag indicating this dataset contains mask data, not image data.
        filename_glob (str): Pattern for matching dataset files.
        nodata (int): NoData value used in the dataset.

    Ownership categories:
        1: Family
        2: Corporate
        3: TIMO/REIT
        4: Other Private
        5: Federal
        6: State
        7: Local
        8: Tribal

    Citation:
        Sass, Emma S., Brett J. Butler, and Marla Markowski-Lindsay. 2020.
        "Estimated Distribution of Forest Ownership across the Conterminous United
        States - Geospatial Database." U.S. Department of Agriculture, Forest Service,
        Northern Research Station. https://doi.org/10.2737/nrs-rmap-11.

    Spatial Resolution: 30 meters
    Source CRS (EPSG): 6269
    """

    _res = 30
    is_image = False
    filename_glob = "*_cog.tif"
    nodata = 0

    def __init__(
        self,
        paths: Path | Iterable[Path] = "data/datasets/forest_own1",
        crs: CRS | None = None,
        res: float | None = 30,
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        cache: bool = False,
    ) -> None:
        """Initialize ForestOwnership dataset.

        Args:
            paths (Path | Iterable[Path]): A directory containing the dataset files
                or a list of paths. Defaults to "data/datasets/forest_own1".
            crs (CRS, optional): Optional coordinate reference system to reproject
                the dataset to.
            res (float, optional): Optional resolution to resample the dataset to.
                Defaults to 30 meters.
            transforms (Callable, optional): Optional function/transform to apply
                to each sample.
            cache (bool, optional): Flag indicating whether to cache the dataset
                in memory. Defaults to False.
        """
        self.paths = paths
        if res:
            self._res = res
        super().__init__(paths, crs, res, transforms=transforms, cache=cache)

    def plot(
        self,
        sample: dict[str, Any],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample (dict[str, Any]): A sample returned by :meth:`RasterDataset.__getitem__`
                containing at least a 'mask' key, and optionally a 'prediction' key.
            show_titles (bool, optional): Flag indicating whether to show titles
                above each panel. Defaults to True.
            suptitle (str, optional): Optional string to use as a suptitle for
                the entire figure.

        Returns:
            Figure: A matplotlib Figure with the rendered sample.

        Note:
            If the sample contains a 'prediction' key, both mask and prediction
            will be plotted side by side. Otherwise, only the mask is displayed.
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
