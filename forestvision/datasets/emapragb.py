"""eMapR Aboveground Biomass dataset module.

This module provides the eMapRAGB class for accessing eMapR Aboveground Biomass data,
which contains biomass estimates across the Contiguous United States (CONUS) from
1990 to 2018. The data combines annually composited satellite imagery, field
observations, and LiDAR data.

To use this dataset, download the data from the `eMapR website <https://emapr.ceoas.oregonstate.edu/metadata.html?para1=lt-stem_biomass_nbcd_v0.1_median,>`_
and place it in the `data/datasets/emapr` directory or specify a different path when initializing
the dataset.

Citation:
    Hooper, S., & Kennedy, R. E. (2018). A spatial ensemble approach for
    broad-area mapping of land surface properties. Remote Sensing of Environment,
    210, 473-489. https://doi.org/10.1016/j.rse.2018.03.032

"""

from pathlib import Path
from typing import Any, Callable, Iterable

from rasterio.crs import CRS
from torchgeo.datasets import RasterDataset, BoundingBox
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class eMapRAGB(RasterDataset):
    """eMapR Aboveground Biomass Dataset.

    Biomass estimates across the Contiguous United States (CONUS) from 1990 to 2018.
    The data are a product of annually composited satellite imagery, field observations,
    and LiDAR data.

    Attributes:
        _res (int): Internal resolution storage (30 meters).
        is_image (bool): Flag indicating this dataset contains mask data, not image data.
        filename_glob (str): Pattern for matching dataset files.
        nodata (int): NoData value used in the dataset.

    Units: Mg/ha (Megagrams per hectare)
    Spatial Resolution: 30 meters
    Source CRS (EPSG): 5070
    """

    _res = 30

    is_image = False

    filename_glob = "*_cog.tif"

    nodata = -32768

    def __init__(
        self,
        paths: Path | Iterable[Path] = "data/datasets/emapr",
        year: int | None = None,
        crs: CRS | None = None,
        res: float | None = 30,
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        cache: bool = False,
    ) -> None:
        """Initialize eMapR Biomass dataset.

        Args:
            paths (Path | Iterable[Path]): A directory containing the dataset files
                or a list of paths. Defaults to "data/datasets/emapr".
            year (int, optional): Optional year to filter the dataset. If provided,
                sets filename regex to match files containing the year.
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
