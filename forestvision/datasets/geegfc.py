"""Global Forest Change dataset module for Google Earth Engine integration.

This module provides the GEEGlobalForestChange class for accessing UMD Global
Forest Change data (2000-2023) from Google Earth Engine, which includes forest
cover change, loss, gain, and related metrics.
"""

from typing import Any, Callable, Dict, Optional, Union
import warnings
import ee
import ee.deprecation
import pystac
import geopandas
from rasterio.crs import CRS
from torchgeo.datasets import BoundingBox

from .geebase import GEERasterDataset


class GEEGlobalForestChange(GEERasterDataset):
    """UMD Global Forest Change 2000 - 2023, Earth Engine Collection.

    This dataset provides access to University of Maryland's Global Forest Change
    data from Google Earth Engine, including forest cover change metrics from
    2000 to 2023.

    Attributes:
        filename_glob (str): File pattern for matching files.
        gee_asset_id (str): Earth Engine asset ID for the collection.
        all_bands (List[str]): List of all available dataset bands.
        rgb_bands (List[str]): Bands to use for RGB visualization (empty for this dataset).
        nodata (int): NoData value for the dataset.
        is_image (bool): Flag indicating this dataset contains mask data, not image data.
        instrument (str): Name of the dataset/instrument.

    Dataset Bands:
        treecover2000 - tree cover circa 2000 (t1), range [0, 100]
        loss          - loss of tree cover (2000 - 2023), values 0 or 1 (loss)
        gain          - gain of tree cover (2000 - 2012), values 0 or 1 (gain)
        lossyear      - year of loss, range [0,23]. 0 is 2000 and 23 is 2023
        first_b30     - landsat band 3 t1
        first_b40     - landsat band 4 t1
        first_b50     - landsat band 5 t1
        first_b70     - landsat band 6 t1
        last_b30      - landsat band 3 t2
        last_b40      - landsat band 4 t2
        last_b50      - landsat band 5 t2
        last_b70      - landsat band 6 t2
        datamask      - data mask with nodata, land surface, and water based on 2000 - 2012 images.

    Citation:
        Hansen, M. C., P. V. Potapov, R. Moore, M. Hancher, S. A. Turubanova, A. Tyukavina, D. Thau, et al. 2013.
        "High-Resolution Global Maps of 21st-Century Forest Cover Change." Science. American Association for
        the Advancement of Science (AAAS). https://doi.org/10.1126/science.1244693.

    Reference:
        https://glad.earthengine.app/view/global-forest-change

    Spatial Resolution: 1 arc-second per pixel (~30 meters)
    """

    filename_glob = "*.tif"

    gee_asset_id = "UMD/hansen/global_forest_change_2023_v1_11"

    all_bands = [
        "treecover2000",
        "loss",
        "gain",
        "lossyear",
        "first_b30",
        "first_b40",
        "first_b50",
        "first_b70",
        "last_b30",
        "last_b40",
        "last_b50",
        "last_b70",
        "datamask",
    ]

    rgb_bands = []

    nodata = 0

    is_image = False

    instrument = "UMD Global Forest Change"

    def __init__(
        self,
        roi: Optional[BoundingBox] = None,
        res: float = 30,
        bands: Optional[list] = None,
        path: Optional[str] = None,
        crs: Optional[CRS] = CRS.from_epsg(5070),
        transforms: Callable[[Dict[str, Any]], Dict[str, Any]] | None = None,
        download: bool = False,
        overwrite: bool = False,
        cache: bool = True,
    ) -> None:
        """Initialize a GEEGlobalForestChange dataset instance.

        Args:
            roi (BoundingBox, optional): Region of interest to fetch data from.
            res (float, optional): Resolution of the dataset in meters. Defaults to 30.
            bands (list, optional): List of bands to select from the dataset. If not provided,
                defaults to ["treecover2000", "gain", "loss", "lossyear"].
            path (str, optional): Directory where data are stored or will be stored if download
                is True. If path is provided and a matching file exists, the image will be
                loaded from that file unless overwrite is True.
            crs (CRS, optional): Coordinate Reference System for fetching images from
                Earth Engine. Defaults to EPSG:5070.
            transforms (Callable, optional): Function/transform that takes in a sample
                and returns a transformed version.
            download (bool, optional): If True, download the dataset to the path directory.
                Defaults to False.
            overwrite (bool, optional): If True, overwrite the dataset if it already exists.
                Defaults to False.
            cache (bool, optional): If True, cache the dataset in memory. Defaults to True.
        """
        super().__init__(
            roi=roi,
            path=path,
            crs=crs,
            transforms=transforms,
            download=download,
            overwrite=overwrite,
            cache=cache,
        )
        self.res = res
        self.bands = bands or ["treecover2000", "gain", "loss", "lossyear"]

    @property
    def collection(self) -> ee.Image:
        """Get the Earth Engine image with selected bands.

        Returns:
            ee.Image: Global Forest Change image with selected bands.
        """
        return ee.Image(self.gee_asset_id).select(self.bands)

    def _reducer(self, collection: ee.Image) -> ee.Image:
        """Reduce method for Global Forest Change dataset (identity function).

        Args:
            collection (ee.Image): Earth Engine image to reduce.

        Returns:
            ee.Image: The same input image (identity function).

        Note:
            For Global Forest Change datasets, the reduction is handled by
            the Earth Engine image itself, so this acts as an identity function.
        """
        return collection

    def _preprocess(self, image: ee.Image) -> ee.Image:
        """Preprocess Global Forest Change image.

        Note:
            This method is only to bypas abstract method requirement.
        """
        pass
