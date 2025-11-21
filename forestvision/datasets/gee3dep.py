"""USGS 3DEP 10m National Map Seamless (1/3 Arc-Second) Earth Engine Collection.

The USGS 3DEP 10m collection is a seamless Digital Elevation Model (DEM) dataset
covering the contiguous U.S., Hawaii, and U.S. territories, with ongoing coverage
expansion in Alaska. This dataset provides elevation data with a ground spacing
of approximately 10 meters north/south.

For more information, see the `USGS 3DEP documentation <https://www.usgs.gov/core-science-systems/ngp/3dep/about-3dep-products-services>`_.

Citation:
    U.S. Geological Survey, 3D Elevation Program 10-Meter Resolution Digital Elevation Model.

Dataset Bands:
    elevation: Elevation in meters

Spatial Resolution: 10.2 meters
"""

# %%
from typing import Any, Callable, Dict, Optional, Union

import ee
from rasterio.crs import CRS
from torchgeo.datasets import BoundingBox

from .geebase import GEERasterDataset


class GEE3Dep(GEERasterDataset):
    """`USGS 3DEP 10m National Map Seamless (1/3 Arc-Second) Earth Engine Collection <https://developers.google.com/earth-engine/datasets/catalog/USGS_3DEP_10m_collection>`_

    The USGS 3DEP 10m collection is a seamless Digital Elevation Model (DEM) dataset
    covering the contiguous U.S., Hawaii, and U.S. territories, with ongoing coverage
    expansion in Alaska. This dataset provides elevation data with a ground spacing
    of approximately 10 meters north/south.

    For more information, see the `USGS 3DEP documentation <https://www.usgs.gov/core-science-systems/ngp/3dep/about-3dep-products-services>`_.

    Citation:
        U.S. Geological Survey, 3D Elevation Program 10-Meter Resolution Digital Elevation Model.

    Dataset Bands:
        elevation: Elevation in meters

    Spatial Resolution: 10.2 meters
    """

    filename_glob = "*.tif"

    gee_asset_id = "USGS/3DEP/10m_collection"

    all_bands = [
        "elevation",
    ]

    nodata = None

    is_image = True

    instrument = "USGS 3DEP"

    def __init__(
        self,
        date_start: str,
        date_end: str,
        roi: Optional[BoundingBox] = None,
        res: float = 10.2,
        bands: Optional[list] = None,
        path: Optional[str] = None,
        crs: Optional[CRS] = CRS.from_epsg(5070),
        transforms: Callable[[Dict[str, Any]], Dict[str, Any]] | None = None,
        download: bool = False,
        overwrite: bool = False,
        cache: bool = True,
    ) -> None:
        """
        Args:
            date_start : str
                Start date for the image collection.
            date_end : str
                End date for the image collection.
            roi : BoundingBox
                Region of interest to fetch data from.
            res : float
                Resolution of the dataset. Default is 10.2 meters.
            bands : list
                List of bands to be used. Default is ["elevation"].
            path : str
                Directory where 3DEP data are stored or will be stored if download option
                is set to True. If path is provided and a matching file exists, the image will be
                loaded from that file unless overwrite = True. Default is None.
            crs : Optional[CRS]
                Images will be fetched from Earth Engine using this Coordinate Reference System.
                Default is None.
            transform : Optional[Callable]
                A function/transform that takes in a sample and returns a transformed version
            download : bool
                If True, download the dataset to the path directory. Default is False.
            overwrite : bool
                If True, overwrite the dataset if it already exists. Default is False.
            cache : bool
                If True, cache the dataset in memory. Default is True.
        """
        super().__init__(
            roi=roi,
            path=path,
            res=res,
            transforms=transforms,
            crs=crs,
            download=download,
            overwrite=overwrite,
            cache=cache,
        )
        self.date_start = date_start
        self.date_end = date_end
        self.bands = bands or ["elevation"]

    @property
    def collection(self):
        return (
            ee.ImageCollection(self.gee_asset_id)
            # .filterDate(self.date_start, self.date_end)
            .select(self.bands)
        )

    def _reducer(self, collection: ee.ImageCollection) -> ee.Image:
        """Reduce collection to a single image."""
        return collection.reduce(ee.Reducer.mode())

    def _preprocess(self):
        """Bypass abstract method."""
        pass
