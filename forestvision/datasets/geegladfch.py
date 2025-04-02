"""
"""

from typing import Any, Callable, Dict, Optional, Union

import ee
import pystac
import geopandas
from rasterio.crs import CRS
from torchgeo.datasets import BoundingBox

from .geebase import GEERasterDataset


class GLADForestCanopyHeight(GEERasterDataset):
    """GLAD Forest Canopy Height from Google Earth Engine."""

    filename_glob = "*.tif"

    gee_asset_id = "users/potapovpeter/GEDI_V27"

    all_bands = []

    rgb_bands = []

    nodata = 0

    is_image = False

    instrument = "GLAD Forest Canopy Height"

    def __init__(
        self,
        tiles: Optional[Union[geopandas.GeoDataFrame, pystac.Collection]] = None,
        roi: Optional[BoundingBox] = None,
        res: float = 30,
        path: Optional[str] = None,
        crs: Optional[CRS] = CRS.from_epsg(5070),
        transforms: Callable[[Dict[str, Any]], Dict[str, Any]] | None = None,
        download: bool = False,
        overwrite: bool = False,
        cache: bool = True,
    ) -> None:
        """
        Args:

            year : int
                Year of the dataset.
            tiles : geopandas.GeoDataFrame
                GeoDataFrame containing the tile collection.
            roi : BoundingBox
                Region of interest to fetch data from.
            res: float
                Resolution of the dataset. Default is 30.
            path : str
                Directory where Sentinel-2 data are stored or will be stored if download option
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
            tiles=tiles,
            roi=roi,
            path=path,
            crs=crs,
            transforms=transforms,
            download=download,
            overwrite=overwrite,
            cache=cache,
        )
        self.res = res
        self.bands = []

    @property
    def collection(self):
        return ee.ImageCollection(self.gee_asset_id)

    def _reducer(self, collection: ee.ImageCollection) -> ee.Image:
        return collection.mosaic()

    def _preprocess(self, image: ee.Image):
        """Scales and masks unwanted pixels."""
        pass
