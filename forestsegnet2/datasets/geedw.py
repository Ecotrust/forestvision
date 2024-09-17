"""

"""

# %%
from typing import Any, Callable, Dict, Optional, Union

import ee
import pystac
import geopandas
from rasterio.crs import CRS
from torchgeo.datasets import BoundingBox

from .geebase import GEERasterDataset


class GEEDynamicWorld(GEERasterDataset):
    """Dynamic World imagery from Google Earth Engine.

    https://dynamicworld.app/
    https://developers.google.com/earth-engine/tutorials/community/introduction-to-dynamic-world-pt-1

    GEE Class indexes:
    0: Water
    1: Trees
    2: Grass
    3: Flooded vegetation
    4: Crops
    5: Shrub & Scrub
    6: Built Area
    7: Bare ground
    8: Snow & Ice
    9: label
    """

    filename_glob = "*.tif"

    gee_asset_id = "GOOGLE/DYNAMICWORLD/V1"

    all_bands = [
        "water",
        "trees",
        "grass",
        "flooded_vegetation",
        "crops",
        "shrub_and_scrub",
        "built",
        "bare",
        "snow_and_ice",
        "label",
    ]

    _cmap = {
        0: "#419BDF",
        1: "#397D49",
        2: "#88B053",
        3: "#7A87C6",
        4: "#E49635",
        5: "#DFC35A",
        6: "#C4281B",
        7: "#A59B8F",
        8: "#B39FE1",
    }

    nodata = 0

    instrument = "Dynamic World"

    def __init__(
        self,
        date_start: str,
        date_end: str,
        tiles: Optional[Union[geopandas.GeoDataFrame, pystac.Collection]] = None,
        roi: Optional[BoundingBox] = None,
        res: float = 10,
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
            tiles : geopandas.GeoDataFrame
                GeoDataFrame containing the tile collection.
            roi : BoundingBox
                Region of interest to fetch data from.
            res : float
                Resolution of the dataset. Default is 10.
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
            path=path,
            tiles=tiles,
            roi=roi,
            transforms=transforms,
            crs=crs,
            download=download,
            overwrite=overwrite,
            cache=cache,
        )
        self.res = res
        self.bands = ["label"]
        self.date_start = date_start
        self.date_end = date_end
        self.is_image = False

    @property
    def _collection(self):
        return (
            ee.ImageCollection(self.gee_asset_id)
            .filterDate(self.date_start, self.date_end)
            .select(self.bands)
            # .reduce(ee.Reducer.mode())
        )

    def _reducer(self, collection: ee.ImageCollection) -> ee.Image:
        return collection.reduce(ee.Reducer.mode())

    def _preprocess(self):
        """Bypass abstract method."""
        pass
