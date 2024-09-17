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


class GEESentinel2(GEERasterDataset):
    """Sentinel-2 Harmonized image collection from Google Earth Engine.

    https://developers.google.com/earth-engine/datasets/catalog/sentinel-2
    """

    filename_glob = "*.tif"

    gee_asset_id = {
        "sr": "COPERNICUS/S2_SR_HARMONIZED",
        "toa": "COPERNICUS/S2_HARMONIZED",
    }

    all_bands = [
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B9",
        "B10",
        "B11",
        "B12",
    ]

    rgb_bands = ["B4", "B3", "B2"]

    nodata = 0

    instrument = "Sentinel 2 MSI"

    def __init__(
        self,
        year: int,
        plevel: str = "sr",
        tiles: Optional[Union[geopandas.GeoDataFrame, pystac.Collection]] = None,
        roi: Optional[BoundingBox] = None,
        season: str = "leafon",
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

            year : int
                Year of the dataset.
            plevel : str
                Sentinel 2 processing level. Can be either `sr` (Surface Reflectance) or
                `toa` (Top of Atmosphere). Default is `sr`.
            tiles : geopandas.GeoDataFrame
                GeoDataFrame containing the tile collection.
            roi : BoundingBox
                Region of interest to fetch data from.
            season : str
                Season of the dataset. The images returned represent the median pixel from the collection
                of images available for the season. Can be either "leafon" (October to September)
                or "leafoff" (October of prior year to March).
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
            tiles=tiles,
            roi=roi,
            path=path,
            crs=crs,
            transforms=transforms,
            download=download,
            overwrite=overwrite,
            cache=cache,
        )
        self.plevel = plevel
        self.res = res
        self.paths = path
        self.bands = [
            "B1",
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

        if season == "leafoff":
            self.date_start = f"{year - 1}-10-01"
            self.date_end = f"{year}-03-31"
        elif season == "leafon":
            self.date_start = f"{year}-04-01"
            self.date_end = f"{year}-09-30"
        else:
            raise ValueError(f"Invalid season: {season}")

    @property
    def _collection(self):
        return (
            ee.ImageCollection(self.gee_asset_id[self.plevel])
            .filterDate(self.date_start, self.date_end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            .map(self._preprocess)
            .select(self.bands)
        )

    def _reducer(self, collection: ee.ImageCollection) -> ee.Image:
        return collection.median()

    def _preprocess(self, image: ee.Image) -> ee.Image:
        """
        Mask pixels likely to be cloud, shadow, water, or snow using QA_PIXELS band.
        """
        qa = image.select("QA60")
        # Bits 10 and 11 are clouds and cirrus
        cloudBitMask = 1 << 10
        cirrusBitMask = 1 << 11
        # Both flags should be set to zero, indicating clear conditions.
        mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))

        return image.updateMask(mask)  # .divide(10000)
