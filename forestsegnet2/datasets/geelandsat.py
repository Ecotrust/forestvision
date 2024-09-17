"""
"""

from typing import Any, Callable, Dict, Optional, Union

import ee
import pystac
import geopandas
from rasterio.crs import CRS
from torchgeo.datasets import BoundingBox

from .geebase import GEERasterDataset


class GEELandsat8(GEERasterDataset):
    """Landsat 8 imagery from Google Earth Engine.

    https://developers.google.com/earth-engine/datasets/catalog/landsat-8

    Bands:
    SR_B1 - Band 1 (ultra blue, coastal aerosol) surface reflectance
    SR_B2 - Band 2 (blue) surface reflectance
    SR_B3 - Band 3 (green) surface reflectance
    SR_B4 - Band 4 (red) surface reflectance
    SR_B5 - Band 5 (near infrared) surface reflectance
    SR_B6 - Band 6 (shortwave infrared 1) surface reflectance
    SR_B7 - Band 7 (shortwave infrared 2) surface reflectance
    SR_QA_AEROSOL - Aerosol quality band
    ST_* - Surface temperature bands
    QA_PIXEL - Pixel quality attributes generated from the CFMASK algorithm.
    """

    filename_glob = "*.tif"

    gee_asset_id = "LANDSAT/LC08/C02/T1_L2"

    all_bands = [
        "SR_B1",
        "SR_B2",
        "SR_B3",
        "SR_B4",
        "SR_B5",
        "SR_B6",
        "SR_B7",
    ]

    rgb_bands = ["SR_B6", "SR_B5", "SR_B4"]

    nodata = 0

    instrument = "Landsat 8 OLI/TIRS"

    is_image = True

    def __init__(
        self,
        year: int,
        tiles: Optional[Union[geopandas.GeoDataFrame, pystac.Collection]] = None,
        roi: Optional[BoundingBox] = None,
        res: float = 30,
        season: str = "leafon",
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
            res : float
                Resolution of the dataset. Default is 30.
            season : str
                Season of the dataset. The images returned represent the median pixel from the collection
                of images available for the season. Can be either "leafon" (October to September)
                or "leafoff" (October of prior year to March).
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
        self.bands = [
            "SR_B1",
            "SR_B2",
            "SR_B3",
            "SR_B4",
            "SR_B5",
            "SR_B6",
            "SR_B7",
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
            ee.ImageCollection(self.gee_asset_id)
            .filter(ee.Filter.lt("CLOUD_COVER", 20))
            .filterDate(self.date_start, self.date_end)
            .map(self._preprocess)
            .select(self.bands)
        )

    def _reducer(self, collection: ee.ImageCollection) -> ee.Image:
        return collection.median()

    def _preprocess(self, image: ee.Image):
        """Scales and masks unwanted pixels.

        For scaling, applies scaling factors to revert 16-bit integer values to surface reflectance units.
        See https://www.usgs.gov/faqs/how-do-i-use-a-scale-factor-landsat-level-2-science-products

        Masks pixels likely to be cloud, shadow, water, or snow using `qa_pixel` band.
        The QA_PIXEl was generated using the CFMASK algorithm. This algorithm does not
        perform well over bright surfaces like snow, ice, or buildings.
        See https://www.usgs.gov/landsat-missions/cfmask-algorithm
        """
        optical_bands = image.select("SR_B.").multiply(0.0000275).add(-0.2)
        thermal_bands = image.select("ST_B.*").multiply(0.00341802).add(149.0)

        qa = image.select("QA_PIXEL")

        shadow = qa.bitwiseAnd(8).eq(0)
        snow = qa.bitwiseAnd(16).eq(0)
        cloud = qa.bitwiseAnd(32).eq(0)
        water = qa.bitwiseAnd(4).eq(0)

        return (
            # image.addBands(optical_bands, None, True)
            # .addBands(thermal_bands, None, True)
            # .updateMask(shadow)
            image.updateMask(cloud).updateMask(snow)
            # .updateMask(water)
        )
