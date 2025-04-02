"""Sentinel-2 dataset module for Google Earth Engine integration.

This module provides the GEESentinel2 class for accessing Sentinel-2 Surface
Reflectance Harmonized data from Google Earth Engine with cloud masking and
preprocessing capabilities.
"""

from typing import Any, Callable, Dict, Optional

import ee
import pystac
import geopandas
from rasterio.crs import CRS
from torchgeo.datasets import BoundingBox

from .geebase import GEERasterDataset
from .utils import valid_date


class GEESentinel2(GEERasterDataset):
    """Sentinel-2 SR Harmonized image collection from Google Earth Engine.

    This dataset provides access to Sentinel-2 Surface Reflectance Harmonized data
    from Google Earth Engine with cloud masking and preprocessing.

    Attributes:
        filename_glob (str): File pattern for matching files.
        gee_asset_id (str): Earth Engine asset ID for Sentinel-2 collection.
        all_bands (List[str]): List of all available spectral bands.
        rgb_bands (List[str]): Bands to use for RGB visualization.
        nodata (int): NoData value for the dataset.
        instrument (str): Name of the sensor/instrument.

    Reference:
        https://developers.google.com/earth-engine/datasets/catalog/sentinel-2
    """

    filename_glob = "*.tif"

    gee_asset_id = "COPERNICUS/S2_SR_HARMONIZED"

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
        year: int | None = None,
        date_start: str | None = None,
        date_end: str | None = None,
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
        """Initialize a GEESentinel2 dataset instance.

        Args:
            year (int, optional): Year of the dataset. Either year or date_start/date_end
                must be provided.
            date_start (str, optional): Start date for data filtering in YYYY-MM-DD format.
            date_end (str, optional): End date for data filtering in YYYY-MM-DD format.
            roi (BoundingBox, optional): Region of interest to fetch data from.
            season (str, optional): Season of the dataset. The images returned represent
                the median pixel from the collection of images available for the season.
                Can be either "leafon" (April to September) or "leafoff" (October of
                prior year to March). Defaults to "leafon".
            res (float, optional): Resolution of the dataset in meters. Defaults to 10.
            path (str, optional): Directory where Sentinel-2 data are stored or will be
                stored if download is True. If path is provided and a matching file exists,
                the image will be loaded from that file unless overwrite is True.
            crs (CRS, optional): Coordinate Reference System for fetching images from
                Earth Engine. Defaults to EPSG:5070.
            transforms (Callable, optional): Function/transform that takes in a sample
                and returns a transformed version.
            download (bool, optional): If True, download the dataset to the path directory.
                Defaults to False.
            overwrite (bool, optional): If True, overwrite the dataset if it already exists.
                Defaults to False.
            cache (bool, optional): If True, cache the dataset in memory. Defaults to True.

        Raises:
            ValueError: If neither year nor date_start/date_end are provided, or if
                an invalid season is specified.
        """
        super().__init__(
            roi=roi,
            path=path,
            res=res,
            crs=crs,
            transforms=transforms,
            download=download,
            overwrite=overwrite,
            cache=cache,
        )
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

        if date_start is not None and date_end is not None:
            self.date_start = valid_date(date_start)
            self.date_end = valid_date(date_end)
        elif year is not None:
            if season == "leafoff":
                self.date_start = f"{year - 1}-10-01"
                self.date_end = f"{year}-03-31"
            elif season == "leafon":
                self.date_start = f"{year}-04-01"
                self.date_end = f"{year}-09-30"
            else:
                raise ValueError(f"Invalid season: {season}")
        else:
            raise ValueError("Either year or date_start and date_end must be provided.")

    @property
    def collection(self) -> ee.ImageCollection:
        """Get the Earth Engine image collection with filters applied.

        Returns:
            ee.ImageCollection: Filtered Sentinel-2 image collection with:
                - Date range filtered
                - Cloud cover < 20%
                - Preprocessing applied
                - Selected bands only
        """
        return (
            ee.ImageCollection(self.gee_asset_id)
            .filterDate(self.date_start, self.date_end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            .map(self._preprocess)
            .select(self.bands)
        )

    def _reducer(self, collection: ee.ImageCollection) -> ee.Image:
        """Reduce image collection to a single image using median.

        Args:
            collection (ee.ImageCollection): Earth Engine image collection to reduce.

        Returns:
            ee.Image: Median composite image.
        """
        return collection.median()

    def _preprocess(self, image: ee.Image) -> ee.Image:
        """Preprocess Sentinel-2 image by masking clouds and cirrus.

        Args:
            image (ee.Image): Raw Sentinel-2 Earth Engine image.

        Returns:
            ee.Image: Preprocessed image with cloud and cirrus masking applied.

        Note:
            Uses QA60 band bits 10 (clouds) and 11 (cirrus) to mask pixels
            with cloud cover. Both flags must be set to zero for clear conditions.
        """
        qa = image.select("QA60")
        # Bits 10 and 11 are clouds and cirrus
        cloudBitMask = 1 << 10
        cirrusBitMask = 1 << 11
        # Both flags should be set to zero, indicating clear conditions.
        mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))

        return image.updateMask(mask)  # .divide(10000)
