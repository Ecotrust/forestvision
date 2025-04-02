from datetime import datetime
from functools import partial
from dateutil.parser import parse
from typing import Any, Callable, Dict, Optional, Tuple

import ee
from rasterio.crs import CRS
from torchgeo.datasets import BoundingBox

from .geebase import GEERasterDataset
from .utils import valid_date


class GEELandsat8(GEERasterDataset):
    """Landsat 8 T1 SR Image Collection from Google Earth Engine.

    This dataset provides access to Landsat 8 Surface Reflectance Tier 1 data
    from Google Earth Engine with cloud masking and preprocessing.

    Attributes:
        filename_glob (str): File pattern for matching files.
        gee_asset_id (str): Earth Engine asset ID for Landsat 8 collection.
        all_bands (List[str]): List of all available spectral bands.
        rgb_bands (List[str]): Bands to use for RGB visualization.
        nodata (int): NoData value for the dataset.
        instrument (str): Name of the sensor/instrument.
        is_image (bool): Whether this dataset contains image data.

    Bands:
        - SR_B1 - Band 1 (ultra blue, coastal aerosol)
        - SR_B2 - Band 2 (blue)
        - SR_B3 - Band 3 (green)
        - SR_B4 - Band 4 (red)
        - SR_B5 - Band 5 (near infrared)
        - SR_B6 - Band 6 (shortwave infrared 1)
        - SR_B7 - Band 7 (shortwave infrared 2)
        - SR_QA_AEROSOL - Aerosol quality band
        - ST_* - Thermal bands
        - QA_PIXEL - Pixel quality attributes generated from the CFMASK algorithm.

    Spatial resolution: 30 meters

    Reference:
        https://developers.google.com/earth-engine/datasets/catalog/landsat-8
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

    nodata = 0  # -32768

    instrument = "Landsat 8 OLI/TIRS"

    is_image = True

    def __init__(
        self,
        year: int,
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
        """Initialize a GEELandsat8 dataset instance.

        Args:
            year (int): The year of the dataset.
            roi (BoundingBox, optional): Region of interest for data retrieval.
            res (float, optional): Resolution of the dataset in meters. Defaults to 30.
            season (str, optional): Season of interest, either "leafon" (April to September)
                or "leafoff" (October to March). Defaults to "leafon".
            path (str, optional): Directory for data storage. Required if download is True.
            crs (CRS, optional): Coordinate reference system used to load the image.
                Defaults to EPSG:5070.
            transforms (Callable, optional): Optional transform function.
            download (bool, optional): Whether to download data. Defaults to False.
            overwrite (bool, optional): Whether to overwrite existing data. Defaults to False.
            cache (bool, optional): Use cache to store data in memory. Defaults to True.

        Raises:
            ValueError: If an invalid season is provided.
        """
        super().__init__(
            # tiles=tiles,
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
        self.filename_suffix = f"_{year}_{season}"

        if season == "leafoff":
            self.date_start = f"{year - 1}-10-01"
            self.date_end = f"{year}-03-31"
        elif season == "leafon":
            self.date_start = f"{year}-04-01"
            self.date_end = f"{year}-09-30"
        else:
            raise ValueError(f"Invalid season: {season}")

    @property
    def collection(self) -> ee.ImageCollection:
        """Get the Earth Engine image collection with filters applied.

        Returns:
            ee.ImageCollection: Filtered Landsat 8 image collection with:
                - Cloud cover < 20%
                - Date range filtered
                - Preprocessing applied
                - Selected bands only
        """
        return (
            ee.ImageCollection(self.gee_asset_id)
            .filter(ee.Filter.lt("CLOUD_COVER", 20))
            .filterDate(self.date_start, self.date_end)
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
        """Preprocess Landsat 8 image

        Masks pixels likely to be cloud, shadow, water, or snow using `qa_pixel` band.
        The QA_PIXEL was generated using the CFMASK algorithm. Note that this algorithm
        does not perform well over bright surfaces like snow, ice, or buildings.
        See: https://www.usgs.gov/landsat-missions/cfmask-algorithm

        Args:
            image (ee.Image): Raw Landsat 8 Earth Engine image.

        Returns:
            ee.Image: Preprocessed image with cloud/snow masking and scaling applied.
        """
        # Mask unwanted pixels.
        qa = image.select("QA_PIXEL")
        snow = qa.bitwiseAnd(16).eq(0)
        cloud = qa.bitwiseAnd(32).eq(0)

        return image.updateMask(cloud).updateMask(snow).unmask(self.nodata)


class GEELandsatFTV(GEERasterDataset):
    """Fit-to-Vertex (FTV) Harmonized Landsat TM/ETM+/OLI dataset.

    This dataset provides access to LandTrendr-processed harmonized Landsat
    time series data with Fit-to-Vertex analysis applied.

    Code adapted from: https://github.com/eMapR/LT-ChangeDB/tree/master

    Attributes:
        filename_glob (str): File pattern for matching files.
        all_bands (List[str]): List of all available spectral bands.
        rgb_bands (List[str]): Bands to use for RGB visualization.
        _bands (List[str]): Internal bands storage.
        _collection: Internal collection storage.
        nodata (int): NoData value for the dataset.
        is_image (bool): Whether this dataset contains image data.
        date_end (int): End year for the time series (current year - 1).
        date_start (int): Start year for the time series (20 year lookback window).
        instrument (str): Name of the sensor/instrument.
    """

    filename_glob = "*.tif"

    all_bands = [
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B7",
    ]

    rgb_bands = ["B5", "B4", "B3"]

    _bands = [
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B7",
    ]

    _collection = None

    nodata = -32768

    is_image = True

    date_end = datetime.now().year - 1

    # 20 year lookback window
    date_start = date_end - 20

    instrument = "FTV Harmonized Landsat TM/ETM+/OLI"

    def __init__(
        self,
        year: int,
        roi: BoundingBox,
        season: str = "leafon",
        spectral_index: str = "NBR",
        # tiles: Optional[Union[geopandas.GeoDataFrame, pystac.Collection]] = None,
        path: Optional[str] = None,
        crs: Optional[CRS] = CRS.from_epsg(5070),
        res: float = 30,
        transforms: Callable[[Dict[str, Any]], Dict[str, Any]] | None = None,
        download: bool = False,
        overwrite: bool = False,
        cache: bool = True,
    ):
        """Initialize a GEELandsatFTV dataset instance.

        Args:
            year (int): Year of the dataset to fetch.
            roi (BoundingBox): Region of interest to fetch data from.
            season (str, optional): Season of the dataset. The images come from the
                median composite for that season, either "leafon" (April to September)
                or "leafoff" (October of the prior year to March). Defaults to "leafon".
            spectral_index (str, optional): Spectral index to use, either "NBR" or "NDVI".
                Defaults to "NBR".
            path (str, optional): Directory where data are stored or downloaded if download is True.
            crs (CRS, optional): Coordinate reference system. Defaults to EPSG:5070.
            res (float, optional): Resolution in meters. Defaults to 30.
            transforms (Callable, optional): Transform function on each sample.
            download (bool, optional): Whether to download data to path. Defaults to False.
            overwrite (bool, optional): Whether to overwrite existing files. Defaults to False.
            cache (bool, optional): Whether to cache data in memory. Defaults to True.
        """
        super().__init__(
            # tiles=tiles,
            roi=roi,
            path=path,
            crs=crs,
            transforms=transforms,
            download=download,
            overwrite=overwrite,
            cache=cache,
        )
        self.res = res
        self.season = season
        self.spectral_index = spectral_index
        self.year = year
        self.collection

    @property
    def collection(self) -> ee.Image:
        """Get the Fit-to-Vertex processed image collection.

        Returns:
            ee.Image: LandTrendr Fit-to-Vertex processed image for the specified year.

        Note:
            This property lazily initializes the collection using GEELandTrendr
            and caches the result for subsequent access.
        """
        if self._collection is None:
            lt = GEELandTrendr(
                roi=self.roi,
                date_start=self.date_start,
                date_end=self.date_end,
                season=self.season,
                spectral_index=self.spectral_index,
                ftv_bands=self.bands,
                crs=self.crs,
            )

            self.date_start = lt.date_start
            self.date_end = lt.date_end

            self._collection = lt.ftv_image(self.year)
            self._bands = self._collection.bandNames().getInfo()

        # TODO: return collection to comply with constructor pattern
        return self._collection

    def _reducer(self, image: ee.Image) -> ee.Image:
        """Reduce method for FTV dataset (identity function).

        Args:
            image (ee.Image): Input Earth Engine image.

        Returns:
            ee.Image: The same input image (identity function).

        Note:
            For FTV datasets, the reduction is handled by LandTrendr,
            so this acts as an identity function.
        """
        return image


class GEELandsatTimeSeries:
    """Harmonized Landsat 5-8 T1 SR time series imagery from Google Earth Engine.

    This class provides access to harmonized Landsat time series data from
    Landsat 5, 7, and 8 sensors with medoid compositing for each year.

    Attributes:
        gee_asset_id (str): Earth Engine asset ID template with SENSOR placeholder.
        bands (List[str]): List of harmonized band names.
        l8_bands (List[str]): Landsat 8 specific band names.
        sr_bands (List[str]): Surface reflectance band names.
        instrument (str): Name of the sensors/instruments.
        rgb_bands (List[str]): Bands to use for RGB visualization.
        nodata (int): NoData value for the dataset.

    Bands:
        Landsat 5 (1984-03-16 to 2012-05-05)
            - SR_B1 - Blue
            - SR_B2 - Green
            - SR_B3 - Red
            - SR_B4 - Near Infrared
            - SR_B5 - Shortwave Infrared 1
            - SR_B7 - Shortwave Infrared 2
            - ST_* - Thermal bands
            - QA_* - Quality assessment bands
        Landsat 7 (1999-05-28 to 2024-01-19)
            - SR_B1 - Blue
            - SR_B2 - Green
            - SR_B3 - Red
            - SR_B4 - Near Infrared
            - SR_B5 - Shortwave Infrared 1
            - SR_B7 - Shortwave Infrared 2
            - ST_* - Thermal bands
            - QA_* - Quality assessment bands
        Landsat 8 (2013-03-18 to present)
            - SR_B1 - Ultra blue, coastal aerosol
            - SR_B2 - Blue
            - SR_B3 - Green
            - SR_B4 - Red
            - SR_B5 - Near infrared
            - SR_B6 - Shortwave infrared 1
            - SR_B7 - Shortwave infrared 2
            - SR_QA_AEROSOL - Aerosol quality band
            - ST_* - Thermal bands
            - QA_PIXEL - Pixel quality attributes generated from the CFMASK algorithm.

    Spatial resolution: 30 meters

    Reference:
        https://developers.google.com/earth-engine/datasets/catalog/landsat
    """

    gee_asset_id = "LANDSAT/SENSOR/C02/T1_L2"

    bands = [
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B7",
    ]

    l8_bands = [
        "SR_B2",
        "SR_B3",
        "SR_B4",
        "SR_B5",
        "SR_B6",
        "SR_B7",
    ]

    sr_bands = [
        "SR_B1",
        "SR_B2",
        "SR_B3",
        "SR_B4",
        "SR_B5",
        "SR_B7",
    ]

    instrument = "Landsat 5 TM / 7 ETM+ / 8 OLI/TIRS"

    rgb_bands = ["B5", "B4", "B3"]

    nodata: int = -32768

    def __init__(
        self,
        roi: BoundingBox,
        date_start: int | str,
        date_end: int | str,
        season: str = "leafon",
        crs: Optional[CRS] = None,  # CRS.from_epsg(5070),
    ) -> None:
        """Initialize a GEELandsatTimeSeries dataset instance.

        Args:
            roi (BoundingBox): Region of interest to fetch data from.
            date_start (int | str): Start date of the time series. If int, the year is assumed.
            date_end (int | str): End date of the time series. If int, the year is assumed.
            season (str, optional): Season of the dataset. The images returned represent
                the median pixel from the collection of images available for the season.
                Can be either "leafon" (April to September) or "leafoff" (October of
                prior year to March). Defaults to "leafon".
            crs (CRS, optional): Coordinate Reference System for fetching images.
                Defaults to None.

        Raises:
            ValueError: If an invalid season is provided.
        """

        season = season.lower()
        if season == "leafon":
            start_day = f"04-01"
            end_day = f"09-30"
        elif season == "leafoff":
            start_day = "10-01"
            end_day = "03-31"
        elif season is None:
            start_day = "01-01"
            end_day = "12-31"
        else:
            raise ValueError(
                "Invalid season. Options are 'leafon', 'leafoff', or None."
            )

        if isinstance(date_start, int) & (len(str(date_start)) == 4):
            date_start = valid_date(f"{date_start}-{start_day}")
        else:
            date_start = valid_date(date_start)
            start_day = date_start[5:]

        if isinstance(date_end, int) & (len(str(date_end)) == 4):
            date_end = valid_date(f"{date_end}-{end_day}")
        else:
            date_end = valid_date(date_end)
            end_day = date_end[5:]

        self.roi = roi
        self.crs = crs
        self.date_start = date_start
        self.date_end = date_end

    def get_tscollection(self) -> ee.ImageCollection:
        """Get the harmonized Landsat time series collection.

        Returns:
            ee.ImageCollection: Time series collection containing medoid-composited
            images for each year in the specified date range.

        Note:
            Uses medoid compositing to generate a single image for each year,
            which minimizes the sum of pairwise distances between all images.
        """
        minx, maxx, miny, maxy, _, _ = self.roi
        roi = ee.Geometry.Rectangle(
            (minx, miny, maxx, maxy),
            proj=f"EPSG:{self.crs.to_epsg()}",
            evenOdd=True,
            geodesic=False,
        )
        start = parse(self.date_start)
        end = parse(self.date_end)
        images = []
        for year in range(start.year, end.year + 1):
            date_start = start.replace(year=year).strftime("%Y-%m-%d")
            date_end = end.replace(year=year).strftime("%Y-%m-%d")
            images.append(
                self._medoid_collection(date_start, date_end, roi).set(
                    "system:time_start", ee.Date(date_end).millis()
                )
            )
        return ee.ImageCollection(images).select(self.sr_bands, self.bands)

    def _get_collection(
        self, sensor: str, date_start: str, date_end: str, roi: ee.Geometry
    ) -> ee.ImageCollection:
        """Get image collection for a specific sensor and date range.

        Args:
            sensor (str): Landsat sensor name ('LT05', 'LE07', or 'LC08').
            date_start (str): Start date in YYYY-MM-DD format.
            date_end (str): End date in YYYY-MM-DD format.
            roi (ee.Geometry): Region of interest geometry.

        Returns:
            ee.ImageCollection: Filtered and preprocessed image collection for the sensor.
        """
        preprocess = partial(self._preprocess, sensor=sensor)
        asset_id = self.gee_asset_id.replace("SENSOR", sensor)
        return (
            ee.ImageCollection(asset_id)
            .filterDate(date_start, date_end)
            .filterBounds(roi)
            .map(preprocess)
            .select(self.sr_bands)
        )

    def _medoid_collection(
        self, date_start: str, date_end: str, roi: ee.Geometry
    ) -> ee.ImageCollection:
        """Create medoid composite collection for a specific date range.

        Code adapted from: https://github.com/eMapR/LT-ChangeDB/tree/master


        Args:
            date_start (str): Start date in YYYY-MM-DD format.
            date_end (str): End date in YYYY-MM-DD format.
            roi (ee.Geometry): Region of interest geometry.

        Returns:
            ee.ImageCollection: Medoid-composited image collection.

        Note:
            Medoid compositing selects the image that minimizes the sum of
            pairwise distances between all images in the collection.
        """
        zero_collection = ee.ImageCollection(
            [ee.Image([0, 0, 0, 0, 0, 0]).mask(ee.Image(0))]
        )
        l5 = self._get_collection("LT05", date_start, date_end, roi)
        l7 = self._get_collection("LE07", date_start, date_end, roi)
        l8 = self._get_collection("LC08", date_start, date_end, roi)

        merged = ee.ImageCollection(l5.merge(l7).merge(l8))
        img_count = merged.toList(1).length()
        merged = ee.ImageCollection(
            ee.Algorithms.If(img_count.gt(0), merged, zero_collection)
        )

        def median_diff(img):
            diff = ee.Image(img).subtract(median).pow(ee.Image.constant(2))
            return diff.reduce("sum").addBands(img)

        median = merged.median()
        difference = merged.map(median_diff)
        return (
            ee.ImageCollection(difference)
            .reduce(ee.Reducer.min(7))
            .select([1, 2, 3, 4, 5, 6], self.sr_bands)
        )

    def _preprocess(self, image: ee.Image, sensor: str) -> ee.Image:
        """Preprocess and harmonize Landsat image bands.

        Args:
            image (ee.Image): Raw Earth Engine image to preprocess.
            sensor (str): Landsat sensor name ('LT05', 'LE07', or 'LC08').

        Returns:
            ee.Image: Preprocessed image with harmonized bands and masking applied.

        Note:
            - Masks unwanted pixels (shadows, snow, clouds)
            - Harmonizes OLI to ETM+ reflectance for Landsat 8
            - Converts to 16-bit integer format
        """
        # Mask unwanted pixels.
        qa = image.select("QA_PIXEL")
        mask = (
            qa.bitwiseAnd(8)
            .eq(0)  # shadows
            .And(qa.bitwiseAnd(16).eq(0))  # snow
            .And(qa.bitwiseAnd(32).eq(0))  # cloud
        )

        # Harmonize OLI to ETM+ reflectance.
        if sensor == "LC08":
            image = self._harmonize_to_etm(image)
        else:
            image = (
                image.unmask()
                .resample("bicubic")
                .set("system:time_start", image.get("system:time_start"))
            )

        return image.mask(mask).toShort()

    def _harmonize_to_etm(self, image: ee.Image) -> ee.Image:
        """Harmonize Landsat 8 OLI bands to ETM+ reflectance.

        Args:
            image (ee.Image): Landsat 8 OLI image to harmonize.

        Returns:
            ee.Image: Harmonized image with reflectance values adjusted to match ETM+.

        Note:
            Uses slope and intercept coefficients from:
            Roy, D.P., Kovalskyy, V., Zhang, H.K., Vermote, E.F., Yan, L., Kumar, S.S,
            Egorov, A., 2016, Characterization of Landsat-7 to Landsat-8 reflective
            wavelength and normalized difference vegetation index continuity,
            Remote Sensing of Environment, 185, 57-70.
            https://doi.org/10.1016/j.rse.2015.12.024
            Table 2 - reduced major axis (RMA) regression coefficients
        """
        slopes = ee.Image.constant(
            [
                0.9785,
                0.9542,
                0.9825,
                1.0073,
                1.0171,
                0.9949,
            ]
        )
        itcp = ee.Image.constant(
            [
                -0.0095,
                -0.0016,
                -0.0022,
                -0.0021,
                -0.0030,
                0.0029,
            ]
        )
        return (
            image.select(self.l8_bands, self.sr_bands)
            .unmask()
            .resample("bicubic")
            .subtract(itcp.multiply(10000))
            .divide(slopes)
            .set("system:time_start", image.get("system:time_start"))
        )

    def __call__(self) -> ee.ImageCollection:
        """Callable interface to get the time series collection.

        Returns:
            ee.ImageCollection: Time series collection (same as get_tscollection()).
        """
        return self.get_tscollection()


class GEELandTrendr:
    """Performs LandTrendr analysis on a Harmonized Landsat time series.

    This class provides LandTrendr (LandTrendr Temporal Segmentation) analysis
    for harmonized Landsat time series data, producing Fit-to-Vertex (FTV) results.

    Code adapted from: https://github.com/eMapR/LT-ChangeDB/tree/master

    Attributes:
        lt_params (dict): LandTrendr algorithm parameters.
        _lt_result: Internal storage for LandTrendr results.

    Returns:
        ee.ImageCollection: LandTrendr results as Fit-to-Vertex (FTV) images.
    """

    lt_params = {
        "maxSegments": 6,
        "spikeThreshold": 0.9,
        "vertexCountOvershoot": 3,
        "preventOneYearRecovery": True,
        "recoveryThreshold": 0.25,
        "pvalThreshold": 0.05,
        "bestModelProportion": 0.75,
        "minObservationsNeeded": 6,
    }

    _lt_result = None

    def __init__(
        self,
        roi: BoundingBox,
        date_start: int | str,
        date_end: int | str,
        season: str = "leafon",
        spectral_index: str = "NBR",
        ftv_bands: Optional[Tuple] = ["B1", "B2", "B3", "B4", "B5", "B7"],
        crs: Optional[CRS] = None,  # CRS.from_epsg(5070),
    ):
        """Initialize a GEELandTrendr instance for time series analysis using LandTrendr.

        Args:
            roi (BoundingBox): Region of interest to analyze.
            date_start (int | str): Start date of the time series (YYYY or YYYY-MM-DD).
            date_end (int | str): End date of the time series (YYYY or YYYY-MM-DD).
            season (str, optional): Target season, either "leafon" (April to September)
                or "leafoff" (October to March). Defaults to "leafon".
            spectral_index (str, optional): Spectral index to use, "NBR" or "NDVI".
                Defaults to "NBR".
            ftv_bands (Tuple, optional): Bands to include in the FTV output.
                Defaults to ["B1", "B2", "B3", "B4", "B5", "B7"].
            crs (CRS, optional): Coordinate Reference System used to fetch images
                from Earth Engine. Defaults to None.
        """
        self.roi = roi
        self.crs = crs
        self.date_start = date_start
        self.date_end = date_end
        self.season = season
        self.spectral_index = spectral_index
        self.ftv_bands = ftv_bands
        self.lt_result

    def normalized_difference(
        self,
        image: ee.Image,
        sindex: str = "NBR",
    ) -> ee.Image:
        """Calculate normalized difference spectral index.

        Args:
            image (ee.Image): Earth Engine image containing spectral bands.
            sindex (str, optional): Spectral index to calculate, either "NBR" or "NDVI".
                Defaults to "NBR".

        Returns:
            ee.Image: Image with normalized difference index calculated.

        Raises:
            ValueError: If an invalid spectral index is provided.
        """
        if sindex == "NBR":
            bands = ["B4", "B7"]
        elif sindex == "NDVI":
            bands = ["B4", "B3"]
        else:
            raise ValueError("Invalid spectral index. Options are 'NBR' or 'NDVI'")

        nd = (
            image.normalizedDifference(bands)
            .multiply(1000)
            .select([0], [sindex])
            .toShort()
        )

        nd = (
            nd.multiply(-1)
            .addBands(nd.select([0], [f"{sindex}_FTV"]))
            .set("system:time_start", image.get("system:time_start"))
        )

        if self.ftv_bands:
            return nd.addBands(image.select(self.ftv_bands))
        else:
            return nd

    @property
    def lt_result(self) -> ee.Image:
        """Get the LandTrendr analysis results.

        Returns:
            ee.Image: LandTrendr processed results.

        Note:
            This property lazily initializes the LandTrendr analysis using
            a harmonized Landsat time series collection.
        """
        if self._lt_result is None:

            ts_collection = GEELandsatTimeSeries(
                roi=self.roi,
                date_start=self.date_start,
                date_end=self.date_end,
                season=self.season,
                crs=self.crs,
            )

            self.date_start = ts_collection.date_start
            self.date_end = ts_collection.date_end

            norm_diff = partial(
                self.normalized_difference,
                sindex=self.spectral_index,
            )

            ts_collection = ts_collection().map(norm_diff)

            self._lt_result = ee.Algorithms.TemporalSegmentation.LandTrendr(
                ts_collection, **self.lt_params
            )

        return self._lt_result

    def ftv_image(self, year: int) -> ee.Image:
        """Get Fit-to-Vertex image for a specific year.

        Args:
            year (int): Year to extract FTV results for.

        Returns:
            ee.Image: Fit-to-Vertex processed image for the specified year.

        Note:
            Extracts the fitted values from LandTrendr results for the given year
            and renames bands to standardized names.
        """
        lt_res = self.lt_result
        start = parse(self.date_start)
        end = parse(self.date_end)
        years = [f"{year}" for year in range(start.year, end.year + 1)]

        fitted_bands = []
        band_names = []
        for idx, band in enumerate(lt_res.bandNames().getInfo()):
            if idx > 0:
                band_names.append(band)
                if band == "rmse":
                    fitted_bands.append(lt_res.select(band))
                else:
                    fit = lt_res.select([idx]).arrayFlatten([years])
                    fitted_bands.append(fit.select([str(year)], [band]))

        renamed_bands = [b.upper().split("_")[0] for b in band_names]
        return ee.Image(fitted_bands).select(band_names, renamed_bands).toShort()
