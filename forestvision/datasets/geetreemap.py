"""USFS TreeMap 2022 dataset module for Google Earth Engine integration.

This module provides the GEETreeMap class for accessing USFS TreeMap 2022 data
from Google Earth Engine, which provides detailed forest characteristics across
the conterminous United States.
"""

from typing import Any, Callable, Dict, Optional

import ee
from rasterio.crs import CRS
from torchgeo.datasets import BoundingBox

from .geebase import GEERasterDataset


class GEETreeMap(GEERasterDataset):
    """USFS TreeMap 2022 Earth Engine Collection.

    TreeMap 2022 is a tree-level model of the forests of the conterminous United States
    circa 2022. It provides detailed spatial information on forest characteristics, including
    the number of live and dead trees, biomass, and carbon. The dataset is derived using a
    random forest machine learning algorithm that assigns the most similar Forest Inventory
    and Analysis (FIA) plot to each pixel of gridded LANDFIRE input data.

    Dataset Bands:
        ALSTK: All-Live-Tree Stocking (%)
        BALIVE: Live Tree Basal Area (ft²/acre)
        CANOPYPCT: Live Canopy Cover (%)
        CARBON_D: Carbon, Standing Dead (tons/acre)
        CARBON_DWN: Carbon, Down Dead (tons/acre)
        CARBON_L: Carbon, Live Above Ground (tons/acre)
        DRYBIO_D: Dry Standing Dead Tree Biomass, Above Ground (tons/acre)
        DRYBIO_L: Dry Live Tree Biomass, Above Ground (tons/acre)
        FLDSZCD: Field Stand-Size Class Code
        FLDTYPCD: Field Forest Type Code
        FORTYPCD: Algorithm Forest Type Code
        GSSTK: Growing-Stock Stocking (%)
        QMD: Stand Quadratic Mean Diameter (in)
        SDIsum: Sum of Stand Density Index
        STANDHT: Height of Dominant Trees (ft)
        STDSZCD: Algorithm Stand-Size Class Code
        TM_ID: TreeMap ID
        TPA_DEAD: Dead Trees Per Acre
        TPA_LIVE: Live Trees Per Acre
        VOLBFNET_L: Volume, Live (sawlog-board-ft/acre)
        VOLCFNET_D: Volume, Standing Dead (ft³/acre)
        VOLCFNET_L: Volume, Live (ft³/acre)

    Spatial Resolution: 30 meters

    Temporal Coverage: Circa 2022

    References:
        Riley, Karin L.; Grenfell, Isaac C.; Shaw, John D.; Finney, Mark A. 2022.
        TreeMap 2016 dataset generates CONUS-wide maps of forest characteristics
        including live basal area, aboveground carbon, and number of trees per acre.
        Journal of Forestry. 120(6): 607–632.
        https://doi.org/10.1093/jofore/fvac029
    """

    filename_glob = "*.tif"

    gee_asset_id = "USFS/GTAC/TreeMap/v2022"

    all_bands = [
        "ALSTK",
        "BALIVE",
        "CANOPYPCT",
        "CARBON_D",
        "CARBON_DWN",
        "CARBON_L",
        "DRYBIO_D",
        "DRYBIO_L",
        "FLDSZCD",
        "FLDTYPCD",
        "FORTYPCD",
        "GSSTK",
        "QMD",
        "SDIsum",
        "STANDHT",
        "STDSZCD",
        "TM_ID",
        "TPA_DEAD",
        "TPA_LIVE",
        "VOLBFNET_L",
        "VOLCFNET_D",
        "VOLCFNET_L",
    ]
        
    rgb_bands = ["BALIVE"]

    nodata = -32768

    is_image = False

    instrument = "USFS TreeMap"

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
        """Initialize a GEETreeMap dataset instance.

        Args:
            roi (BoundingBox, optional): Region of interest to fetch data from.
            res (float, optional): Resolution of the dataset in meters. Defaults to 30.
            bands (list, optional): List of bands to select from the dataset. If not provided,
                defaults to ["ALST", "BIOMASS", "CARBON"].
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
        self.bands = bands or self.all_bands

    @property
    def collection(self) -> ee.Image:
        """Get the Earth Engine image with selected bands.

        Returns:
            ee.Image: TreeMap 2022 image with selected bands.
        """
        return (
            ee.ImageCollection(self.gee_asset_id)
                .select(self.bands)
                .filter(ee.Filter.eq("year", "2022"))
                .filter(ee.Filter.eq("study_area", "CONUS"))
        )

    def _reducer(self, collection: ee.Image) -> ee.Image:
        """Reduce method for TreeMap dataset (identity function).

        Args:
            collection (ee.Image): Earth Engine image to reduce.

        Returns:
            ee.Image: The same input image (identity function).

        Note:
            For TreeMap 2022, which is a static image dataset (circa 2022),
            the reduction is handled by the Earth Engine image itself,
            so this acts as an identity function.
        """
        image = collection.first().toInt16()
        mask = image.gte(0).And(image.lte(36522)) 
        return image.updateMask(mask)

    def _preprocess(self, image: ee.Image) -> ee.Image:
        """Preprocess TreeMap image.

        Note:
            This method is only to bypass abstract method requirement.
            No preprocessing is needed for TreeMap data.
        """
        pass
