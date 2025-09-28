""" """

# %%
from typing import Any, Callable, Dict, Optional, Union

import ee
from rasterio.crs import CRS
from torchgeo.datasets import BoundingBox

from .geebase import GEERasterDataset


class GEEDynamicWorld(GEERasterDataset):
    """`Dynamic World Earth Engine Collection <https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1>`_

    The Dynamic World dataset is a high resolution global land cover map. It is generated using a
    combination of satellite imagery and machine learning techniques. The dataset is updated
    frequently, allowing for near real-time monitoring of land cover changes.

    For more information, see the `WRI Dynamic World documentation <https://dynamicworld.app/>`_.

    Citation:
        Brown, C.F., Brumby, S.P., Guzder-Williams, B. et al. Dynamic World, Near real-time
        global 10 m land use land cover mapping. Sci Data 9, 251 (2022).
        doi:10.1038/s41597-022-01307-4

    Dataset Bands:
        0: Water
        1: Trees
        2: Grass
        3: Flooded vegetation
        4: Crops
        5: Shrub & Scrub
        6: Built Area
        7: Bare ground
        8: Snow & Ice
        9: label - discrete label [0,8] for each pixel based on the class with the highest probability

    Spatial Resolution: 10 meters
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

    is_image = False

    instrument = "Dynamic World"

    def __init__(
        self,
        date_start: str,
        date_end: str,
        roi: Optional[BoundingBox] = None,
        res: float = 10,
        class_name: Optional[str] | None = None,
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
                Resolution of the dataset. Default is 10.
            bands : list
                List of bands to be used. Default is ["label"].
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
            roi=roi,
            path=path,
            res=res,
            transforms=transforms,
            crs=crs,
            download=download,
            overwrite=overwrite,
            cache=cache,
        )
        self.class_name = class_name
        self.date_start = date_start
        self.date_end = date_end
        self.bands = ["label"]

    @property
    def collection(self):
        return (
            ee.ImageCollection(self.gee_asset_id)
            .filterDate(self.date_start, self.date_end)
            .select(self.bands)
        )

    def _reducer(self, collection: ee.ImageCollection) -> ee.Image:
        """Reduce collection to a single image."""
        image = collection.reduce(ee.Reducer.mode())
        if self.class_name in self.all_bands[:7]:
            class_idx = self.all_bands.index(self.class_name)
            return image.eq(class_idx)
        else:
            return image

    def _preprocess(self):
        """Bypass abstract method."""
        pass
