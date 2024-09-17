"""
"""

import abc
import sys
import functools
import hashlib
from typing import Any, Callable, Optional, Union, cast, List, Tuple

import torch
import pystac
import geopandas
import rasterio
from rasterio.crs import CRS
from rasterio.io import DatasetReader
from rasterio.vrt import WarpedVRT
from torchgeo.datasets import GeoDataset
from torchgeo.datasets.utils import BoundingBox


class CloudRasterDataset(GeoDataset):
    """Abstract base class for imagery served from cloud data providers like
    Google Earth Engine and MS Planetary Computer.

    The dataset doesn't need to exist in the local machine, however some metadata
    is required to define and fetch image tiles from a cloud service.

    Expecting to support geojson and STAC collections for pre-defined tiles. This
    class is also compatible with Torchgeo geosamplers.
    """

    #: Date format string used to parse date from filename.
    date_format = "%Y%m%d"

    #: True if the dataset only contains model inputs (such as images). False if the
    #: dataset only contains ground truth model outputs (such as segmentation masks).
    #:
    #: The sample returned by the dataset/data loader will use the "image" key if
    #: *is_image* is True, otherwise it will use the "mask" key.
    #:
    #: For datasets with both model inputs and outputs, a custom
    #: :func:`~RasterDataset.__getitem__` method must be implemented.
    is_image = True

    all_bands: List[str] = []

    #: Names of RGB bands in the dataset, used for plotting
    rgb_bands: List[str] = []

    #: Color map for the dataset, used for plotting
    cmap: dict[int, Tuple[int, int, int, int]] = {}

    #: Populated when a tile collection is provided.
    _tile_size: Optional[Tuple[int, int]] = None

    #: Cloud-based platform API URL
    _api_url: str = None

    _bands: List[str] = []

    _res: float = None

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the dataset (overrides the dtype of the data file via a cast).

        Defaults to float32 if :attr:`~RasterDataset.is_image` is True, else long.
        Can be overridden for tasks like pixel-wise regression where the mask should be
        float32 instead of long.

        Returns:
            the dtype of the dataset
        """
        if self.is_image:
            return torch.float32
        else:
            return torch.long

    def __init__(
        self,
        path: Optional[str] = None,
        tiles: Optional[Union[geopandas.GeoDataFrame, pystac.Collection]] = None,
        roi: Optional[BoundingBox] = None,
        res: Optional[float] = None,
        transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        crs: Optional[CRS] = None,
        download: bool = False,
        cache: bool = True,
    ) -> None:
        """Initialize a new CloudRasterDataset instance.

        Args:
            path: str
                Path to store downloaded dataset.
            tiles: Union[geopandas.GeoDataFrame, pystac.Collection]
                GeoDataFrame containing the tile collection.
            roi: BoundingBox
                Region of interest to fetch data from.
            res: float
                Resolution of the dataset.
            transforms: Callable[[dict[str, Any]], dict[str, Any]]
                A function/transform that takes in a sample and returns a transformed version.
            crs: CRS
                Coordinate Reference System of the dataset.
            download: bool
                If True, download the dataset to the path directory.
            cache: bool
                If True, cache the dataset in memory.

        Raises:
            AssertionError: If download is True and path is not set.
            ValueError: If a tile collection or ROI bounds are not provided.
        """
        self.paths = path
        self.cache = cache
        self.roi = roi
        self._res = res
        self._download = download
        super().__init__(transforms=transforms)

        if self._download:
            if path is None:
                msg = f"Set path to store {self.__class__.__name__} images."
                raise AssertionError(msg)

        # Populate the dataset index
        if isinstance(tiles, geopandas.GeoDataFrame):
            if crs is not None and crs != tiles.crs:
                print("Reprojecting tiles to match dataset CRS")
                tiles = tiles.to_crs(crs)

            for idx, geom in enumerate(tiles.geometry):
                tile_id = hashlib.md5(geom.bounds.__repr__().encode()).hexdigest()
                mint: float = 0
                maxt: float = sys.maxsize
                minx, miny, maxx, maxy = geom.bounds
                self.index.insert(idx, (minx, maxx, miny, maxy, mint, maxt), tile_id)
                if idx == 0:
                    self._tile_size = (maxx - minx, maxy - miny)

        elif isinstance(tiles, pystac.Collection):
            raise NotImplementedError("STAC collections are not yet supported")

        elif roi:
            minx, maxx, miny, maxy, mint, maxt = roi
            roi_id = hashlib.md5(
                (minx, miny, maxx, maxy).__str__().encode()
            ).hexdigest()
            self.index.insert(0, (minx, maxx, miny, maxy, mint, maxt), roi_id)

        else:
            raise ValueError(
                "ROI bounds must be provided if a tile collection is not provided"
            )

        self._crs = cast(CRS, crs)
        self._res = cast(float, res)

    @property
    def bands(self):
        """Select bands to use."""
        return self._bands

    @bands.setter
    def bands(self, value: List[str]):
        invalid_bands = set(value) - set(self.all_bands)
        if invalid_bands:
            raise ValueError(f"Invalid bands: {invalid_bands}")
        self._bands = value

    @property
    def res(self) -> float:
        """Resolution of the dataset."""
        return self._res

    @res.setter
    def res(self, value: float):
        assert value > 0, "Resolution must be greater than 0"
        self._res = value

    @property
    def tile_size(self) -> int:
        """Size of tiles in tiles collection."""
        if self._tile_size and self.res:
            self._tile_size = tuple([int(x // self.res) for x in self._tile_size])
        return self._tile_size

    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query."""

        sample = {"crs": self.crs, "bbox": query}

        if self.cache:
            data, transform = self._cached_get_pixels(query)
        else:
            data, transform = self._get_pixels(query)

        data = torch.from_numpy(data).to(self.dtype)
        sample["transform"] = transform
        if self.is_image:
            sample["image"] = data
        else:
            sample["mask"] = data

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    @functools.lru_cache(maxsize=128)
    def _cached_get_pixels(self, query: BoundingBox) -> torch.Tensor:
        """Cached version of `_get_pixels`."""
        return self._get_pixels(query)

    @abc.abstractmethod
    def _get_pixels(self, query: BoundingBox) -> torch.Tensor:
        """Fetch data from cloud data provider or local file."""

    def _load_warp_file(self, filepath: str) -> DatasetReader:
        """Load and warp a file to the correct CRS and resolution.

        Args:
            filepath: file to load and warp

        Returns:
            file handle of warped VRT
        """
        src = rasterio.open(filepath)

        # Only warp if necessary
        if src.crs != self.crs:
            vrt = WarpedVRT(src, crs=self.crs)
            src.close()
            return vrt
        else:
            return src
