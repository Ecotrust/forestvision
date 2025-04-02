import abc
import functools
import hashlib
from typing import Any, Callable, Optional, cast, List, Tuple

import numpy
import torch
import rasterio
from rasterio.crs import CRS
from rasterio.io import DatasetReader
from rasterio.vrt import WarpedVRT
from torchgeo.datasets import GeoDataset
from torchgeo.datasets.utils import BoundingBox


class CloudRasterDataset(GeoDataset):
    """Abstract base class for imagery served from cloud data providers.

    This class provides an abstract interface for fetching geospatial imagery
    from cloud data providers like Google Earth Engine. The dataset doesn't need
    to exist locally, but requires metadata to define and fetch data from cloud services.

    Attributes:
        is_image (bool): Whether this dataset contains image data (True) or mask data (False).
        all_bands (List[str]): List of all available bands in the dataset.
        rgb_bands (List[str]): List of band names to use for RGB visualization.
        cmap (dict[int, Tuple[int, int, int, int]]): Color map for visualization.
        _tile_size (Optional[Tuple[int, int]]): Internal tile size storage.
        _api_url (str): Internal API URL storage.
        _bands (List[str]): Internal bands storage.
        _res (float): Internal resolution storage.
    """

    is_image = True

    all_bands: List[str] = []

    rgb_bands: List[str] = []

    cmap: dict[int, Tuple[int, int, int, int]] = {}

    _tile_size: Optional[Tuple[int, int]] = None

    _api_url: str = None

    _bands: List[str] = []

    _res: float = None

    @property
    def dtype(self) -> torch.dtype:
        """Get the data type of the dataset.

        Overrides the dtype of the data file via a cast. Defaults to float32
        if :attr:`~CloudRasterDataset.is_image` is True, else long.

        Returns:
            torch.dtype: The data type of the dataset.
        """
        if self.is_image:
            return torch.float32
        else:
            return torch.long

    def __init__(
        self,
        roi: BoundingBox,
        path: Optional[str] = None,
        res: Optional[float] = None,
        transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        crs: Optional[CRS] = None,
        download: bool = False,
        cache: bool = True,
    ) -> None:
        """Initialize a new CloudRasterDataset instance.

        Args:
            roi (BoundingBox): A bounding box defining the region of interest.
            path (str, optional): Path to store downloaded data. Required if download is True.
            res (float, optional): Resolution of the dataset in meters per pixel.
            transforms (Callable, optional): Callable transform applied to each sample.
            crs (CRS, optional): Coordinate reference system of the dataset.
            download (bool, optional): Whether to download dataset files. Defaults to False.
            cache (bool, optional): If True, cache pixel data in memory. Defaults to True.

        Raises:
            ValueError: If download is True and path is not provided.
            ValueError: If the provided ROI is invalid.
        """
        self.paths = path
        self.cache = cache
        self.roi = roi
        self._res = res
        self._download = download
        super().__init__(transforms=transforms)

        if self._download and path is None:
            msg = f"Set path to store {self.__class__.__name__} images."
            raise ValueError(msg)

        try:
            minx, maxx, miny, maxy, mint, maxt = roi
        except Exception as e:
            raise ValueError(f"Provide a valid ROI: {e}")

        roi_id = hashlib.md5((minx, maxx, miny, maxy).__str__().encode()).hexdigest()
        self.index.insert(0, (minx, maxx, miny, maxy, mint, maxt), roi_id)
        self._crs = cast(CRS, crs)
        self._res = cast(float, res)

    @property
    def bands(self) -> List[str]:
        """Get the bands to use for data fetching.

        Returns:
            List[str]: List of band names currently selected for use.
        """
        return self._bands

    @bands.setter
    def bands(self, value: List[str]):
        """Set the bands to use for data fetching.

        Args:
            value (List[str]): List of band names to use.

        Raises:
            ValueError: If any band in value is not available in the dataset.
        """
        invalid_bands = set(value) - set(self.all_bands)
        if invalid_bands:
            raise ValueError(f"Invalid bands: {invalid_bands}")
        self._bands = value

    @property
    def res(self) -> float:
        """Get the resolution of the dataset.

        Returns:
            float: Resolution in meters per pixel.
        """
        return self._res

    @res.setter
    def res(self, value: float):
        """Set the resolution of the dataset.

        Args:
            value (float): Resolution in meters per pixel.

        Raises:
            AssertionError: If value is not greater than 0.
        """
        assert value > 0, "Resolution must be greater than 0"
        self._res = value

    @property
    def tile_size(self) -> Tuple[int, int]:
        """Get the size of tiles in the tiles collection.

        Returns:
            Tuple[int, int]: Tile dimensions in pixels (width, height).
        """
        if self._tile_size and self.res:
            self._tile_size = tuple([int(x // self.res) for x in self._tile_size])
        return self._tile_size

    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query (BoundingBox): Bounding box defining the region to retrieve.

        Returns:
            dict[str, Any]: Sample dictionary containing:
                - 'crs': Coordinate reference system
                - 'bbox': Bounding box of the sample
                - 'image' or 'mask': Tensor data depending on is_image flag
        """
        sample = {"crs": self.crs, "bbox": query}

        if self.cache:
            data = self._cached_get_pixels(query)
        else:
            data = self._get_pixels(query)

        data = torch.from_numpy(data).to(self.dtype)

        if self.is_image:
            sample["image"] = data
        else:
            sample["mask"] = data

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    @functools.lru_cache(maxsize=128)
    def _cached_get_pixels(self, query: BoundingBox) -> numpy.ndarray:
        """Cached version of `_get_pixels`.

        Args:
            query (BoundingBox): Bounding box defining the region to retrieve.

        Returns:
            numpy.ndarray: Pixel data array.

        Note:
            Uses LRU caching with a maximum size of 128 entries.
        """
        return self._get_pixels(query)

    @abc.abstractmethod
    def _get_pixels(self, query: BoundingBox) -> numpy.ndarray:
        """Fetch data from cloud data provider or local file.

        Args:
            query (BoundingBox): Bounding box defining the region to fetch.

        Returns:
            numpy.ndarray: Pixel data array.

        Note:
            This is an abstract method that must be implemented by subclasses.
        """
        pass

    def _load_warp_file(self, filepath: str) -> DatasetReader:
        """Load and warp a file to the correct CRS and resolution.

        Args:
            filepath (str): Path to the file to load and warp.

        Returns:
            DatasetReader: File handle of warped VRT.

        Note:
            Only warps the file if the source CRS differs from the target CRS.
        """
        src = rasterio.open(filepath)

        # Only warp if necessary
        if src.crs != self.crs:
            vrt = WarpedVRT(src, crs=self.crs)
            src.close()
            return vrt
        else:
            return src
