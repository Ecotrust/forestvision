import sys
from typing import Iterator, Iterable, Callable, Union, Optional
import hashlib
import random

import torch
from geopandas import GeoDataFrame
from torchgeo.samplers import GeoSampler
from torchgeo.datasets import BoundingBox, GeoDataset
from torchgeo.samplers import Units, tile_to_chips
from torchgeo.samplers.utils import _to_tuple
from rtree.index import Index, Property

from .utils import roi_to_tiles


class TileGeoSampler(GeoSampler):
    """Sample dataset using a tile collection."""

    def __init__(
        self,
        dataset: GeoDataset,
        tiles: GeoDataFrame,
        shuffle: bool = False,
    ) -> None:
        """Initialize a new TileGeoSampler instance.

        Args:
            dataset: The dataset to sample from.
            tiles: GeoDataFrame with the tile geometries.
            shuffle: If True, shuffle the tiles before sampling.
        """
        super().__init__(dataset)
        self.shuffle = shuffle
        # Overwrite geodataset index
        self.index = Index(interleaved=False, properties=Property(dimension=3))
        for idx, geom in enumerate(tiles.geometry):
            tile_id = hashlib.md5(geom.bounds.__repr__().encode()).hexdigest()
            mint: float = 0
            maxt: float = sys.maxsize
            minx, miny, maxx, maxy = geom.bounds
            self.index.insert(idx, (minx, maxx, miny, maxy, mint, maxt), tile_id)

        self.hits = []
        for hit in self.index.intersection(tuple(self.roi), objects=True):
            self.hits.append(hit)

    def __iter__(self) -> Iterator[BoundingBox]:
        generator: Callable[[int], Iterable[int]] = range
        if self.shuffle:
            generator = torch.randperm

        for idx in generator(len(self)):
            yield BoundingBox(*self.hits[idx].bounds)

    def __len__(self) -> int:
        return len(self.hits)


class ROIGridGeoSampler(GeoSampler):
    """Sample dataset using a tile grid."""

    def __init__(
        self,
        dataset: GeoDataset,
        size: Union[tuple[float, float], float],
        stride: Union[tuple[float, float], float] = None,
        roi: Optional[BoundingBox] = None,
        sample: Optional[int] = None,
        units: Units = Units.PIXELS,
    ) -> None:
        """
        Initialize a new ROIGridGeoSampler instance.

        dataset: The dataset to sample from.
        size: Dimensions of each tile. If a single value is provided, the same value is used
            for both width and height.
        stride: Distance to move between adjacent tiles. If a single value is provided, the same
            value is used for both dimensions. If not provided, defaults to size.
        roi: Region of interest to sample from (minx, maxx, miny, maxy, mint, maxt). If provided,
            the grid will use the instersection of the roi and the dataset bounds. Defaults to the full
            dataset bounds if None.
        sample: Number of tiles to randomly sample from the grid. If None, use all tiles.
        units: Defines whether size and stride are in pixel or CRS units (defaults to Units.PIXELS).
        """
        super().__init__(dataset)

        size = _to_tuple(size)

        if stride is None:
            stride = size
        stride = _to_tuple(stride)

        # Overwrite GeoSampler roi
        if roi:
            self.roi = self.roi & roi
        else:
            self.roi = dataset.bounds

        if self.res is None:
            raise ValueError(f"{dataset.__class__.__name__} resolution is not defined.")

        if units == Units.PIXELS:
            size = (size[0] * self.res, size[1] * self.res)
            stride = (stride[0] * self.res, stride[1] * self.res)

        rows, cols = tile_to_chips(self.roi, size, stride)
        self.length = rows * cols
        self.tiles = roi_to_tiles(self.roi, size, stride)
        if sample:
            self.tiles = random.sample(self.tiles, sample)
            self.length = sample

    def __iter__(self) -> Iterator[BoundingBox]:
        """Return tile bounds.

        Returns:
            BoundingBox(minx, maxx, miny, maxy, mint, maxt)
        """
        for tile in self.tiles:
            mint = self.roi.mint
            maxt = self.roi.maxt
            minx, miny, maxx, maxy = tile
            yield BoundingBox(minx, maxx, miny, maxy, mint, maxt)

    def __len__(self) -> int:
        return self.length
