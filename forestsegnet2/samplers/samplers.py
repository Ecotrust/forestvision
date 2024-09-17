import sys
from typing import Iterator, Iterable, Callable
import hashlib

import torch
from rasterio.crs import CRS
from geopandas import GeoDataFrame
from torchgeo.samplers import GeoSampler
from torchgeo.datasets import BoundingBox, GeoDataset
from rtree.index import Index, Property


class TileGeoSampler(GeoSampler):

    def __init__(
        self,
        dataset: GeoDataset,
        tiles: GeoDataFrame,
        shuffle: bool = False,
    ) -> None:
        super().__init__(dataset)
        self.shuffle = shuffle
        # Replace geodataset index
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


if __name__ == "__main__":
    import geopandas as gpd
    from forestsegnet2.datasets import eMapRAGB
    from torch.utils.data import DataLoader
    from torchgeo.datasets import BoundingBox, stack_samples

    tiles = gpd.read_file("data/vector/tiles_80x80.geojson")
    agb = eMapRAGB(year=2018)
    sampler = TileGeoSampler(agb, tiles)
    ds = DataLoader(agb, sampler=sampler, collate_fn=stack_samples)
    for sample in ds:
        print(sample)
