import os
import sys
import geopandas as gpd
from torchgeo.datasets import BoundingBox
from sklearn.model_selection import train_test_split


class GPDFeatureCollection:
    def __init__(self, tile_path: str, sample_size: int = None):
        self.tile_path = tile_path
        if sample_size:
            self.data = gpd.read_file(tile_path).sample(sample_size)
        else:
            self.data = gpd.read_file(tile_path)

    @property
    def bounds(self):
        minx, miny, maxx, maxy = self.data.total_bounds
        return BoundingBox(minx, miny, maxx, maxy, 0, sys.maxsize)

    def split(self, test_size: float = 0.2, random_state=42, **kwargs):
        return train_test_split(self.data, **kwargs)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __repr__(self):
        return f"Tiles(tile_path={self.tile_path})"

    def __str__(self):
        return f"Tiles(tile_path={self.tile_path})"

    def __getitem__(self, idx):
        minx, miny, maxx, maxy = self.data.iloc[idx].geometry.bounds
        return BoundingBox(minx, maxx, miny, maxy, 0, sys.maxsize)


if __name__ == "__main__":
    TILES80x80 = "data/vector/tiles_80x80.geojson"
    tiles = GPDFeatureCollection(TILES80x80, sample_size=100)
    tiles.split()
    iter_tiles = iter(tiles)
    item = next(iter_tiles)
    print(item)
    train, test = tiles.split()
    print(train.shape)
    print(test.shape)
