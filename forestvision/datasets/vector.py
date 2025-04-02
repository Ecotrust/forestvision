import os
import sys
from io import StringIO
import geopandas as gpd
from geopandas import GeoDataFrame
from rasterio.crs import CRS
from torchgeo.datasets import BoundingBox
from sklearn.model_selection import train_test_split


class GPDFeatureCollection:
    """Helper class to handle pandas GeoDataFrame objects."""

    def __init__(self, tiles: str | GeoDataFrame, sample: int = None, crs: CRS = None):
        """
        Initialize a new GPDFeatureCollection instance.

        Parameters:
            tiles (str | GeoDataFrame): Input tile data or path to a GeoJSON-like file.
            sample (int, optional): Select a random sample of tiles. Defaults to None.
            crs (CRS, optional): Coordinate reference system to project data into. Defaults to None.
        """
        if isinstance(tiles, str):
            tiles = gpd.read_file(tiles)
        if sample:
            self.data = tiles.sample(sample)
        else:
            self.data = tiles
        if crs:
            self.data = self.data.to_crs(crs)

        self.crs = self.data.crs

    @property
    def bounds(self):
        minx, miny, maxx, maxy = self.data.total_bounds
        return BoundingBox(minx, maxx, miny, maxy, 0, sys.maxsize)

    @property
    def shape(self):
        return self.data.shape

    def split(self, **kwargs):
        return train_test_split(self.data, **kwargs)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __str__(self):
        buf = StringIO()
        self.data.info(buf=buf)
        parts = buf.getvalue().split("\n")
        info = "\n".join([f"  {c}" for c in parts[1:-1]])
        data_cls = self.data.__class__.__name__
        return (
            f"{self.__class__.__name__}\n"
            f"data: {data_cls}\n"
            f"{info}\n"
            f"crs: {self.crs}\n"
            f"bounds:\n"
            f"  minx: {self.bounds.minx}\n"
            f"  maxx: {self.bounds.maxx}\n"
            f"  miny: {self.bounds.miny}\n"
            f"  maxy: {self.bounds.maxy}\n"
            f"  mint: {self.bounds.mint}\n"
            f"  maxt: {self.bounds.maxt}\n"
        )

    def __getitem__(self, idx):
        minx, miny, maxx, maxy = self.data.iloc[idx].geometry.bounds
        return BoundingBox(minx, maxx, miny, maxy, 0, sys.maxsize)

    def plot(self, **kwargs):
        return self.data.plot(**kwargs)
