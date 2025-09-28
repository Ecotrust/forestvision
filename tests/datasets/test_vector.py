import os
import sys
import pytest
import geopandas as gpd
from geopandas import GeoDataFrame
from rasterio.crs import CRS
from torchgeo.datasets import BoundingBox

from forestvision.datasets.vector import GPDFeatureCollection


class TestGPDFeatureCollection:
    @pytest.fixture
    def tiles_path(self) -> str:
        """Path to the test tiles GeoJSON file."""
        return os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "..", "data", "tiles", "tiles.geojson"
            )
        )
        # return os.path.join("data", "tiles", "tiles.geojson")

    @pytest.fixture
    def tiles_gdf(self, tiles_path) -> GeoDataFrame:
        """Load the test tiles as a GeoDataFrame."""
        return gpd.read_file(tiles_path)

    def test_init_with_path(self, tiles_path):
        """Test initialization with a file path."""
        collection = GPDFeatureCollection(tiles_path)
        assert isinstance(collection, GPDFeatureCollection)
        assert isinstance(collection.data, GeoDataFrame)
        assert collection.crs is not None

    def test_init_with_gdf(self, tiles_gdf):
        """Test initialization with a GeoDataFrame."""
        collection = GPDFeatureCollection(tiles_gdf)
        assert isinstance(collection, GPDFeatureCollection)
        assert isinstance(collection.data, GeoDataFrame)
        assert collection.crs is not None
        assert collection.data.equals(tiles_gdf)

    def test_init_with_sample(self, tiles_path):
        """Test initialization with a sample."""
        sample_size = 3
        collection = GPDFeatureCollection(tiles_path, sample=sample_size)
        assert isinstance(collection, GPDFeatureCollection)
        assert len(collection.data) == sample_size

    def test_init_with_crs(self, tiles_path):
        """Test initialization with a CRS."""
        target_crs = CRS.from_epsg(4326)  # WGS84
        collection = GPDFeatureCollection(tiles_path, crs=target_crs)
        assert isinstance(collection, GPDFeatureCollection)
        assert collection.crs == target_crs
        assert collection.data.crs == target_crs

    def test_bounds_property(self, tiles_path):
        """Test the bounds property."""
        collection = GPDFeatureCollection(tiles_path)
        bounds = collection.bounds
        assert isinstance(bounds, BoundingBox)
        assert bounds.minx <= bounds.maxx
        assert bounds.miny <= bounds.maxy
        assert bounds.mint <= bounds.maxt

    def test_shape_property(self, tiles_path):
        """Test the shape property."""
        collection = GPDFeatureCollection(tiles_path)
        shape = collection.shape
        assert isinstance(shape, tuple)
        assert len(shape) == 2
        assert shape[0] == len(collection.data)
        assert shape[1] == len(collection.data.columns)

    def test_split_method(self, tiles_path):
        """Test the split method."""
        collection = GPDFeatureCollection(tiles_path)
        train, test = collection.split(test_size=0.3, random_state=42)
        assert isinstance(train, GeoDataFrame)
        assert isinstance(test, GeoDataFrame)
        assert len(train) + len(test) == len(collection.data)

    def test_len_method(self, tiles_path):
        """Test the __len__ method."""
        collection = GPDFeatureCollection(tiles_path)
        assert len(collection) == len(collection.data)

    def test_iter_method(self, tiles_path):
        """Test the __iter__ method."""
        collection = GPDFeatureCollection(tiles_path)
        items = list(collection)
        assert len(items) == len(collection)
        assert all(isinstance(item, BoundingBox) for item in items)

    def test_str_method(self, tiles_path):
        """Test the __str__ method."""
        collection = GPDFeatureCollection(tiles_path)
        string_repr = str(collection)
        assert isinstance(string_repr, str)
        assert collection.__class__.__name__ in string_repr
        assert "data:" in string_repr
        assert "crs:" in string_repr
        assert "bounds:" in string_repr

    def test_getitem_method(self, tiles_path):
        """Test the __getitem__ method."""
        collection = GPDFeatureCollection(tiles_path)
        item = collection[0]
        assert isinstance(item, BoundingBox)
        assert item.minx <= item.maxx
        assert item.miny <= item.maxy
        assert item.mint <= item.maxt

    def test_plot_method(self, tiles_path):
        """Test the plot method."""
        import matplotlib

        matplotlib.use("agg")  # Use non-interactive backend for testing

        collection = GPDFeatureCollection(tiles_path)
        ax = collection.plot()
        assert ax is not None
