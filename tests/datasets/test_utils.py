import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch
from torchgeo.datasets import BoundingBox
from torchgeo.samplers import GeoSampler, Units
from rtree.index import Index, Property

from forestvision.datasets import CloudRasterDataset, DatasetStats
from forestvision.samplers import TileGeoSampler


class MockGeoDataset(CloudRasterDataset):
    """Mock GeoDataset for testing."""

    # Define all_bands to satisfy the bands setter validation
    all_bands = ["B1", "B2", "B3"]

    def __init__(self, is_image=True, bands=None, nodata=None, roi=None):
        """Initialize a new MockGeoDataset instance.

        Args:
            is_image: Whether this dataset returns images or masks.
            bands: List of band names.
            nodata: No data value.
            roi: Region of interest to use for the dataset bounds.
        """
        if roi is None:
            # Default ROI covering Oregon
            roi = BoundingBox(
                minx=-2276145.0,
                maxx=-1587825.0,
                miny=2343855.0,
                maxy=3154575.0,
                mint=0,
                maxt=sys.maxsize,
            )
        
        super().__init__(roi=roi, res=30.0)
        self._crs = "EPSG:5070"
        self.is_image = is_image
        self.bands = bands or ["B1", "B2", "B3"]
        self.nodata = nodata
        self.paths = tempfile.mkdtemp()

        # Create a simple index with one entry
        self.index = Index(interleaved=False, properties=Property(dimension=3))
        self.index.insert(
            0,
            (
                roi.minx,
                roi.maxx,
                roi.miny,
                roi.maxy,
                roi.mint,
                roi.maxt,
            ),
            "roi",
        )

    def _get_pixels(self, query: BoundingBox):
        """Fetch data for the given query.

        Args:
            query: BoundingBox defining an area of interest.

        Returns:
            numpy.ndarray: Pixel data array.
        """
        # Calculate pixel dimensions based on resolution
        height = int((query.maxy - query.miny) / self.res)
        width = int((query.maxx - query.minx) / self.res)

        # Create a numpy array with known values for predictable stats
        import numpy as np
        channels = len(self.bands)
        if self.is_image:
            data = np.ones((channels, height, width), dtype=np.float32) * 5.0
            # Add some variation for testing
            if channels > 1:
                data[1] = data[1] * 2.0  # Second channel has value 10.0

            # Add some nodata values if specified
            if self.nodata is not None:
                data[0, 0, 0] = self.nodata
        else:
            data = np.ones((1, height, width), dtype=np.int64)

        return data

    def __getitem__(self, query: BoundingBox):
        """Return a sample from the dataset.

        Args:
            query: BoundingBox defining an area of interest.

        Returns:
            A sample containing the data and metadata at that location.
        """
        # Calculate pixel dimensions based on resolution
        height = int((query.maxy - query.miny) / self.res)
        width = int((query.maxx - query.minx) / self.res)

        # Create a tensor with known values for predictable stats
        channels = len(self.bands)
        if self.is_image:
            data = torch.ones(channels, height, width) * 5.0
            # Add some variation for testing
            if channels > 1:
                data[1] = data[1] * 2.0  # Second channel has value 10.0

            # Add some nodata values if specified
            if self.nodata is not None:
                data[0, 0, 0] = self.nodata

            key = "image"
        else:
            data = torch.ones(1, height, width).long()
            key = "mask"

        return {key: data, "crs": self._crs, "bbox": query}


class TestDatasetStats:
    @pytest.fixture
    def dataset(self):
        """Create a mock dataset for testing."""
        return MockGeoDataset()

    @pytest.fixture
    def sampler(self, tiles, dataset):
        """Create a mock sampler for testing."""
        return TileGeoSampler(dataset, tiles=tiles)

    @pytest.fixture
    def stats_path(self):
        """Create a temporary path for saving stats."""
        with tempfile.TemporaryDirectory() as d:
            yield os.path.join(d, "stats.pt")

    def test_init(self, dataset, sampler):
        """Test DatasetStats initialization."""
        stats = DatasetStats(dataset, sampler)

        # Check that the dataloader was initialized correctly
        assert stats.dataloader is not None
        assert stats.data_key == "image"
        assert stats.path is not None  # Should be set to dataset.paths/stats.pt
        assert stats.samples == 6  # Default length of MockGeoSampler
        assert stats.dim == (0, 2, 3)  # Default dimensions
        assert stats.nodata is None  # Default nodata value

        # Check that the stats tensors were initialized correctly
        assert stats._sum.shape == torch.Size([3])  # 3 channels
        assert stats._sum_sq.shape == torch.Size([3])
        assert stats._count.shape == torch.Size([3])
        assert stats._min.shape == torch.Size([3])
        assert stats._max.shape == torch.Size([3])

    def test_init_with_mask_dataset(self, sampler):
        """Test DatasetStats initialization with a mask dataset."""
        dataset = MockGeoDataset(is_image=False)
        stats = DatasetStats(dataset, sampler)

        assert stats.data_key == "mask"

    def test_init_with_custom_path(self, dataset, sampler, stats_path):
        """Test DatasetStats initialization with a custom path."""
        stats = DatasetStats(dataset, sampler, path=stats_path)
        assert stats.path == stats_path

    def test_init_with_custom_channels(self, dataset, sampler):
        """Test DatasetStats initialization with custom channels."""
        stats = DatasetStats(dataset, sampler, channels=3)

        assert stats._sum.shape == torch.Size([3])
        assert stats._sum_sq.shape == torch.Size([3])
        assert stats._count.shape == torch.Size([3])
        assert stats._min.shape == torch.Size([3])
        assert stats._max.shape == torch.Size([3])

    def test_init_with_custom_nodata(self, sampler):
        """Test DatasetStats initialization with custom nodata value."""
        dataset = MockGeoDataset(nodata=-9999)
        stats = DatasetStats(dataset, sampler)
        assert stats.nodata == -9999

        # Override dataset nodata value
        stats = DatasetStats(dataset, sampler, nodata=-1)
        assert stats.nodata == -1

    def test_init_with_custom_dims(self, dataset, sampler):
        """Test DatasetStats initialization with custom dimensions."""
        stats = DatasetStats(dataset, sampler, on_dims=(0, 1))
        assert stats.dim == (0, 1)

    def test_compute(self, dataset, sampler):
        """Test DatasetStats compute method."""
        stats = DatasetStats(dataset, sampler)
        result = stats.compute()

        # Check that the result contains the expected keys
        assert "mean" in result
        assert "std" in result
        assert "min" in result
        assert "max" in result
        assert "nodata" in result
        assert "sample_size" in result

        # All values are 5.0 except for the second channel which is 10.0
        assert torch.allclose(result["mean"][0], torch.tensor(5.0))
        if len(result["mean"]) > 1:
            assert torch.allclose(result["mean"][1], torch.tensor(10.0))

        # Standard deviation should be close to 0
        assert torch.allclose(result["std"], torch.tensor(0.0), atol=1e-2)

        # Min and max values
        assert torch.allclose(result["min"][0], torch.tensor(5.0))
        assert torch.allclose(result["max"][0], torch.tensor(5.0))
        if len(result["min"]) > 1:
            assert torch.allclose(result["min"][1], torch.tensor(10.0))
            assert torch.allclose(result["max"][1], torch.tensor(10.0))

        # Sample size should match the sampler length
        assert result["sample_size"] == 6

    def test_compute_with_nodata(self, sampler):
        """Test DatasetStats compute method with nodata values."""
        dataset = MockGeoDataset(nodata=-9999)
        stats = DatasetStats(dataset, sampler)
        result = stats.compute()

        # Check that the nodata value is correctly reported
        assert result["nodata"] == -9999
        assert torch.allclose(result["mean"][0], torch.tensor(5.0))

    def test_compute_save_and_load(self, dataset, sampler, stats_path):
        """Test saving and loading stats."""
        # Compute and save stats
        stats = DatasetStats(dataset, sampler, path=stats_path)
        result1 = stats.compute()

        # Check that the file was created
        assert os.path.exists(stats_path)

        # Create a new DatasetStats instance and load the saved stats
        stats2 = DatasetStats(dataset, sampler, path=stats_path)
        result2 = stats2.compute()

        # Check that the loaded stats match the original stats
        assert torch.allclose(result1["mean"], result2["mean"])
        assert torch.allclose(result1["std"], result2["std"])
        assert torch.allclose(result1["min"], result2["min"])
        assert torch.allclose(result1["max"], result2["max"])
        assert result1["nodata"] == result2["nodata"]
        assert result1["sample_size"] == result2["sample_size"]

    def test_compute_with_overwrite(self, dataset, sampler, stats_path):
        """Test overwriting existing stats."""
        # Compute and save stats
        stats1 = DatasetStats(dataset, sampler, path=stats_path)
        stats1.compute()

        # Create a new dataset with different values (use only valid bands)
        dataset2 = MockGeoDataset(bands=["B1", "B2"])

        # Create a new DatasetStats instance with overwrite=True
        stats2 = DatasetStats(dataset2, sampler, path=stats_path, overwrite=True)
        result = stats2.compute()

        # Check that the stats were recomputed
        assert result["mean"].shape[0] == 2  # Should have 2 channels now

    def test_error_no_channels(self, sampler):
        """Test error when no channel information is available."""
        # Create a dataset without bands attribute by creating a minimal GeoDataset
        from torchgeo.datasets import GeoDataset
        
        class MinimalGeoDataset(GeoDataset):
            def __init__(self):
                super().__init__()
                self._crs = "EPSG:5070"
                self.is_image = True  # Add is_image attribute
                self.paths = tempfile.mkdtemp()  # Add paths attribute
                self.index = Index(interleaved=False, properties=Property(dimension=3))
                self.index.insert(0, (-2276145.0, -1587825.0, 2343855.0, 3154575.0, 0, sys.maxsize), "roi")
            
            def __getitem__(self, query):
                return {"image": torch.rand(3, 64, 64), "crs": self._crs, "bbox": query}

        dataset = MinimalGeoDataset()

        # Should raise ValueError when initializing DatasetStats
        with pytest.raises(ValueError, match="No channel information available"):
            DatasetStats(dataset, sampler)

    def test_error_invalid_path(self, dataset, sampler):
        """Test error when path is invalid."""
        # Create a DatasetStats instance with a non-existent path
        stats = DatasetStats(dataset, sampler)
        stats.path = "/non/existent/path/stats.pt"
        stats.overwrite = False

        # Should raise RuntimeError when computing stats
        with pytest.raises(RuntimeError, match="Parent directory .* does not exist"):
            stats.compute()
