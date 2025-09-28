import sys

import torch
from torchgeo.datasets import BoundingBox
from torchgeo.samplers import Units
from rtree.index import Index, Property

from forestvision.datasets import CloudRasterDataset
from forestvision.samplers import TileGeoSampler, ROIGridGeoSampler


class MockGeoDataset(CloudRasterDataset):
    """Mock GeoDataset for testing."""

    # Define all_bands to satisfy the bands setter validation
    all_bands = ["B1", "B2", "B3"]

    def __init__(self, roi=None):
        """Initialize a new MockGeoDataset instance.

        Args:
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

        # Create a simple numpy array with random values
        import numpy as np
        image = np.random.rand(3, height, width).astype(np.float32)
        return image

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

        # Create a simple tensor with random values
        image = torch.rand(3, height, width)

        return {"image": image, "crs": self._crs, "bbox": query}


class TestTileGeoSampler:
    def test_init(self, roi, tiles):
        """Test TileGeoSampler initialization."""
        dataset = MockGeoDataset(roi)
        sampler = TileGeoSampler(dataset, tiles)

        # Check that the sampler was initialized correctly
        assert len(sampler.hits) > 0
        assert sampler.shuffle is False

    def test_len(self, tiles):
        """Test TileGeoSampler __len__ method."""
        dataset = MockGeoDataset()
        sampler = TileGeoSampler(dataset, tiles)

        # Length should match the number of tiles
        assert len(sampler) == len(sampler.hits)
        assert len(sampler) > 0

    def test_iter(self, tiles):
        """Test TileGeoSampler __iter__ method."""
        dataset = MockGeoDataset()
        sampler = TileGeoSampler(dataset, tiles)

        # Get all bounding boxes from the iterator
        bboxes = list(sampler)

        # Check that we got the expected number of bounding boxes
        assert len(bboxes) == len(sampler)

        # Check that bounding box is valid
        bbox = bboxes[0]
        assert isinstance(bbox, BoundingBox)
        assert bbox.minx < bbox.maxx
        assert bbox.miny < bbox.maxy
        assert bbox.mint <= bbox.maxt

    def test_with_shuffle(self, tiles):
        """Test TileGeoSampler with shuffle=True."""
        dataset = MockGeoDataset()
        sampler = TileGeoSampler(dataset, tiles, shuffle=True)

        # Check that shuffle flag is set
        assert sampler.shuffle is True
        assert list(sampler) != list(sampler)

    def test_with_dataset(self, tiles):
        """Test TileGeoSampler with a dataset."""
        dataset = MockGeoDataset()
        sampler = TileGeoSampler(dataset, tiles)

        # Get the first bounding box
        bbox = next(iter(sampler))

        # Get the corresponding sample from the dataset
        sample = dataset[bbox]

        # Check that the sample is valid
        assert "image" in sample
        assert isinstance(sample["image"], torch.Tensor)
        assert sample["image"].shape[0] == 3  # 3 channels
        assert sample["image"].shape[1] > 0  # Height
        assert sample["image"].shape[2] > 0  # Width
        assert sample["crs"] == dataset._crs
        assert sample["bbox"] == bbox


class TestROIGridGeoSampler:
    size = 64
    stride = 54

    def test_init(self):
        """Test ROIGridGeoSampler initialization."""
        dataset = MockGeoDataset()
        sampler = ROIGridGeoSampler(dataset, self.size)

        # Check that the sampler was initialized correctly
        assert len(sampler.tiles) > 0
        assert sampler.length > 0

    def test_init_with_stride(self, dunes):
        """Test ROIGridGeoSampler initialization with stride."""
        dataset = MockGeoDataset(dunes)
        sampler = ROIGridGeoSampler(dataset, self.size, self.stride)

        # Check that the sampler was initialized correctly
        assert len(sampler.tiles) == 9
        assert len(sampler.tiles) == sampler.length

        # Check that the stride was applied correctly
        items = [b for b in sampler]
        b1 = items[0]
        b2 = items[1]
        assert abs(b2.miny - b1.miny) // dataset.res == self.stride

    def test_init_with_roi(self, dunes):
        """Test ROIGridGeoSampler initialization with custom ROI."""
        dataset = MockGeoDataset()
        sampler = ROIGridGeoSampler(dataset, self.size, roi=dunes)

        # Check that the sampler was initialized correctly
        assert sampler.roi == dunes
        assert len(sampler.tiles) == 9
        assert sampler.length == len(sampler.tiles)
        assert dataset.bounds.minx <= sampler.roi.minx
        assert dataset.bounds.maxx >= sampler.roi.maxx
        assert dataset.bounds.miny <= sampler.roi.miny
        assert dataset.bounds.maxy >= sampler.roi.maxy

    def test_init_with_sample(self):
        """Test ROIGridGeoSampler initialization with sample parameter."""
        dataset = MockGeoDataset()
        sample = 5
        sampler = ROIGridGeoSampler(dataset, self.size, sample=sample)

        assert len(sampler.tiles) == sample
        assert sampler.length == sample

    def test_init_with_units(self, dunes):
        """Test ROIGridGeoSampler initialization with different units."""
        dataset = MockGeoDataset(dunes)

        # Test with pixel units
        sampler_pixels = ROIGridGeoSampler(dataset, self.size, units=Units.PIXELS)

        # Test with CRS units
        size_crs = self.size * dataset.res
        sampler_crs = ROIGridGeoSampler(dataset, size_crs, units=Units.CRS)

        assert len(sampler_pixels.tiles) == len(sampler_crs.tiles)

    def test_len(self, dunes):
        """Test ROIGridGeoSampler __len__ method."""
        dataset = MockGeoDataset(dunes)
        sampler = ROIGridGeoSampler(dataset, self.size)

        # Length should match the number of tiles
        assert len(sampler) == len(sampler.tiles)
        assert len(sampler) > 0

    def test_iter(self, dunes):
        """Test ROIGridGeoSampler __iter__ method."""
        dataset = MockGeoDataset(dunes)
        sampler = ROIGridGeoSampler(dataset, self.size)

        # Get all bounding boxes from the iterator
        bboxes = list(sampler)

        # Check that we got the expected number of bounding boxes
        assert len(bboxes) == len(sampler)

        # Check that bounding box is valid
        bbox = bboxes[0]
        assert isinstance(bbox, BoundingBox)
        assert bbox.minx < bbox.maxx
        assert bbox.miny < bbox.maxy
        assert bbox.mint <= bbox.maxt

    def test_with_dataset(self, dunes):
        """Test ROIGridGeoSampler with a dataset."""
        dataset = MockGeoDataset(dunes)
        sampler = ROIGridGeoSampler(dataset, self.size)

        # Get the first bounding box
        bbox = next(iter(sampler))

        # Get the corresponding sample from the dataset
        sample = dataset[bbox]

        # Check that the sample is valid
        assert "image" in sample
        assert isinstance(sample["image"], torch.Tensor)
        assert sample["image"].shape[0] == 3  # 3 channels
        assert sample["image"].shape[1] > 0  # Height
        assert sample["image"].shape[2] > 0  # Width
        assert sample["crs"] == dataset._crs
        assert sample["bbox"] == bbox
