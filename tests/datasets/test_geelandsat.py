import os
from unittest.mock import Mock, patch
import pytest
import torch
from torch import nn
import matplotlib.pyplot as plt
import matplotlib
from rasterio.crs import CRS

from torchgeo.datasets import (
    BoundingBox,
    IntersectionDataset,
    UnionDataset,
)

from forestvision.datasets import GEELandsat8, CloudRasterDataset, GEELandsatFTV


class TestGEELandsat8:
    @pytest.fixture
    def dataset(self, tile, request, monkeypatch) -> CloudRasterDataset:
        # Mock Earth Engine initialization and API calls
        mock_ee = Mock()
        mock_image_collection = Mock()
        mock_image = Mock()

        # Mock band names
        mock_bands = Mock()
        mock_bands.getInfo.return_value = ["B1", "B2", "B3", "B4", "B5", "B6", "B7"]

        # Mock image properties
        mock_image.bandNames.return_value = mock_bands
        mock_image_collection.first.return_value = mock_image
        mock_ee.ImageCollection.return_value = mock_image_collection

        # Mock reducer
        mock_reducer = Mock()
        mock_reducer.return_value = mock_image

        # Mock GEELandTrendr for FTV tests
        mock_landtrendr = Mock()
        mock_landtrendr.ftv_image.return_value = mock_image
        mock_landtrendr.lt_result = mock_image

        # Mock GEELandsatTimeSeries
        mock_timeseries = Mock()
        mock_timeseries.return_value = mock_image_collection
        mock_timeseries.get_tscollection.return_value = mock_image_collection

        # Patch the ee module and dependencies
        monkeypatch.setattr("forestvision.datasets.geelandsat.ee", mock_ee)
        monkeypatch.setattr(
            "forestvision.datasets.geelandsat.GEELandsat8._reducer", mock_reducer
        )
        monkeypatch.setattr(
            "forestvision.datasets.geelandsat.GEELandsatFTV._reducer", mock_reducer
        )
        monkeypatch.setattr(
            "forestvision.datasets.geelandsat.GEELandTrendr",
            lambda *args, **kwargs: mock_landtrendr,
        )
        monkeypatch.setattr(
            "forestvision.datasets.geelandsat.GEELandsatTimeSeries",
            lambda *args, **kwargs: mock_timeseries,
        )

        setup = {
            "l8": {
                "class": GEELandsat8,
                "args": {
                    "year": 2024,
                    "roi": tile,
                    "path": os.path.join(
                        os.path.dirname(__file__), "..", "data", "geelandsat8"
                    ),
                    "transforms": nn.Identity(),
                },
            },
            "ftv": {
                "class": GEELandsatFTV,
                "args": {
                    "year": 2024,
                    "roi": tile,
                    "path": os.path.join(
                        os.path.dirname(__file__), "..", "data", "geelandsatftv"
                    ),
                    "transforms": nn.Identity(),
                },
            },
        }
        dataset = setup[request.param]["class"]
        return dataset(**setup[request.param]["args"])

    @pytest.mark.parametrize("dataset", ["l8", "ftv"], indirect=True)
    def test_separate_files(self, dataset: CloudRasterDataset) -> None:
        assert dataset.index.count(dataset.index.bounds) == 1

    @pytest.mark.parametrize("dataset", ["l8", "ftv"], indirect=True)
    def test_getitem(self, dataset: CloudRasterDataset) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["bbox"], BoundingBox)

    @pytest.mark.parametrize("dataset", ["l8", "ftv"], indirect=True)
    def test_len(self, dataset: CloudRasterDataset) -> None:
        assert len(dataset) == 1

    @pytest.mark.parametrize("dataset", ["l8", "ftv"], indirect=True)
    def test_and(self, dataset: CloudRasterDataset) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    @pytest.mark.parametrize("dataset", ["l8", "ftv"], indirect=True)
    def test_or(self, dataset: CloudRasterDataset) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    @pytest.mark.parametrize("dataset", ["l8", "ftv"], indirect=True)
    def test_plot(self, dataset: CloudRasterDataset) -> None:
        matplotlib.use("agg")

        x = dataset[dataset.bounds]
        fig = dataset.plot(x, suptitle="Test")
        assert isinstance(fig, plt.Figure)
        plt.close()

    @pytest.mark.parametrize("dataset", ["l8"], indirect=True)
    def test_collection_property(self, dataset: CloudRasterDataset) -> None:
        # Test that the collection property returns a mock object
        collection = dataset.collection
        assert collection is not None
        # The collection should be a mock object from our fixture
        assert hasattr(collection, "first")

    @pytest.mark.parametrize("dataset", ["l8", "ftv"], indirect=True)
    def test_invalid_query(self, dataset: CloudRasterDataset) -> None:
        query = BoundingBox(0, 0, 0, 0, 0, 0)
        with pytest.raises(IndexError, match="query: .* outside of ROI .*"):
            dataset[query]


@pytest.mark.parametrize("epsg", [4326, 3857, 5070])
def test_different_crs(tile, epsg, monkeypatch) -> None:
    # Mock Earth Engine for this test
    mock_ee = Mock()
    mock_image_collection = Mock()
    mock_image = Mock()

    # Mock band names
    mock_bands = Mock()
    mock_bands.getInfo.return_value = ["B1", "B2", "B3", "B4", "B5", "B6", "B7"]

    # Mock image properties
    mock_image.bandNames.return_value = mock_bands
    mock_image_collection.first.return_value = mock_image
    mock_ee.ImageCollection.return_value = mock_image_collection

    # Mock reducer
    mock_reducer = Mock()
    mock_reducer.return_value = mock_image

    # Mock GEELandTrendr for FTV tests
    mock_landtrendr = Mock()
    mock_landtrendr.ftv_image.return_value = mock_image
    mock_landtrendr.lt_result = mock_image

    # Mock GEELandsatTimeSeries
    mock_timeseries = Mock()
    mock_timeseries.return_value = mock_image_collection
    mock_timeseries.get_tscollection.return_value = mock_image_collection

    # Patch the ee module and dependencies
    monkeypatch.setattr("forestvision.datasets.geelandsat.ee", mock_ee)
    monkeypatch.setattr(
        "forestvision.datasets.geelandsat.GEELandsat8._reducer", mock_reducer
    )
    monkeypatch.setattr(
        "forestvision.datasets.geelandsat.GEELandsatFTV._reducer", mock_reducer
    )
    monkeypatch.setattr(
        "forestvision.datasets.geelandsat.GEELandTrendr",
        lambda *args, **kwargs: mock_landtrendr,
    )
    monkeypatch.setattr(
        "forestvision.datasets.geelandsat.GEELandsatTimeSeries",
        lambda *args, **kwargs: mock_timeseries,
    )

    root = os.path.join("data", "geelandsat8")
    transforms = nn.Identity()
    crs = CRS.from_epsg(epsg)

    dataset = GEELandsat8(2024, roi=tile, path=root, transforms=transforms, crs=crs)
    assert dataset.crs.to_epsg() == epsg
