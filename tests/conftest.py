import sys
import pytest
import torch
from geopandas import GeoDataFrame
from torchgeo.datasets import BoundingBox, GeoDataset
from rtree.index import Index, Property


@pytest.fixture
def tiles() -> GeoDataFrame:
    """GeoDataFrame with 6 tile footprints across Oregon."""
    return GeoDataFrame.from_features(
        {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:EPSG::5070"},
            },
            "features": [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [-1861185.0, 3046815.0],
                                [-1861185.0, 3049215.0],
                                [-1863585.0, 3049215.0],
                                [-1863585.0, 3046815.0],
                                [-1861185.0, 3046815.0],
                            ]
                        ],
                    },
                },
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [-1933185.0, 2725215.0],
                                [-1933185.0, 2727615.0],
                                [-1935585.0, 2727615.0],
                                [-1935585.0, 2725215.0],
                                [-1933185.0, 2725215.0],
                            ]
                        ],
                    },
                },
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [-2165985.0, 2766015.0],
                                [-2165985.0, 2768415.0],
                                [-2168385.0, 2768415.0],
                                [-2168385.0, 2766015.0],
                                [-2165985.0, 2766015.0],
                            ]
                        ],
                    },
                },
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [-1942785.0, 2662815.0],
                                [-1942785.0, 2665215.0],
                                [-1945185.0, 2665215.0],
                                [-1945185.0, 2662815.0],
                                [-1942785.0, 2662815.0],
                            ]
                        ],
                    },
                },
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [-1736385.0, 2504415.0],
                                [-1736385.0, 2506815.0],
                                [-1738785.0, 2506815.0],
                                [-1738785.0, 2504415.0],
                                [-1736385.0, 2504415.0],
                            ]
                        ],
                    },
                },
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [-1894785.0, 2931615.0],
                                [-1894785.0, 2934015.0],
                                [-1897185.0, 2934015.0],
                                [-1897185.0, 2931615.0],
                                [-1894785.0, 2931615.0],
                            ]
                        ],
                    },
                },
            ],
        }
    )


@pytest.fixture
def tile() -> BoundingBox:
    """A single tile footprint somewhere in Oregon.

    crs: EPSG:5070
    """
    return BoundingBox(-2108145.0, -2106225.0, 2994255.0, 2996175.0, 0, sys.maxsize)


@pytest.fixture
def roi() -> BoundingBox:
    """A small ROI somewhere in Oregon.

    crs: EPSG:5070
    """
    return BoundingBox(-2276145.0, -1587825.0, 2343855.0, 3154575.0, 0, sys.maxsize)


@pytest.fixture
def dunes() -> BoundingBox:
    """A single tile at Oregon Dunes National Recreation Area, Oregon.

    crs: EPSG:5070
    """
    return BoundingBox(
        -2219976.5070072114,
        -2216076.5070072114,
        2650010.507788832,
        2653910.507788832,
        0,
        9223372036854775807,
    )




@pytest.fixture
def sample_nodata() -> dict[str, torch.Tensor]:
    return {
        "image": torch.tensor(
            [
                [
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    [[1, 2, 3], [4, 5, 6], [7, -5, 9]],
                    [[1, 2, 3], [4, 5, 6], [7, 8, -5]],
                ]
            ],
            dtype=torch.float,
        ),
        "mask": torch.tensor([[[0, 0, 1], [0, 1, 1], [1, 1, 1]]], dtype=torch.long),
    }


@pytest.fixture
def sample_rgb() -> dict[str, torch.Tensor]:
    return {
        "image": torch.tensor(
            [
                [
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                ]
            ],
            dtype=torch.float,
        ),
        "mask": torch.tensor([[[0, 0, 1], [0, 1, 1], [1, 1, 1]]], dtype=torch.long),
    }
