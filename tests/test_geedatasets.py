import unittest
from unittest.mock import MagicMock
from functools import partial

import ee
import geopandas
import torch
from torch.utils.data import DataLoader
from torchgeo.datasets import BoundingBox, stack_samples
from torchgeo.samplers import PreChippedGeoSampler, RandomGeoSampler

from forestsegnet2.datasets import GEESentinel2, GEELandsat8

ee.Initialize()
tiles = geopandas.read_file("data/tiles.geojson")
mint = 0
maxt = 9.223372036854776e18
minx, miny, maxx, maxy = tiles.geometry[0].bounds
xsize = maxx - minx
ysize = maxy - miny
tile_bbox = BoundingBox(minx, maxx, miny, maxy, mint, maxt)
minx, miny, maxx, maxy = tiles.total_bounds
region_bbox = BoundingBox(minx, maxx, miny, maxy, mint, maxt)
year = 2020


# test with transforms argument
# test with download argument
# test with cache argument
# test bands
# test image value range
# test with path argument and local files


class TestGEERasterDatasetSetup1(unittest.TestCase):

    def setUp(self):
        print("Testing GEESentinel2 class with geopandas.GeoDataFrame tile collection")
        self.geedataset = GEESentinel2(year=year, plevel="sr", tiles=tiles)

    def test_init(self):
        self.assertEqual(self.geedataset._crs.to_epsg(), 5070)
        self.assertEqual(self.geedataset.date_start, f"{year}-04-01")
        self.assertEqual(self.geedataset.date_end, f"{year}-09-30")

    def test_getitem(self):
        sample = self.geedataset.__getitem__(tile_bbox)
        self.assertIsInstance(sample["image"], torch.Tensor)
        self.assertIn("transform", sample)
        self.assertEqual(
            tuple(sample["image"].shape),
            (
                len(self.geedataset.bands),
                xsize / self.geedataset.res,
                ysize / self.geedataset.res,
            ),
        )


class TestGEERasterDatasetSetup2(TestGEERasterDatasetSetup1):

    def setUp(self):
        self.geedataset = GEESentinel2(year=year, plevel="sr", roi=region_bbox)


class TestGEERasterDatasetSetup3(TestGEERasterDatasetSetup1):

    def setUp(self):
        self.geedataset = GEELandsat8(year=year, tiles=tiles)


class TestAndOrGEERasterDatasetSetup1(unittest.TestCase):
    res = 10

    def setUp(self):
        # For IntersectionDataset (AND) the tile shape must be the same
        # So landsar scale = 10
        self.geedataset1 = GEELandsat8(year=year, tiles=tiles, res=self.res)
        # For UnionDataset (OR) the number of bands from both datasets must be the same
        # So we select only 6 bands from Sentinel-2
        self.geedataset2 = GEESentinel2(
            year=year,
            plevel="sr",
            tiles=tiles,
        )
        self.geedataset2.bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7"]
        self.sampler = PreChippedGeoSampler

    def test_union(self):
        union = self.geedataset1 | self.geedataset2
        sampler = self.sampler(union)
        union_dataloader = DataLoader(union, sampler=sampler, collate_fn=stack_samples)
        sample = next(iter(union_dataloader))
        if not self.geedataset2.roi:
            self.assertEqual(len(union.index), len(tiles) * 2)
        self.assertIsInstance(sample["image"], torch.Tensor)
        self.assertEqual(
            tuple(sample["image"].shape),
            (
                1,
                len(self.geedataset1.bands),
                xsize / self.res,
                ysize / self.res,
            ),
        )

    def test_intersection(self):
        intersection = self.geedataset1 & self.geedataset2
        sampler = self.sampler(intersection)
        s2_bands = len(self.geedataset2.bands)
        l8_bands = len(self.geedataset1.bands)
        intersection_dataloader = DataLoader(
            intersection, sampler=sampler, collate_fn=stack_samples
        )
        sample = next(iter(intersection_dataloader))
        if not self.geedataset2.roi:
            self.assertEqual(len(intersection.index), len(tiles))
        self.assertIsInstance(sample["image"], torch.Tensor)
        self.assertEqual(
            tuple(sample["image"].shape),
            (
                1,
                s2_bands + l8_bands,
                xsize / self.res,
                ysize / self.res,
            ),
        )


class TestAndOrGEERasterDatasetSetup2(TestAndOrGEERasterDatasetSetup1):

    def setUp(self):
        self.geedataset1 = GEELandsat8(year=year, roi=region_bbox, res=10)
        self.geedataset2 = GEESentinel2(year=year, plevel="sr", roi=region_bbox)
        self.geedataset2.bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7"]
        self.sampler = partial(RandomGeoSampler, size=xsize / self.res, length=1)


if __name__ == "__main__":
    unittest.main()
