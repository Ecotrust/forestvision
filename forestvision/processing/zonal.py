from abc import abstractmethod, ABC
import hashlib
import numpy
import rasterio
import pandas
import rasterio.transform
from torchgeo.datasets import RasterDataset, BoundingBox

from forestvision.datasets import (
    CloudRasterDataset,
    ForestOwnership,
    GPDFeatureCollection,
)


class ZonalStatsBase(ABC):
    """Base class for computing zonal statistics."""

    errors = []

    def __init__(
        self,
        boundaries: GPDFeatureCollection,
        raster: RasterDataset | CloudRasterDataset,
        zones_raster: RasterDataset | CloudRasterDataset = None,
        boundary_key: str = "geometry",
        data_key: str = "mask",
        nodata: int = None,
        val_range: tuple = None,
    ):
        assert raster.crs == boundaries.crs
        assert boundary_key in boundaries.data.columns
        if zones_raster:
            assert zones_raster.crs == raster.crs
            assert zones_raster.res == raster.res
        if nodata is None:
            nodata = raster.nodata or 0

        self.raster = raster
        self.zones = boundaries
        self.zones_raster = zones_raster
        self.data_key = data_key
        self.boundary_key = boundary_key
        self.nodata = nodata
        self.val_range = val_range
        hits = []
        outside = 0
        for idx in range(len(boundaries)):
            bbox = boundaries[idx]
            try:
                if bbox == raster.bounds & bbox:
                    hits.append((bbox, boundaries.data.iloc[idx]))
                else:
                    outside += 1
                    # print(f"Geometry partially outside raster bounds: {bbox}")
            except ValueError:
                # print(f"Geometry outside raster bounds: {bbox}")
                outside += 1
        if outside:
            print(
                f"{outside} geometries fully or partially outside raster bounds excluded."
            )
        self.hits = hits
        # self.outside = outside

    @abstractmethod
    def reduce_func(
        self,
        bbox: BoundingBox,
        row: object,
        data: numpy.ndarray,
        zones: numpy.ndarray = None,
    ):
        """Define function to compute raster statistics."""

    def __len__(self):
        return len(self.hits)

    def __iter__(self):
        for idx in range(len(self)):
            yield self.__getitem__(idx)

    def __getitem__(self, idx):
        zones = None
        bbox, row = self.hits[idx]

        try:
            rdata = self.raster.__getitem__(bbox)[self.data_key].squeeze().numpy()
        except IndexError as e:
            print(f"Failed to retrieve raster data for boundary geometry {row.name}")
            self.errors.append(
                {"object": self.raster.__class__.__name__, "row": row.name, "error": e}
            )
            return None

        height, width = rdata.shape
        transform = rasterio.transform.from_bounds(
            bbox.minx,
            bbox.miny,
            bbox.maxx,
            bbox.maxy,
            width=width,
            height=height,
        )

        rgeom = rasterio.features.rasterize(
            [(row.geometry, 1)],
            out_shape=rdata.shape,
            transform=transform,
            fill=0,
            all_touched=True,
            dtype="uint8",
        )

        rdata[rgeom == 0] = self.nodata

        if self.zones_raster:
            try:
                zones = (
                    self.zones_raster.__getitem__(bbox)[self.data_key].squeeze().numpy()
                )
            except IndexError as e:
                print(f"Error: {e}")
                return None
            zones[rgeom == 0] = self.nodata

        return self.reduce_func(bbox, row, rdata, zones)
