from typing import Union

import numpy as np
from rasterio.crs import CRS

from torchgeo.datasets import BoundingBox
from torchgeo.samplers import Units, tile_to_chips
from torchgeo.samplers.utils import _to_tuple


def roi_to_tiles(
    roi: BoundingBox,
    size: Union[int, tuple],
    stride: Union[int, tuple] = None,
    res: int = 1,
    units=Units.PIXELS,
) -> list:
    """Split ROI bounding box into n = rows x cols bounding boxes.

    Args:
        roi: ROI bounding box to split
        size: size of the bounding box to split
        res: resolution of the dataset
        stride: stride of the bounding box to split. If None, stride is equal to size.
        units: units of the size and stride arguments. Defaults to Units.PIXELS

    Returns:
        list: list of bounding boxes [(minx, miny, maxx, maxy), ...]
    """
    size = _to_tuple(size)
    if stride is None:
        stride = size
    stride = _to_tuple(stride)
    if units == Units.PIXELS:
        size = (size[0] * res, size[1] * res)
        stride = (stride[0] * res, stride[1] * res)

    rows, cols = tile_to_chips(roi, size, stride)

    cminy = [roi.miny + stride[0] * i for i in range(rows)]
    cmaxy = [y + size[0] for y in cminy]
    cminx = [roi.minx + stride[1] * i for i in range(cols)]
    cmaxx = [x + size[1] for x in cminx]
    cmin = np.array(np.meshgrid(cminx, cminy)).T
    cmax = np.array(np.meshgrid(cmaxx, cmaxy)).T

    return [
        x.tolist() + y.tolist()
        for x, y in zip(
            cmin.flatten().reshape((rows * cols, 2)),
            cmax.flatten().reshape((rows * cols, 2)),
        )
    ]
