from typing import Tuple

import numpy
import torch
from torchgeo.samplers.utils import _to_tuple
from torchgeo.datasets import BoundingBox
import rasterio.transform as riotransform
from rasterio.crs import CRS
from rasterio.profiles import DefaultGTiffProfile
from rasterio.enums import Resampling
from rasterio import MemoryFile


def resize_raster(
    data: torch.Tensor,
    bbox: BoundingBox,
    size: Tuple[int, int] | int,
    crs: CRS,
    interpolation: str = "nearest",
) -> torch.Tensor:
    """
    Resize a raster tensor.

    Args:
        data: The tensor to resize.
        bbox: The bounding box of the raster.
        size: The new size of the raster.
        interpolation: The interpolation method to use. One of "nearest", "bilinear", "cubic", "cubic_spline", "lanczos", "average", "mode".

    Returns:
        The resized tensor.
    """
    resampling_options = [
        "nearest",
        "bilinear",
        "cubic",
        "cubic_spline",
        "lanczos",
        "average",
        "mode",
    ]
    size = _to_tuple(size)

    try:
        option_idx = resampling_options.index(interpolation)
    except ValueError:
        raise ValueError(f"Interpolation method must be one of {resampling_options}")
    interpolation = Resampling(option_idx)

    dtype = data.dtype
    if data.ndim < 3:
        data = data.unsqueeze(0)
    bands, height, width = data.shape
    data = data.numpy()
    transform = riotransform.from_bounds(
        bbox.minx, bbox.miny, bbox.maxx, bbox.maxy, width, height
    )
    profile = DefaultGTiffProfile(
        width=width,
        height=height,
        count=bands,
        transform=transform,
        dtype="uint8" if data.dtype == bool else data.dtype,
        crs=crs,
    )
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dst:
            dst.write(data)
            h, w = size
            resampled = dst.read(
                out_shape=(bands, h, w),
                resampling=interpolation,
            )
            return torch.tensor(resampled, dtype=dtype)
