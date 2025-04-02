import os
from typing import Tuple, Callable
from datetime import datetime
from pathlib import Path
import hashlib

import torch
from torch.utils.data import DataLoader
from torchgeo.samplers import GeoSampler
from torchgeo.datasets import stack_samples, GeoDataset, BoundingBox
from tqdm import tqdm

import numpy
from rasterio.windows import Window
from rasterio import MemoryFile
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles


class DatasetStats:
    path: Path = None
    nodata: int | None = None

    def __init__(
        self,
        dataset: GeoDataset,
        sampler: GeoSampler,
        path: str | Path = None,
        collate_fn: Callable = stack_samples,
        channels: int | None = None,
        nodata: int | None = None,
        on_dims: Tuple[int, ...] = (0, 2, 3),
        batch_size: int = 1,
        num_workers: int = 1,
        overwrite: bool = False,
    ) -> None:
        """Initialize a new DatasetStats instance.

        Args:
            dataset: The dataset to compute statistics for.
            sampler: The sampler to use for sampling the dataset.
            path: The path to save the statistics.
            collate_fn: Function to collate samples into batches.
            channels: Number of channels in the dataset. If None, it will be inferred
                from dataset.bands.
            nodata: Value to ignore when computing statistics.
            on_dims: Dimensions to reduce when computing statistics.
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            overwrite: If True, recompute statistics even if they already exist.
        """
        self.dataloader = DataLoader(
            dataset,
            sampler=sampler,
            collate_fn=collate_fn,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.data_key = "image" if dataset.is_image else "mask"
        if path and isinstance(dataset.paths, (str, Path)):
            self.path = path
        elif dataset.paths and isinstance(dataset.paths, str):
            self.path = os.path.join(dataset.paths, "stats.pt")
        else:
            self.path = None

        if channels is None and hasattr(dataset, "bands"):
            channels = len(dataset.bands)
        elif channels is None:
            raise ValueError("No channel information available.")

        self.datset_name = dataset.__class__.__name__
        self.samples = len(self.dataloader) * batch_size
        self.dim = on_dims
        self.overwrite = overwrite
        if nodata is None and hasattr(dataset, "nodata"):
            self.nodata = dataset.nodata
        else:
            self.nodata = nodata
        self._sum = torch.zeros(channels, dtype=torch.float32)
        self._sum_sq = torch.zeros(channels, dtype=torch.float32)
        self._count = torch.zeros(channels, dtype=torch.float32)
        self._min = torch.full((channels,), float("inf"), dtype=torch.float32)
        self._max = torch.zeros(channels, dtype=torch.float32)

    def compute(self) -> dict[str, torch.Tensor]:
        none_or_notexists = Path(self.path).exists() if self.path else False
        ndpixels = 0
        if any([none_or_notexists == False, self.overwrite]):
            for batch in tqdm(self.dataloader):
                image = batch[self.data_key].float()
                ndmask = image == self.nodata
                image[ndmask] = float("nan")
                self._sum += torch.nansum(image, dim=self.dim)
                self._sum_sq += torch.nansum(image**2, dim=self.dim)
                self._count += torch.sum(~torch.isnan(image), dim=self.dim)
                image[ndmask] = float("inf")
                self._min = torch.stack([self._min, image.amin(dim=self.dim)]).amin(
                    dim=0
                )
                image[ndmask] = float("-inf")
                self._max = torch.stack([self._max, image.amax(dim=self.dim)]).amax(
                    dim=0
                )
                if self.nodata is not None:
                    ndpixels += ndmask.sum().item()

            mean = self._sum / self._count
            meansq = self._sum_sq / self._count
            std = torch.sqrt(meansq - mean**2)

            stats = {
                "mean": mean,
                "std": std,
                "min": self._min,
                "max": self._max,
                "nodata": self.nodata,
                "nodata_pixels": f"{ndpixels} ({(100*ndpixels/self._count.sum()).item():.2f}%)",
                "sample_size": self.samples,
            }

            if self.path:
                torch.save(stats, self.path)
                print(
                    f"{self.__class__.__name__} stats saved to {Path(self.path).parent}"
                )

        elif Path(self.path).exists():
            stats = torch.load(self.path)
            print(
                f"File exists, loading stats from: {self.path}. Set `overwrite=True` to recompute. "
            )
        else:
            raise ValueError(f"Invalid path: {self.path}")

        return stats


def minmax_scaling(data: torch.Tensor, nodata: float) -> torch.Tensor:
    """
    Min-Max scaling of multi-dimensional tensor with shape CxHxW.

    Assumes bands are stacked along the first dimension.
    """
    dim = (1, 2)
    if len(data.shape) == 2:
        data = data.unsqueeze(0)
    if len(data.shape) > 3:
        raise ValueError("Input tensor must have shape CxHxW")
    mask = data == nodata
    data[data == nodata] = float("inf")
    min_val = data.amin(dim=dim)
    data[data == float("inf")] = float("-inf")
    max_val = data.amax(dim=dim)
    data[mask] = nodata

    scaled = (data - min_val.reshape(-1, 1, 1)) / (max_val - min_val).reshape(
        -1, 1, 1
    )
    scaled[mask] = nodata
    return scaled

def save_cog(
    data: numpy.ndarray,
    profile: dict,
    path: str,
    overwrite: bool = False,
    window: Window = None,
) -> None:
    cog_profile = cog_profiles.get("deflate")
    if not os.path.exists(path) or overwrite:
        with MemoryFile() as memfile:
            with memfile.open(**profile) as dst:
                dst.write(data)
                if window is not None:
                    w_trf = dst.window_transform(window)
                    w_profile = dst.profile.copy()
                    w_profile.update(
                        width=window.width,
                        height=window.height,
                        transform=w_trf,
                    )
                    w_data = dst.read(window=window)
                    with MemoryFile() as memfile:
                        with memfile.open(**w_profile) as w_dst:
                            w_dst.write(w_data)
                            cog_translate(
                                w_dst, path, cog_profile, in_memory=True, quiet=False
                            )
                            return
                else:
                    cog_translate(dst, path, cog_profile, in_memory=True, quiet=False)
                    return
    else:
        print(f"File {path} already exists")


def split_bbox(dim: int, bbox: Tuple[float, float, float, float]) -> list:
    """Split bounding box into dim x dim bounding boxes.

    Args:
        dim: int
            Number of splits per bbox with size (xmax-xmin, ymax-ymin). The number
            of splits (n) returned is dim x dim.
        bbox: list-like
            The bounding box to split, with format [xmin, ymin, xmax, ymax].

    Returns
        list of bounding boxes
    """
    xmin, ymin, xmax, ymax = bbox

    w = (xmax - xmin) / dim
    h = (ymax - ymin) / dim

    # For testing
    # cols = ['xmin', *[f'xmin + w*{dim + 1}' for dim in range(dim - 1)], 'xmax']
    # rows = ['ymin', *[f'ymin + l*{dim + 1}' for dim in range(dim - 1)], 'ymax']

    cols = [xmin, *[xmin + w * (dim + 1) for dim in range(dim - 1)], xmax]
    rows = [ymin, *[ymin + h * (dim + 1) for dim in range(dim - 1)], ymax]

    coords = numpy.array(numpy.meshgrid(cols, rows)).T

    bbox_splitted = []
    for i in range(dim):
        bbox_splitted.append(
            [
                numpy.array([coords[i][j], coords[i + 1][k]]).flatten()
                for j, k in zip(range(dim), range(1, dim + 1))
            ]
        )

    return [x for sbl in bbox_splitted for x in sbl]


def hash_bbox(bbox: BoundingBox) -> str:
    return hashlib.md5(
        f"({bbox.minx}, {bbox.miny}, {bbox.maxx}, {bbox.maxy})".encode()
    ).hexdigest()[:10]


def valid_date(date: str) -> Tuple[str, str]:
    """Parse a date string and return the year, month, and day."""
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise ValueError("Date format should be YYYY-MM-DD")
    return date
