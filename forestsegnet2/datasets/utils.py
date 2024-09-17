from pathlib import Path
from typing import List, Tuple, Union

import torch
from torch.utils.data import DataLoader
from torchgeo.samplers import GeoSampler
from torchgeo.datasets import stack_samples, GeoDataset
from kornia.enhance import Normalize
from tqdm import tqdm


class DatasetStats:
    path: Path = None
    nodata: int | None = None

    def __init__(
        self,
        dataset: GeoDataset,
        sampler: GeoSampler,
        collate_fn=stack_samples,
        channels: int = None,
        nodata: int | None = None,
        on_dims=(0, 2, 3),
        batch_size: int = 1,
        num_workers: int | None = None,
        overwrite: bool = False,
    ):
        # sampler = sampler(dataset, size=size, length=length)
        self.dataloader = DataLoader(
            dataset,
            sampler=sampler,
            collate_fn=collate_fn,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.data_key = "image" if dataset.is_image else "mask"

        if dataset.paths and isinstance(dataset.paths, str):
            self.path = Path(dataset.paths) / "stats.pth"
        else:
            raise ValueError("Dataset `paths` attribute must be of type `str`")

        self.datset_name = dataset.__class__.__name__
        self.samples = len(self.dataloader) * batch_size
        self.dim = on_dims
        self.overwrite = overwrite
        self.nodata = nodata or dataset.nodata
        self._sum = torch.zeros(channels, dtype=torch.float64)
        self._sum_sq = torch.zeros(channels, dtype=torch.float64)
        self._count = torch.zeros(channels, dtype=torch.float64)
        self._min = torch.zeros(channels, dtype=torch.float64)
        self._max = torch.zeros(channels, dtype=torch.float64)

    def compute(self):
        if self.path.exists() and not self.overwrite:
            stats = torch.load(self.path)
            print(
                f"{self.datset_name} stats already exists, collecting stats from file. Set `overwrite=True` to replace. "
            )

        else:
            for batch in tqdm(self.dataloader):
                image = batch[self.data_key].float()
                if self.nodata is not None:
                    image[image == self.nodata] = float("nan")
                self._sum += torch.nansum(image, dim=self.dim)
                self._sum_sq += torch.nansum(image**2, dim=self.dim)
                self._count += torch.sum(~torch.isnan(image), dim=self.dim)
                self._min = torch.stack([self._min, image.amin(dim=self.dim)]).amin(
                    dim=0
                )
                self._max = torch.stack([self._max, image.amax(dim=self.dim)]).amax(
                    dim=0
                )

            mean = self._sum / self._count
            meansq = self._sum_sq / self._count
            std = torch.sqrt(meansq - mean**2)

            stats = {
                "mean": mean,
                "std": std,
                "min": self._min,
                "max": self._max,
                "nodata": self.nodata,
                "samples": len(self.dataloader),
            }

            torch.save(stats, self.path)

            print(f"{self.__class__.__name__} stats saved to {self.path.parent}")

        return stats


class Denormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, y):
        x = y.new(*y.size())
        for i in range(x.shape[0]):
            x[i, :, :] = y[i, :, :] * self.std[i] + self.mean[i]
        return x


class ReplaceNodataVal:
    """Change no data value .

    Args:
        nodata (int): The no data value to change.
        new_nodata (int): The new no data value.
        on_key (str): The key of the data to change the no data value.
    """

    def __init__(self, nodata: int, new_nodata: int, on_key: str = "mask"):
        self.nodata = nodata
        self.new_nodata = new_nodata
        assert on_key in ["image", "mask"], "on_key must be either 'image' or 'mask'"
        self.on_key = on_key

    def __call__(self, sample):
        data = sample[self.on_key]
        sample[self.on_key] = torch.where(data == self.nodata, self.new_nodata, data)

        return sample


class Normalize:
    """Normalize the input data.

    Args:
        mean: The mean value to normalize the data.
        std: The standard deviation value to normalize the data.
        on_key: The key of the data to change the no data value.
    """

    def __init__(
        self,
        mean: Union[torch.Tensor, Tuple[float], List[float], float],
        std: Union[torch.Tensor, Tuple[float], List[float], float],
        on_key: str = "image",
        nodata: int = None,
    ):
        if isinstance(mean, float):
            mean = torch.tensor([mean])

        if isinstance(std, float):
            std = torch.tensor([std])

        if isinstance(mean, (tuple, list)):
            mean = torch.tensor(mean)

        if isinstance(std, (tuple, list)):
            std = torch.tensor(std)

        assert on_key in ["image", "mask"], "on_key must be either 'image' or 'mask'"

        self.mean = mean
        self.std = std
        self.on_key = on_key
        self.nodata = nodata

    def __call__(self, sample):
        data = (sample[self.on_key].float() - self.mean) / self.std
        if self.nodata is not None:
            data[data == self.nodata] = self.nodata
        sample[self.on_key] = data
        return sample

    def __repr__(self):
        repr = f"(mean={self.mean}, std={self.std})"
        return self.__class__.__name__ + repr


def minmax_scaling(data: torch.Tensor) -> torch.Tensor:
    """
    Min-Max scaling of multi-dimensional tensor of shape CxHxW.

    Assumes bands are stacked along the first dimension.
    """
    min_val = [data[i].min() for i in range(data.shape[0])]
    max_val = [data[i].max() for i in range(data.shape[0])]

    return torch.stack(
        [
            (data[i] - min_val[i]) / (max_val[i] - min_val[i])
            for i in range(data.shape[0])
        ]
    )
