from typing import List, Tuple, Union, Any

import numpy
import torch
from torchvision.transforms import functional as tvF
from torchgeo.samplers.utils import _to_tuple
from torchgeo.datasets import GeoDataset

from .utils import resize_raster


class ReplaceNodataVal:
    """Change no data value .

    Args:
        nodata (int): The nodata value to change.
        new_nodata (int): The new no data value.
        on_key (str): The key of the data to change the no data value.
    """

    def __init__(self, nodata: int, new_nodata: int, on_key: str = "mask"):
        assert on_key in ["image", "mask"], "on_key must be either 'image' or 'mask'"
        self.nodata = nodata
        self.new_nodata = new_nodata
        self.on_key = on_key

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        data = sample[self.on_key]
        sample[self.on_key] = torch.where(data == self.nodata, self.new_nodata, data)

        return sample


class Normalize:
    """Normalize mask or image data on an sample dict.

    Args:
        mean: The mean value to normalize the data.
        std: The standard deviation value to normalize the data.
        on_key: The key of the data to change the no data value.
        nodata: If provided nodata values won't not be normalized.
    """

    def __init__(
        self,
        mean: Union[torch.Tensor, Tuple[float], List[float], float],
        std: Union[torch.Tensor, Tuple[float], List[float], float],
        on_key: str = "image",
        nodata: int = None,
    ):
        assert on_key in ["image", "mask"], "on_key must be either 'image' or 'mask'"

        if isinstance(mean, float):
            mean = torch.tensor([mean])

        if isinstance(std, float):
            std = torch.tensor([std])

        if isinstance(mean, (tuple, list)):
            mean = torch.tensor(mean)

        if isinstance(std, (tuple, list)):
            std = torch.tensor(std)

        self.mean = mean
        self.std = std
        self.on_key = on_key
        self.nodata = nodata

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        data = sample[self.on_key].float()
        nodata_mask = data == self.nodata
        data = tvF.normalize(data, self.mean, self.std)
        if self.nodata is not None:
            data[nodata_mask] = self.nodata
        sample[self.on_key] = data
        return sample

    def __repr__(self):
        repr = f"(mean={self.mean}, std={self.std})"
        return self.__class__.__name__ + repr


class Denormalize:
    """Denormalize a tensor by applying the inverse of normalization."""

    def __init__(
        self,
        mean: Union[torch.Tensor, Tuple[float], List[float], float],
        std: Union[torch.Tensor, Tuple[float], List[float], float],
        nodata: int = None,
    ):
        """Initialize a new Denormalize instance.

        Args:
            mean: The mean value used in normalization.
            std: The standard deviation value used in normalization.
            nodata: If provided, nodata values won't be denormalized.
        """
        if isinstance(mean, float):
            mean = torch.tensor([mean])

        if isinstance(std, float):
            std = torch.tensor([std])

        if isinstance(mean, (tuple, list)):
            mean = torch.tensor(mean)

        if isinstance(std, (tuple, list)):
            std = torch.tensor(std)

        self.mean = mean
        self.std = std
        self.nodata = nodata

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        if self.mean.ndim == 1:
            mean = self.mean.view(-1, 1, 1)
        if self.std.ndim == 1:
            std = self.std.view(-1, 1, 1)
        nodata_mask = tensor == self.nodata
        tensor = tensor * std + mean
        if self.nodata is not None:
            tensor[nodata_mask] = self.nodata

        return tensor


class MaskFromRaster:
    """Create a boolean mask from raster.

    This class creates a boolean mask from a raster classification by selecting
    pixels with a specific class value. Optionally applies morphological filtering
    and can invert the mask.
    """

    # we want to avoid importing cv2 unless necessary
    cv = __import__("cv2")

    def __init__(
        self,
        from_class: int = 0,
        kernel_size: int = 3,
        apply_filter: bool = True,
        invert: bool = False,
    ):
        """Initialize a new MaskFromRaster instance.

        Args:
            from_class: The class value to select for the mask.
            kernel_size: Size of the kernel for morphological operations.
            apply_filter: If True, apply morphological opening to the mask.
            invert: If True, invert the mask (select everything except the class).
        """
        self.from_class = from_class
        self.apply_filter = apply_filter
        self.kernel = numpy.ones(_to_tuple(kernel_size), numpy.uint8)
        self.invert = invert

    def _filter(self, mask: torch.Tensor) -> torch.Tensor:
        filtered = self.cv.morphologyEx(
            numpy.asarray(tvF.to_pil_image(mask.float())),
            self.cv.MORPH_OPEN,
            self.kernel,
        )
        return torch.Tensor(filtered / 255).bool().unsqueeze(0)

    def get_mask(self, sample: dict[str, Any]) -> torch.Tensor:
        return sample["mask"] == self.from_class

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        bool_mask = self.get_mask(sample)

        if self.invert:
            bool_mask = ~bool_mask

        if self.apply_filter:
            bool_mask = self._filter(bool_mask)

        sample["mask"] = bool_mask

        return sample


class ApplyMasks:
    """Apply masks from one or more datasets to the sample data.

    This transform applies boolean masks from one or more GeoDatasets to the sample data.
    Pixels where the mask is False will be set to the nodata value.
    """

    def __init__(
        self,
        mask_dataset: GeoDataset | List[GeoDataset],
        nodata: float = 0,
        data_key: str = "mask",
    ):
        """Initialize a new ApplyMasks instance.

        Args:
            mask_dataset: One or more GeoDatasets containing boolean masks.
            nodata: Value to set for masked pixels.
            data_key: Key in the sample dictionary to apply the mask to.
        """
        if isinstance(mask_dataset, GeoDataset):
            mask_dataset = [mask_dataset]
        self.mask_dataset = mask_dataset
        self.nodata = nodata
        self.data_key = data_key

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        bbox = sample["bbox"]
        crs = sample["crs"]
        # if isinstance(self.mask_dataset, list):
        data = sample[self.data_key]
        mask = torch.ones_like(data, dtype=torch.bool)
        for dataset in self.mask_dataset:
            mask1 = dataset.__getitem__(bbox)["mask"]
            if mask1.dtype != torch.bool:
                raise ValueError("Mask must be a boolean tensor.")
            if data.shape != mask1.shape:
                print(f"Mask shape mismatch. Attempting to resize.")
                mask1 = resize_raster(
                    mask1, bbox, tuple([data.shape[-2], data.shape[-1]]), crs
                )
            mask = mask & mask1
        data[mask == False] = self.nodata
        sample[self.data_key] = data

        return sample


class ResizeRaster:
    """Resize raster data in a sample.

    This transform resizes raster data (image or mask) in a sample to a specified size
    using the specified interpolation method.
    """

    def __init__(
        self,
        size: int | Tuple[int, int],
        interpolation: str = "nearest",
    ):
        """Initialize a new ResizeRaster instance.

        Args:
            size: Target size, either a single integer for square output or (height, width).
            interpolation: Interpolation method to use. One of "nearest", "bilinear",
                "cubic", "cubic_spline", "lanczos", "average", or "mode".
        """
        self.interpolation = interpolation
        self.size = size

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        bbox = sample["bbox"]
        crs = sample["crs"]
        for key in ["image", "mask"]:
            if key not in sample:
                continue
            sample[key] = resize_raster(
                sample[key], bbox, self.size, crs, self.interpolation
            )

        return sample
