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

        # Store original shape for restoration
        original_shape = data.shape

        # Create nodata mask before any shape modifications
        nodata_mask = data == self.nodata if self.nodata is not None else None

        # Ensure data is 4D (B, C, H, W) for tvF.normalize
        if data.ndim == 2:
            # (H, W) -> (1, 1, H, W)
            data = data.unsqueeze(0).unsqueeze(0)
        elif data.ndim == 3:
            # Distinguish between (B, H, W) for masks and (C, H, W) for images
            if self.on_key == "mask" and len(self.mean) == 1:
                # (B, H, W) -> (B, 1, H, W)
                data = data.unsqueeze(1)
            else:
                # (C, H, W) -> (1, C, H, W)
                data = data.unsqueeze(0)
        # else: data.ndim == 4, already in correct format

        # Apply normalization
        data = tvF.normalize(data, self.mean, self.std)

        # Restore original shape
        data = data.view(original_shape)

        # Apply nodata mask after all shape transformations
        if self.nodata is not None and nodata_mask is not None:
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
    try:
        import cv2 as cv
    except ImportError:
        cv = None

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
        if self.cv is None:
            raise ImportError(
                "OpenCV (cv2) is required for morphological filtering. "
                "Please install it with 'pip install opencv-python'."
            )
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


class RemapFortypba:
    """Remap forest type codes using the GNN to ODF mapping.

    This transform applies the same remapping logic as GNNForestAttr.__getitem__()
    but ensures it happens consistently in the transform pipeline regardless of
    dataset composition.

    Args:
        remap_dict (dict): Dictionary mapping GNN codes to ODF codes
        on_key (str): The key of the data to remap (should be "mask")
    """

    def __init__(self, remap_dict: dict, on_key: str = "mask"):
        assert on_key in ["image", "mask"], "on_key must be either 'image' or 'mask'"
        self.remap_dict = remap_dict
        self.on_key = on_key

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        data = sample[self.on_key]

        # Apply remapping to each unique value in the tensor
        unique_vals = torch.unique(data)
        for val in unique_vals:
            val_int = int(val.item())
            if val_int in self.remap_dict:
                data[data == val] = self.remap_dict[val_int]

        sample[self.on_key] = data
        return sample

    def __repr__(self):
        return f"{self.__class__.__name__}(remap_dict={self.remap_dict})"


class AppendNDVI:
    """Append NDVI band to an image.

    This class wraps torchgeo.transforms.AppendNDVI to handle dictionary samples.

    Args:
        index_nir (int): Index of the NIR band.
        index_red (int): Index of the red band.
    """

    def __init__(self, index_nir: int, index_red: int):
        from torchgeo.transforms import AppendNDVI as TGAppendNDVI

        self.index_nir = index_nir
        self.index_red = index_red
        self.transform = TGAppendNDVI(index_nir=index_nir, index_red=index_red)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        data = sample["image"]
        original_ndim = data.ndim

        # Ensure data is 4D (B, C, H, W) for Kornia-based transforms
        if data.ndim == 3:
            # (C, H, W) -> (1, C, H, W)
            data = data.unsqueeze(0)

        # Apply NDVI append
        data = self.transform(data)

        # Restore original dimensionality if we added a batch dimension
        if original_ndim == 3:
            data = data.squeeze(0)

        sample["image"] = data
        return sample

    def __repr__(self):
        return f"{self.__class__.__name__}(index_nir={self.index_nir}, index_red={self.index_red})"


class AppendNBR:
    """Append NBR band to an image.

    This class wraps torchgeo.transforms.AppendNBR to handle dictionary samples.

    Args:
        index_nir (int): Index of the NIR band.
        index_swir (int): Index of the SWIR band (typically SWIR2).
    """

    def __init__(self, index_nir: int, index_swir: int):
        from torchgeo.transforms import AppendNBR as TGAppendNBR

        self.index_nir = index_nir
        self.index_swir = index_swir
        self.transform = TGAppendNBR(index_nir=index_nir, index_swir=index_swir)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        data = sample["image"]
        original_ndim = data.ndim

        # Ensure data is 4D (B, C, H, W) for Kornia-based transforms
        if data.ndim == 3:
            # (C, H, W) -> (1, C, H, W)
            data = data.unsqueeze(0)

        # Apply NBR append
        data = self.transform(data)

        # Restore original dimensionality if we added a batch dimension
        if original_ndim == 3:
            data = data.squeeze(0)

        sample["image"] = data
        return sample

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"index_nir={self.index_nir}, index_swir={self.index_swir})"
        )


class AppendEVI:
    """Append EVI band to an image.

    The EVI (Enhanced Vegetation Index) is an atmospherically-corrected index that
    reduces soil and atmospheric noise. It performs better than NDVI in dense forests.

    EVI = 2.5 * (NIR - Red) / (NIR + 6 * Red - 7.5 * Blue + 1)

    Args:
        index_nir (int): Index of the NIR band.
        index_red (int): Index of the red band.
        index_blue (int): Index of the blue band.
    """

    def __init__(self, index_nir: int, index_red: int, index_blue: int):
        self.index_nir = index_nir
        self.index_red = index_red
        self.index_blue = index_blue

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        data = sample["image"]
        original_ndim = data.ndim

        # Ensure data is 4D (B, C, H, W)
        if data.ndim == 3:
            # (C, H, W) -> (1, C, H, W)
            data = data.unsqueeze(0)

        nir = data[:, self.index_nir, :, :].float()
        red = data[:, self.index_red, :, :].float()
        blue = data[:, self.index_blue, :, :].float()

        # EVI = 2.5 * (NIR - Red) / (NIR + 6 * Red - 7.5 * Blue + 1)
        numerator = 2.5 * (nir - red)
        denominator = nir + 6.0 * red - 7.5 * blue + 1.0
        evi = numerator / (denominator + 1e-8)

        # Clean up any potential NaNs or Inf
        evi = torch.nan_to_num(evi, nan=0.0, posinf=1.0, neginf=-1.0).unsqueeze(1)

        # Append EVI band
        data = torch.cat([data, evi], dim=1)

        # Restore original dimensionality if we added a batch dimension
        if original_ndim == 3:
            data = data.squeeze(0)

        sample["image"] = data
        return sample

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"index_nir={self.index_nir}, index_red={self.index_red}, "
            f"index_blue={self.index_blue})"
        )


class AppendSAVI:
    """Append SAVI band to an image.

    The SAVI (Soil-Adjusted Vegetation Index) accounts for soil brightness under
    sparse vegetation.

    SAVI = (NIR - Red) * (1 + L) / (NIR + Red + L)

    Args:
        index_nir (int): Index of the NIR band.
        index_red (int): Index of the red band.
        L (float): Soil brightness correction factor (default: 0.5).
    """

    def __init__(self, index_nir: int, index_red: int, L: float = 0.5):
        self.index_nir = index_nir
        self.index_red = index_red
        self.L = L

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        data = sample["image"]
        original_ndim = data.ndim

        if data.ndim == 3:
            data = data.unsqueeze(0)

        nir = data[:, self.index_nir, :, :].float()
        red = data[:, self.index_red, :, :].float()

        # SAVI = (NIR - Red) * (1 + L) / (NIR + Red + L)
        numerator = (nir - red) * (1.0 + self.L)
        denominator = nir + red + self.L
        savi = numerator / (denominator + 1e-8)

        # Clean up any potential NaNs or Inf
        savi = torch.nan_to_num(savi, nan=0.0, posinf=1.0, neginf=-1.0).unsqueeze(1)

        data = torch.cat([data, savi], dim=1)

        if original_ndim == 3:
            data = data.squeeze(0)

        sample["image"] = data
        return sample

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"index_nir={self.index_nir}, index_red={self.index_red}, L={self.L})"
        )


class AppendMSAVI:
    """Append MSAVI band to an image.

    The MSAVI (Modified Soil-Adjusted Vegetation Index) reduces soil background
    effects and is effective for areas with visible bare soil.

    MSAVI = (2 * NIR + 1 - sqrt((2 * NIR + 1)^2 - 8 * (NIR - Red))) / 2

    Args:
        index_nir (int): Index of the NIR band.
        index_red (int): Index of the red band.
    """

    def __init__(self, index_nir: int, index_red: int):
        self.index_nir = index_nir
        self.index_red = index_red

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        data = sample["image"]
        original_ndim = data.ndim

        if data.ndim == 3:
            data = data.unsqueeze(0)

        nir = data[:, self.index_nir, :, :].float()
        red = data[:, self.index_red, :, :].float()

        # MSAVI = (2 * NIR + 1 - sqrt((2 * NIR + 1)^2 - 8 * (NIR - Red))) / 2
        term1 = 2.0 * nir + 1.0
        term2 = 8.0 * (nir - red)
        # Ensure the value under sqrt is non-negative
        msavi = (term1 - torch.sqrt(torch.clamp(term1**2 - term2, min=0))) / 2.0

        # Clean up any potential NaNs or Inf
        msavi = torch.nan_to_num(msavi, nan=0.0, posinf=1.0, neginf=-1.0).unsqueeze(1)

        data = torch.cat([data, msavi], dim=1)

        if original_ndim == 3:
            data = data.squeeze(0)

        sample["image"] = data
        return sample

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"index_nir={self.index_nir}, index_red={self.index_red})"
        )


class AppendNIRv:
    """Append NIRv band to an image.

    The NIRv (Near-Infrared Reflectance of Vegetation) is calculated by multiplying
    the total scene near-infrared reflectance by the NDVI. It isolates the vegetated
    signal and reduces noise.

    NIRv = NIR * (NIR - Red) / (NIR + Red)

    Args:
        index_nir (int): Index of the NIR band.
        index_red (int): Index of the red band.
    """

    def __init__(self, index_nir: int, index_red: int):
        self.index_nir = index_nir
        self.index_red = index_red

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        data = sample["image"]
        original_ndim = data.ndim

        if data.ndim == 3:
            data = data.unsqueeze(0)

        nir = data[:, self.index_nir, :, :].float()
        red = data[:, self.index_red, :, :].float()

        # NIRv = NIR * (NIR - Red) / (NIR + Red)
        numerator = nir - red
        denominator = nir + red
        # Add epsilon to avoid division by zero
        ndvi = numerator / (denominator + 1e-8)
        nirv = nir * ndvi

        # Clean up any potential NaNs or Inf
        nirv = torch.nan_to_num(nirv, nan=0.0, posinf=1.0, neginf=-1.0).unsqueeze(1)

        data = torch.cat([data, nirv], dim=1)

        if original_ndim == 3:
            data = data.squeeze(0)

        sample["image"] = data
        return sample

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"index_nir={self.index_nir}, index_red={self.index_red})"
        )


class SelectBands:
    """Select and/or reorder bands in an image.

    Args:
        indices (list[int]): List of band indices to select and/or reorder.
    """

    def __init__(self, indices: list[int]):
        self.indices = indices

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        data = sample["image"]
        original_ndim = data.ndim

        if data.ndim == 3:
            # (C, H, W) -> (1, C, H, W)
            data = data.unsqueeze(0)

        num_bands = data.shape[1]
        for idx in self.indices:
            if idx < 0 or idx >= num_bands:
                raise IndexError(
                    f"Band index {idx} out of range (num_bands={num_bands})"
                )

        # Select and reorder bands
        data = data[:, self.indices, :, :]

        if original_ndim == 3:
            data = data.squeeze(0)

        sample["image"] = data
        return sample

    def __repr__(self):
        return f"{self.__class__.__name__}(indices={self.indices})"


class MinMaxScaler:
    """Scale mask or image data to [0, 1] range using min-max scaling.

    Args:
        min: The minimum value for scaling.
        max: The maximum value for scaling.
        on_key: The key of the data to scale.
        nodata: If provided, nodata values won't be scaled.
    """

    def __init__(
        self,
        min: Union[torch.Tensor, Tuple[float], List[float], float],
        max: Union[torch.Tensor, Tuple[float], List[float], float],
        on_key: str = "image",
        nodata: int = None,
    ):
        assert on_key in ["image", "mask"], "on_key must be either 'image' or 'mask'"

        if isinstance(min, float):
            min = torch.tensor([min])

        if isinstance(max, float):
            max = torch.tensor([max])

        if isinstance(min, (tuple, list)):
            min = torch.tensor(min)

        if isinstance(max, (tuple, list)):
            max = torch.tensor(max)

        self.min = min
        self.max = max
        self.on_key = on_key
        self.nodata = nodata

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        data = sample[self.on_key].float()
        nodata_mask = data == self.nodata if self.nodata is not None else None

        # Store original shape for later restoration
        original_shape = data.shape
        original_ndim = data.ndim

        # Handle different tensor shapes for masks vs images
        if self.on_key == "mask":
            # For masks, we need to preserve the batch dimension
            # Masks typically come as (B, H, W) or (H, W)
            if data.ndim == 2:
                # Add batch and channel dimensions: (H, W) -> (1, 1, H, W)
                data = data.unsqueeze(0).unsqueeze(0)
                if nodata_mask is not None:
                    nodata_mask = nodata_mask.unsqueeze(0).unsqueeze(0)
            elif data.ndim == 3:
                # Add channel dimension: (B, H, W) -> (B, 1, H, W)
                data = data.unsqueeze(1)
                if nodata_mask is not None:
                    nodata_mask = nodata_mask.unsqueeze(1)
        else:
            # For images, add batch dimension if needed
            if data.ndim == 3:
                # Add batch dimension: (C, H, W) -> (1, C, H, W)
                data = data.unsqueeze(0)
                if nodata_mask is not None:
                    nodata_mask = nodata_mask.unsqueeze(0)

        # Apply min-max scaling
        # Reshape min and max to match data dimensions
        if self.min.ndim == 1:
            min_val = self.min.view(-1, 1, 1)
        else:
            min_val = self.min

        if self.max.ndim == 1:
            max_val = self.max.view(-1, 1, 1)
        else:
            max_val = self.max

        # Avoid division by zero
        range_val = max_val - min_val
        range_val[range_val == 0] = 1.0  # Prevent division by zero

        data = (data - min_val) / range_val

        # Remove extra dimensions we added
        if self.on_key == "mask":
            # Remove channel dimension: (B, 1, H, W) -> (B, H, W)
            data = data.squeeze(1)
            if nodata_mask is not None:
                nodata_mask = nodata_mask.squeeze(1)
        else:
            # Remove batch dimension if we added it: (1, C, H, W) -> (C, H, W)
            if original_ndim == 3:
                data = data.squeeze(0)
                if nodata_mask is not None:
                    nodata_mask = nodata_mask.squeeze(0)

        # Apply nodata mask after all shape transformations
        if self.nodata is not None and nodata_mask is not None:
            data[nodata_mask] = self.nodata

        sample[self.on_key] = data
        return sample

    def __repr__(self):
        repr = f"(min={self.min}, max={self.max})"
        return self.__class__.__name__ + repr


class InverseMinMaxScaler:
    """Inverse min-max scaling to revert values back to original range."""

    def __init__(
        self,
        min: Union[torch.Tensor, Tuple[float], List[float], float],
        max: Union[torch.Tensor, Tuple[float], List[float], float],
        nodata: int = None,
    ):
        """Initialize a new InverseMinMaxScaler instance.

        Args:
            min: The minimum value used in scaling.
            max: The maximum value used in scaling.
            nodata: If provided, nodata values won't be inverse scaled.
        """
        if isinstance(min, float):
            min = torch.tensor([min])

        if isinstance(max, float):
            max = torch.tensor([max])

        if isinstance(min, (tuple, list)):
            min = torch.tensor(min)

        if isinstance(max, (tuple, list)):
            max = torch.tensor(max)

        self.min = min
        self.max = max
        self.nodata = nodata

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be inverse scaled.
        Returns:
            Tensor: Inverse scaled image.
        """
        if self.min.ndim == 1:
            min_val = self.min.view(-1, 1, 1)
        else:
            min_val = self.min

        if self.max.ndim == 1:
            max_val = self.max.view(-1, 1, 1)
        else:
            max_val = self.max

        nodata_mask = tensor == self.nodata
        range_val = max_val - min_val
        tensor = tensor * range_val + min_val
        if self.nodata is not None:
            tensor[nodata_mask] = self.nodata

        return tensor
