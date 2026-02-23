"""Kornia-based data augmentation transforms for geospatial imagery.

This module provides GPU-accelerated, differentiable data augmentations
for land cover classification tasks. All transforms apply the same spatial
transformation to both images and masks to maintain alignment.

The recommended base augmentation pipeline (from paper):
    - Random horizontal flip (p=0.5)
    - Random rotation (90°, 180°, 270° or continuous within ±30°)
    - Random scaling (0.8–1.2)
    - Random cropping with resizing to fixed input size

Example:
    >>> from forestvision.transforms.augmentations import (
    ...     RandomHorizontalFlip,
    ...     RandomRotation,
    ...     RandomCropResize,
    ... )
    >>> transform = RandomHorizontalFlip(p=0.5)
    >>> augmented_sample = transform(sample)
"""

from typing import Any, Dict, List, Tuple, Union, Optional
import torch
import torch.nn as nn
from kornia.augmentation import (
    RandomHorizontalFlip as KorniaHFlip,
    RandomRotation as KorniaRotation,
    RandomResizedCrop as KorniaResizedCrop,
)


class BaseAugmentation:
    """Base class for Kornia-based augmentations.
    
    Handles tensor shape normalization (3D↔4D) and ensures consistent
    application to both images and masks.
    
    Args:
        interpolation_mode: Interpolation for image ('bilinear' or 'nearest')
        mask_interpolation: Interpolation for mask ('nearest' to preserve labels)
    """
    
    def __init__(
        self,
        interpolation_mode: str = "bilinear",
        mask_interpolation: str = "nearest",
    ):
        self.interpolation_mode = interpolation_mode
        self.mask_interpolation = mask_interpolation
    
    def _normalize_shape(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Normalize tensor to 4D (B, C, H, W) and return original ndim.
        
        Args:
            tensor: Input tensor of shape (C, H, W) or (B, C, H, W)
            
        Returns:
            Tuple of (normalized_tensor, original_ndim)
        """
        original_ndim = tensor.ndim
        if tensor.ndim == 3:
            # (C, H, W) -> (1, C, H, W)
            tensor = tensor.unsqueeze(0)
        return tensor, original_ndim
    
    def _restore_shape(self, tensor: torch.Tensor, original_ndim: int) -> torch.Tensor:
        """Restore tensor to original shape.
        
        Args:
            tensor: Tensor of shape (B, C, H, W)
            original_ndim: Original number of dimensions (3 or 4)
            
        Returns:
            Tensor with original shape restored
        """
        if original_ndim == 3:
            tensor = tensor.squeeze(0)
        return tensor
    
    def _apply_to_sample(
        self,
        sample: Dict[str, Any],
        aug_fn_image,
        aug_fn_mask=None,
    ) -> Dict[str, Any]:
        """Apply augmentation to sample with consistent transforms.
        
        Args:
            sample: Dictionary with 'image' and 'mask' keys
            aug_fn_image: Kornia augmentation function for images
            aug_fn_mask: Optional separate function for masks (uses same params)
            
        Returns:
            Augmented sample dictionary
        """
        # Get tensors
        image = sample["image"]
        mask = sample.get("mask")
        
        # Normalize shapes
        image, image_ndim = self._normalize_shape(image)
        if mask is not None:
            mask, mask_ndim = self._normalize_shape(mask)
            # Store original dtype for restoration
            mask_dtype = mask.dtype
            # Convert mask to float for Kornia (then back after)
            mask_float = mask.float()
        else:
            mask_float = None
        
        # Apply to image
        image_aug = aug_fn_image(image)
        sample["image"] = self._restore_shape(image_aug, image_ndim)
        
        # Apply to mask with same parameters
        if mask_float is not None:
            if aug_fn_mask is not None:
                mask_aug = aug_fn_mask(mask_float)
            else:
                mask_aug = aug_fn_image(mask_float)
            # Convert back to original dtype and restore shape
            sample["mask"] = self._restore_shape(mask_aug, mask_ndim).to(mask_dtype)
        
        # Preserve metadata (bbox, crs)
        return sample


class RandomHorizontalFlip(BaseAugmentation):
    """Random horizontal flip with probability p.
    
    Applies the same flip to both image and mask to maintain alignment.
    
    Args:
        p: Probability of applying the flip (default: 0.5)
        
    Example:
        >>> transform = RandomHorizontalFlip(p=0.5)
        >>> sample = {"image": img, "mask": mask, "bbox": bbox, "crs": crs}
        >>> aug_sample = transform(sample)
    """
    
    def __init__(self, p: float = 0.5):
        super().__init__(interpolation_mode="bilinear", mask_interpolation="nearest")
        self.p = p
        self._aug = KorniaHFlip(p=p, same_on_batch=False)
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply random horizontal flip.
        
        Args:
            sample: Dictionary with 'image' and optionally 'mask'
            
        Returns:
            Augmented sample with same keys
        """
        return self._apply_to_sample(
            sample,
            aug_fn_image=self._aug,
            aug_fn_mask=self._aug,
        )
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class RandomRotation(BaseAugmentation):
    """Random continuous rotation within specified degree range.
    
    Rotates both image and mask by the same random angle. Uses nearest-neighbor
    interpolation for masks to preserve class labels.
    
    Args:
        degrees: Maximum rotation angle in degrees (default: 30)
                 Rotation will be in range [-degrees, +degrees]
        
    Example:
        >>> transform = RandomRotation(degrees=30)  # ±30 degrees
        >>> aug_sample = transform(sample)
    """
    
    def __init__(self, degrees: float = 30.0):
        super().__init__(interpolation_mode="bilinear", mask_interpolation="nearest")
        self.degrees = degrees
        # Store config for creating augmentations dynamically
        self._degrees = (-degrees, degrees)
    
    def _create_aug(self, resample: str = "bilinear"):
        """Create Kornia rotation augmentation with specified resample mode."""
        return KorniaRotation(
            degrees=self._degrees,
            resample=resample,
            same_on_batch=False,
        )
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply random rotation.
        
        Args:
            sample: Dictionary with 'image' and optionally 'mask'
            
        Returns:
            Augmented sample with same keys
        """
        # Create augmentations dynamically
        aug_image = self._create_aug(resample="bilinear")
        aug_mask = self._create_aug(resample="nearest")
        
        # Get tensors to generate params from image
        image = sample["image"]
        image_norm, _ = self._normalize_shape(image)
        
        # Generate params from image
        params = aug_image.generate_parameters(image_norm.shape)
        
        # Apply with same params to both
        def apply_with_params(tensor, is_mask=False):
            tensor_norm, orig_ndim = self._normalize_shape(tensor)
            
            # Convert to float for Kornia if needed
            if is_mask and tensor_norm.dtype != torch.float32:
                orig_dtype = tensor_norm.dtype
                tensor_norm = tensor_norm.float()
            else:
                orig_dtype = None
            
            aug = aug_mask if is_mask else aug_image
            aug._params = params  # Set same params
            result = aug(tensor_norm)
            
            # Restore original dtype for masks
            if orig_dtype is not None:
                result = result.to(orig_dtype)
            
            return self._restore_shape(result, orig_ndim)
        
        sample["image"] = apply_with_params(sample["image"], is_mask=False)
        if "mask" in sample:
            sample["mask"] = apply_with_params(sample["mask"], is_mask=True)
        
        return sample
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(degrees={self.degrees})"


class RandomDiscreteRotation(BaseAugmentation):
    """Random rotation by fixed angles: 90°, 180°, or 270°.
    
    Useful for geospatial data where 90° rotations maintain pixel alignment
    and don't require interpolation (just transpose/flip operations).
    
    Args:
        degrees: List of rotation angles (default: [90, 180, 270])
        
    Example:
        >>> transform = RandomDiscreteRotation(degrees=[90, 180, 270])
        >>> aug_sample = transform(sample)
    """
    
    def __init__(self, degrees: List[int] = None):
        super().__init__(interpolation_mode="bilinear", mask_interpolation="nearest")
        self.degrees = degrees or [90, 180, 270]
        # Create augmentation for each possible angle
        self._augs = {
            angle: KorniaRotation(
                degrees=(angle, angle),
                resample="bilinear",
                same_on_batch=False,
            )
            for angle in self.degrees
        }
        self._augs_mask = {
            angle: KorniaRotation(
                degrees=(angle, angle),
                resample="nearest",
                same_on_batch=False,
            )
            for angle in self.degrees
        }
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply random discrete rotation.
        
        Args:
            sample: Dictionary with 'image' and optionally 'mask'
            
        Returns:
            Augmented sample with same keys
        """
        # Randomly select angle
        device = sample["image"].device
        angle_idx = torch.randint(0, len(self.degrees), (1,), device=device).item()
        angle = self.degrees[angle_idx]
        
        # Get augmentation functions
        aug_image = self._augs[angle]
        aug_mask = self._augs_mask[angle]
        
        return self._apply_to_sample(
            sample,
            aug_fn_image=aug_image,
            aug_fn_mask=aug_mask,
        )
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(degrees={self.degrees})"


class RandomScale(BaseAugmentation):
    """Random scaling/resizing within specified scale range.
    
    Scales the image and mask by a random factor, then resizes back to
    original dimensions. Uses nearest-neighbor for masks.
    
    Args:
        scale: Tuple of (min_scale, max_scale), default (0.8, 1.2)
               Scale factor relative to original size
        
    Example:
        >>> transform = RandomScale(scale=(0.8, 1.2))  # 80% to 120%
        >>> aug_sample = transform(sample)
    """
    
    def __init__(self, scale: Tuple[float, float] = (0.8, 1.2)):
        super().__init__(interpolation_mode="bilinear", mask_interpolation="nearest")
        self.scale = scale
        self._aug = None  # Will be created dynamically based on input size
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply random scaling.
        
        Args:
            sample: Dictionary with 'image' and optionally 'mask'
            
        Returns:
            Augmented sample with same keys
        """
        from kornia.geometry.transform import resize
        
        # Get image shape to determine output size
        image = sample["image"]
        if image.ndim == 3:
            _, h, w = image.shape
        else:
            _, _, h, w = image.shape
        
        # Create augmentation with target size = original size
        # RandomResizedCrop handles the scaling internally
        aug_image = KorniaResizedCrop(
            size=(h, w),
            scale=self.scale,
            resample="bilinear",
            same_on_batch=False,
        )
        
        # Normalize and apply
        image_norm, image_ndim = self._normalize_shape(image)
        mask = sample.get("mask")
        
        # Generate params from image
        params = aug_image.generate_parameters(image_norm.shape)
        
        # Apply to image
        aug_image._params = params
        image_aug = aug_image(image_norm)
        sample["image"] = self._restore_shape(image_aug, image_ndim)
        
        # Apply to mask using manual crop + resize to avoid align_corners issue
        if mask is not None:
            mask_norm, mask_ndim = self._normalize_shape(mask)
            mask_dtype = mask_norm.dtype  # Store original dtype (e.g., Long)
            mask_float = mask_norm.float()  # Convert to float for resize
            
            # Get crop boxes from params
            crop_boxes = params['src']  # [B, 4, 2] format: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            
            # Extract crop coordinates (convert from corners to x1, y1, x2, y2)
            x1 = crop_boxes[:, 0, 0].long()
            y1 = crop_boxes[:, 0, 1].long()
            x2 = crop_boxes[:, 2, 0].long()
            y2 = crop_boxes[:, 2, 1].long()
            
            # Manual crop and resize for mask using nearest with align_corners=None
            B, C, H, W = mask_float.shape
            cropped_masks = []
            for b in range(B):
                # Crop the mask (use float version)
                cropped = mask_float[b:b+1, :, y1[b]:y2[b], x1[b]:x2[b]]
                # Resize back to original size using nearest without align_corners
                resized = resize(cropped, size=(h, w), interpolation='nearest', align_corners=None)
                cropped_masks.append(resized)
            
            mask_aug = torch.cat(cropped_masks, dim=0)
            # Convert back to original dtype and restore shape
            sample["mask"] = self._restore_shape(mask_aug, mask_ndim).to(mask_dtype)
        
        return sample
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(scale={self.scale})"


class RandomCropResize(BaseAugmentation):
    """Random crop followed by resize to fixed output size.
    
    Crops a random region from the image (with scale variation), then resizes
    to the specified output size. This is a standard augmentation for training
    CNNs with fixed input dimensions.
    
    Args:
        size: Output size as (height, width) or single int for square
        scale: Range of crop scale relative to original (default: (0.8, 1.0))
        ratio: Aspect ratio range (default: (0.75, 1.33))
        
    Example:
        >>> # Crop random region, resize to 224x224
        >>> transform = RandomCropResize(size=(224, 224), scale=(0.8, 1.0))
        >>> aug_sample = transform(sample)
        
        >>> # Square output
        >>> transform = RandomCropResize(size=256)
    """
    
    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        scale: Tuple[float, float] = (0.8, 1.0),
        ratio: Tuple[float, float] = (0.75, 1.33),
    ):
        super().__init__(interpolation_mode="bilinear", mask_interpolation="nearest")
        
        # Normalize size to tuple
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
            
        self.scale = scale
        self.ratio = ratio
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply random crop and resize.
        
        Args:
            sample: Dictionary with 'image' and optionally 'mask'
            
        Returns:
            Augmented sample with resized dimensions
        """
        from kornia.geometry.transform import resize
        
        # Create augmentations
        aug_image = KorniaResizedCrop(
            size=self.size,
            scale=self.scale,
            ratio=self.ratio,
            resample="bilinear",
            same_on_batch=False,
        )
        
        # Normalize and apply
        image = sample["image"]
        image_norm, image_ndim = self._normalize_shape(image)
        mask = sample.get("mask")
        
        # Generate params from image
        params = aug_image.generate_parameters(image_norm.shape)
        
        # Apply to image
        aug_image._params = params
        image_aug = aug_image(image_norm)
        sample["image"] = self._restore_shape(image_aug, image_ndim)
        
        # Apply to mask using manual crop + resize to avoid align_corners issue
        if mask is not None:
            mask_norm, mask_ndim = self._normalize_shape(mask)
            mask_dtype = mask_norm.dtype  # Store original dtype (e.g., Long)
            mask_float = mask_norm.float()  # Convert to float for resize
            
            # Get crop boxes from params
            crop_boxes = params['src']  # [B, 4, 2] format
            
            # Extract crop coordinates
            x1 = crop_boxes[:, 0, 0].long()
            y1 = crop_boxes[:, 0, 1].long()
            x2 = crop_boxes[:, 2, 0].long()
            y2 = crop_boxes[:, 2, 1].long()
            
            # Manual crop and resize for mask using nearest with align_corners=None
            B, C, H, W = mask_float.shape
            cropped_masks = []
            for b in range(B):
                # Crop the mask (use float version)
                cropped = mask_float[b:b+1, :, y1[b]:y2[b], x1[b]:x2[b]]
                # Resize to target size using nearest without align_corners
                resized = resize(cropped, size=self.size, interpolation='nearest', align_corners=None)
                cropped_masks.append(resized)
            
            mask_aug = torch.cat(cropped_masks, dim=0)
            # Convert back to original dtype and restore shape
            sample["mask"] = self._restore_shape(mask_aug, mask_ndim).to(mask_dtype)
        
        return sample
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"size={self.size}, scale={self.scale}, ratio={self.ratio})"
        )


class ComposeAugmentations:
    """Compose multiple augmentations into a single transform.
    
    Applies a sequence of augmentations in order. Useful for building
    the complete augmentation pipeline.
    
    Args:
        transforms: List of augmentation transforms to apply
        
    Example:
        >>> pipeline = ComposeAugmentations([
        ...     RandomHorizontalFlip(p=0.5),
        ...     RandomRotation(degrees=30),
        ...     RandomCropResize(size=(224, 224)),
        ... ])
        >>> aug_sample = pipeline(sample)
    """
    
    def __init__(self, transforms: List):
        self.transforms = transforms
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply all transforms in sequence.
        
        Args:
            sample: Input sample dictionary
            
        Returns:
            Sample after all transformations
        """
        for transform in self.transforms:
            sample = transform(sample)
        return sample
    
    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "(["
        for transform in self.transforms:
            format_string += "\n"
            format_string += f"    {transform}"
        format_string += "\n])"
        return format_string


# Convenience function for the recommended base pipeline
def create_base_augmentation_pipeline(
    crop_size: Union[int, Tuple[int, int]] = 256,
    flip_p: float = 0.5,
    rotation_degrees: float = 30.0,
    scale_range: Tuple[float, float] = (0.8, 1.2),
) -> ComposeAugmentations:
    """Create the recommended base augmentation pipeline.
    
    Implements the paper's recommended base augmentations:
    - Random horizontal flip (p=0.5)
    - Random rotation (continuous ±30°)
    - Random scaling (0.8-1.2)
    - Random crop and resize to fixed size
    
    Args:
        crop_size: Output size for crop/resize (default: 256x256)
        flip_p: Probability of horizontal flip (default: 0.5)
        rotation_degrees: Max rotation angle (default: 30)
        scale_range: Min/max scale factor (default: 0.8-1.2)
        
    Returns:
        Composed augmentation pipeline
        
    Example:
        >>> pipeline = create_base_augmentation_pipeline(
        ...     crop_size=224,
        ...     flip_p=0.5,
        ...     rotation_degrees=30,
        ...     scale_range=(0.8, 1.2),
        ... )
    """
    return ComposeAugmentations([
        RandomHorizontalFlip(p=flip_p),
        RandomRotation(degrees=rotation_degrees),
        RandomScale(scale=scale_range),
        RandomCropResize(size=crop_size, scale=(0.8, 1.0)),
    ])
