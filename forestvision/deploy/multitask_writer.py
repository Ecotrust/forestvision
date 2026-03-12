"""Multi-task prediction writer for saving multi-task model outputs to GeoTIFF."""

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
from rasterio.profiles import DefaultGTiffProfile
from rasterio.windows import Window
from rasterio.crs import CRS
import rasterio
import numpy as np
import torch

from forestvision.datasets.utils import save_cog


class MultiTaskPredictionSaver(BasePredictionWriter):
    """Saves multi-task model predictions to separate GeoTIFF files per task.

    This class handles multi-task model outputs where each task may require
    different data types (e.g., uint8 for classification, float32 for regression).
    Each task's predictions are saved to a separate GeoTIFF file.

    The output file naming follows:
        {tile_id}_{task_name}_{model_class}.tif

    Example:
        >>> from forestvision.deploy.multitask_writer import MultiTaskPredictionSaver
        >>> saver = MultiTaskPredictionSaver(
        ...     output_dir="predictions/",
        ...     task_types=["classification", "regression"],
        ...     task_names=["forest_type", "biomass"],
        ...     crs="EPSG:4326",
        ... )
        >>> trainer = Trainer(callbacks=[saver])
        >>> # To export regression predictions as integers:
        >>> saver_int = MultiTaskPredictionSaver(
        ...     output_dir="predictions/",
        ...     task_types=["classification", "regression"],
        ...     task_names=["forest_type", "biomass"],
        ...     export_dtypes={"forest_type": "uint8", "biomass": "int16"},
        ...     crs="EPSG:4326",
        ... )
    """

    def __init__(
        self,
        output_dir: str | Path,
        task_types: List[str],
        task_names: Optional[List[str]] = None,
        task_dtypes: Optional[Dict[str, str]] = None,
        write_interval: str = "batch",
        crs: Optional[CRS] = None,
        crop: int = 0,
        overwrite: bool = False,
        nodata_values: Optional[Dict[str, Any]] = None,
        target_stats: Optional[Dict[str, Any]] = None,
        export_dtypes: Optional[Dict[str, str]] = None,
        round_predictions: Optional[Dict[str, bool]] = None,
    ):
        """Initialize the MultiTaskPredictionSaver.

        Args:
            output_dir: Directory to save prediction GeoTIFFs.
            task_types: List of task types ("classification" or "regression").
            task_names: Optional list of human-readable task names. If None,
                defaults to "task_0", "task_1", etc.
            task_dtypes: Dictionary mapping task names to rasterio data types for storage.
                Deprecated: Use export_dtypes instead. If provided, must match export_dtypes.
            write_interval: When to write predictions ("batch" or "epoch").
            crs: Coordinate reference system for output GeoTIFFs.
            crop: Number of pixels to crop from edges of predictions.
            overwrite: Whether to overwrite existing files.
            nodata_values: Dictionary mapping task names to NoData values.
                If None, defaults are: classification -> 255, regression -> -9999.
            export_dtypes: Dictionary mapping task names to numpy export dtypes.
                Specifies the dtype for exported predictions (e.g., "int16", "uint8").
                This replaces the deprecated round_predictions parameter.
            round_predictions: Deprecated. Use export_dtypes instead.
        """
        super().__init__(write_interval)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.task_types = task_types
        self.task_names = task_names or [f"task_{i}" for i in range(len(task_types))]
        self.crs = crs
        self.crop = crop
        self.overwrite = overwrite

        # Validate task configuration
        if len(self.task_names) != len(task_types):
            raise ValueError(
                f"Number of task_names ({len(self.task_names)}) must match "
                f"number of task_types ({len(task_types)})"
            )

        # Set up export dtypes for processed predictions (also used for storage)
        if export_dtypes is not None:
            self.export_dtypes = export_dtypes
            # Use export_dtypes for storage dtype as well (they must match)
            self.task_dtypes = export_dtypes.copy()
        elif round_predictions is not None:
            # Backward compatibility: convert round_predictions to export_dtypes
            self.export_dtypes = {}
            for name, ttype in zip(self.task_names, task_types):
                if round_predictions.get(name, ttype == "classification"):
                    self.export_dtypes[name] = "int16" if ttype == "regression" else "uint8"
                else:
                    self.export_dtypes[name] = "float32"
            self.task_dtypes = self.export_dtypes.copy()
        else:
            # Default: classification -> uint8, regression -> float32
            self.export_dtypes = {}
            for name, ttype in zip(self.task_names, task_types):
                self.export_dtypes[name] = "uint8" if ttype == "classification" else "float32"
            self.task_dtypes = self.export_dtypes.copy()

        # Override with user-provided task_dtypes if given (for backward compatibility)
        if task_dtypes is not None:
            self.task_dtypes.update(task_dtypes)

        # Set up default NoData values
        if nodata_values is None:
            self.nodata_values = {}
            for name, ttype in zip(self.task_names, task_types):
                self.nodata_values[name] = 255 if ttype == "classification" else -9999
        else:
            self.nodata_values = nodata_values

        # Store target stats for denormalization
        self.target_stats = target_stats

    def _generate_tile_id(self, bounds) -> str:
        """Generate a unique tile ID from bounding box coordinates."""
        # Handle both BoundingBox objects and tuples (minx, maxx, miny, maxy)
        if hasattr(bounds, "minx"):
            minx, maxx, miny, maxy = bounds.minx, bounds.maxx, bounds.miny, bounds.maxy
        else:
            minx, maxx, miny, maxy = bounds
        return hashlib.md5(f"({minx}, {miny}, {maxx}, {maxy})".encode()).hexdigest()

    def _get_profile(self, task_name: str, height: int, width: int, bounds) -> dict:
        """Generate a rasterio profile for a given task."""
        dtype = self.task_dtypes[task_name]
        nodata = self.nodata_values[task_name]

        profile = DefaultGTiffProfile(
            count=1,
            dtype=dtype,
            width=width,
            height=height,
            nodata=nodata,
        )

        # Handle both BoundingBox objects and tuples (minx, maxx, miny, maxy)
        if hasattr(bounds, "minx"):
            minx, maxx, miny, maxy = bounds.minx, bounds.maxx, bounds.miny, bounds.maxy
        else:
            minx, maxx, miny, maxy = bounds

        profile.update(
            transform=rasterio.transform.from_bounds(
                minx,
                miny,
                maxx,
                maxy,
                width=width,
                height=height,
            ),
            crs=self.crs,
        )

        return profile

    def postprocess_fn(
        self,
        predictions: torch.Tensor,
        metadata: Dict[str, Any],
    ) -> np.ndarray:
        """Post-process all task predictions together.

        This method handles denormalization of regression tasks, casting to
        export dtypes, and any cross-task logic. Override this method
        in subclasses to implement custom cross-task modifications.

        Args:
            predictions: Tensor of shape [num_tasks, H, W] with raw model outputs.
            metadata: Dictionary containing:
                - 'task_names': List of task names in order
                - 'task_types': List of task types in order
                - 'export_dtypes': Dictionary mapping task names to export dtypes
                - 'target_stats': Stats for denormalization (mean, std)
                - 'nodata_values': Dictionary of NoData values

        Returns:
            Processed numpy array of shape [num_tasks, H, W] ready for saving.

        Example:
            >>> class MySaver(MultiTaskPredictionSaver):
            ...     def postprocess_fn(self, predictions, metadata):
            ...         # First do standard denormalization
            ...         predictions = super().postprocess_fn(predictions, metadata)
            ...         # Then apply custom cross-task logic
            ...         task_names = metadata["task_names"]
            ...         fortypba_idx = task_names.index("fortypba")
            ...         cancov_idx = task_names.index("cancov")
            ...         mask = predictions[cancov_idx] < 500
            ...         predictions[fortypba_idx][mask] = 0
            ...         return predictions
        """
        task_names = metadata["task_names"]
        task_types = metadata["task_types"]
        export_dtypes = metadata.get("export_dtypes", {})
        target_stats = metadata.get("target_stats")
        nodata_values = metadata.get("nodata_values", {})

        # Move to CPU and convert to numpy
        predictions = predictions.cpu().numpy()

        # Process each task
        processed_preds = []
        regression_indices = []

        for task_idx, (task_name, task_type) in enumerate(zip(task_names, task_types)):
            pred = predictions[task_idx]
            export_dtype = export_dtypes.get(task_name, "float32")

            if task_type == "classification":
                # Ensure classification values are integers
                pred = pred.round().astype(export_dtype)
            else:  # regression
                # Denormalize regression predictions: pred * std + mean
                if target_stats is not None:
                    mean = target_stats.get("mean", [0.0])
                    std = target_stats.get("std", [1.0])
                    if task_idx < len(mean):
                        pred = pred / 100 * std[task_idx] + mean[task_idx]

                # Handle NaN and inf values before casting to integer
                # Replace with -1 (NoData) to ensure proper integer casting
                pred = np.nan_to_num(pred, nan=-1, posinf=-1, neginf=-1)

                # Truncate negative values (NoData = -1)
                pred[pred < 0] = -1

                # Cast to export dtype (handles both integer and float exports)
                pred = pred.astype(export_dtype)

                pred = pred.copy()  # Make writeable
                nodata_val = nodata_values.get(task_name, -1)
                pred[pred == nodata_val] = nodata_val

                # Track regression task indices for NoData masking
                regression_indices.append(task_idx)

            processed_preds.append(pred)

        # Stack into array
        processed_preds = np.stack(processed_preds, axis=0)

        # Create NoData mask from all regression tasks (where value == -1)
        if regression_indices and "fortypba" in task_names:
            nodata_mask = np.zeros_like(processed_preds[0], dtype=bool)
            for reg_idx in regression_indices:
                nodata_mask |= processed_preds[reg_idx] == -1

            # Apply NoData mask to fortypba (set to 0)
            fortypba_idx = task_names.index("fortypba")
            processed_preds[fortypba_idx][nodata_mask] = 0

        # ((fortypba in [10,11,12]) AND (cancov < 2500)) OR
        # ((fortypba in [10,11,12]) AND (qmd_dom < 300) AND (cancov < 6000))
        # Valid pixels keep original fortypba value, invalid pixels set to 0
        if "fortypba" in task_names and "cancov" in task_names:
            fortypba_idx = task_names.index("fortypba")
            cancov_idx = task_names.index("cancov")

            fortypba = processed_preds[fortypba_idx]
            cancov = processed_preds[cancov_idx]

            # Condition 1: fortypba in [10, 11, 12]
            valid_fortypba = np.isin(fortypba, [10, 11, 12])

            # First part: valid_fortypba AND (cancov < 2500)
            part1 = valid_fortypba & (cancov < 2500)

            # Second part (if qmd_dom exists): valid_fortypba AND (qmd_dom < 300) AND (cancov < 6000)
            if "qmd_dom" in task_names:
                qmd_dom_idx = task_names.index("qmd_dom")
                qmd_dom = processed_preds[qmd_dom_idx]
                part2 = valid_fortypba & (qmd_dom < 300) & (cancov < 6000)
            else:
                part2 = np.zeros_like(part1, dtype=bool)

            # Valid mask: pixels that satisfy either condition
            valid_mask = part1 | part2

            # Remap: keep original fortypba value if valid, else set to 0
            processed_preds[fortypba_idx] = np.where(valid_mask, fortypba, 0)

        return processed_preds

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: Dict[str, Any],
        batch_indices: Any,
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Write predictions for a batch.

        Args:
            trainer: Lightning Trainer instance.
            pl_module: Lightning Module being evaluated.
            prediction: Dictionary with "predictions" tensor and "batch" metadata.
            batch_indices: Indices for the current batch.
            batch: Original batch data containing "bounds" for georeferencing.
            batch_idx: Index of current batch.
            dataloader_idx: Index of current dataloader.
        """
        # Extract predictions tensor: [B, num_tasks, H, W]
        predictions = prediction["predictions"]
        batch_metadata = prediction.get("batch", batch)

        # Get bounds from batch for georeferencing
        bounds_list = batch_metadata.get("bounds", None)
        if bounds_list is None:
            # Fallback: try to get from original batch
            bounds_list = batch.get("bounds", None)

        if bounds_list is None:
            raise ValueError(
                "Batch must contain 'bounds' for georeferencing. "
                "Ensure the datamodule's collate_fn includes bounds."
            )

        batch_size = predictions.shape[0]

        # Prepare metadata for postprocess_fn
        metadata = {
            "task_names": self.task_names,
            "task_types": self.task_types,
            "export_dtypes": self.export_dtypes,
            "target_stats": self.target_stats,
            "nodata_values": self.nodata_values,
        }

        # Process each sample in the batch
        for sample_idx in range(batch_size):
            bounds = bounds_list[sample_idx]
            tile_id = self._generate_tile_id(bounds)

            # Get raw predictions for this sample: [num_tasks, H, W]
            raw_preds = predictions[sample_idx]

            # Apply postprocessing (denormalization + cross-task logic)
            processed_preds = self.postprocess_fn(raw_preds, metadata)

            # Save each task's prediction
            for task_idx, (task_name, task_type) in enumerate(
                zip(self.task_names, self.task_types)
            ):
                pred = processed_preds[task_idx]

                # Generate output filename
                filepath = self.output_dir / f"{tile_id}_{task_name}_{pl_module.__class__.__name__}.tif"

                # Get raster profile
                height, width = pred.shape[-2], pred.shape[-1]

                profile = self._get_profile(task_name, height, width, bounds)

                # Handle additional cropping if specified
                window = None
                if self.crop > 0:
                    window = Window(
                        self.crop,
                        self.crop,
                        width - self.crop * 2,
                        height - self.crop * 2,
                    )
                    # The window parameter tells save_cog which region to extract
                    # and it will compute the correct transform via window_transform

                # Save as Cloud-Optimized GeoTIFF
                save_cog(
                    pred,
                    profile,
                    str(filepath),
                    overwrite=self.overwrite,
                    window=window,
                )

    def write_on_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        predictions: List[Any],
        batch_indices: Any,
    ) -> None:
        """Write predictions at the end of an epoch.

        This method aggregates all batch predictions and could be used for
        post-processing or summary statistics. For now, it logs completion.

        Args:
            trainer: Lightning Trainer instance.
            pl_module: Lightning Module being evaluated.
            predictions: List of all predictions from the epoch.
            batch_indices: Indices for all batches.
        """
        # Currently, we write per-batch. This method could be extended
        # for epoch-level operations like mosaic generation.
        print(f"Prediction epoch complete. Files saved to: {self.output_dir}")
