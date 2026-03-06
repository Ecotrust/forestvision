"""Multi-task prediction writer for saving multi-task model outputs to GeoTIFF."""

import os
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
from rasterio.profiles import DefaultGTiffProfile
from rasterio.windows import Window
from rasterio.crs import CRS
import rasterio

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
        ...     task_dtypes=["uint8", "float32"],
        ...     crs="EPSG:4326",
        ... )
        >>> trainer = Trainer(callbacks=[saver])
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
    ):
        """Initialize the MultiTaskPredictionSaver.

        Args:
            output_dir: Directory to save prediction GeoTIFFs.
            task_types: List of task types ("classification" or "regression").
            task_names: Optional list of human-readable task names. If None,
                defaults to "task_0", "task_1", etc.
            task_dtypes: Dictionary mapping task names to rasterio data types.
                If None, defaults are: classification -> uint8, regression -> float32.
            write_interval: When to write predictions ("batch" or "epoch").
            crs: Coordinate reference system for output GeoTIFFs.
            crop: Number of pixels to crop from edges of predictions.
            overwrite: Whether to overwrite existing files.
            nodata_values: Dictionary mapping task names to NoData values.
                If None, defaults are: classification -> 255, regression -> -9999.
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

        # Set up default dtypes for each task
        if task_dtypes is None:
            self.task_dtypes = {}
            for name, ttype in zip(self.task_names, task_types):
                self.task_dtypes[name] = "uint8" if ttype == "classification" else "float32"
        else:
            self.task_dtypes = task_dtypes

        # Set up default NoData values
        if nodata_values is None:
            self.nodata_values = {}
            for name, ttype in zip(self.task_names, task_types):
                self.nodata_values[name] = 255 if ttype == "classification" else -9999
        else:
            self.nodata_values = nodata_values

    def _generate_tile_id(self, bounds) -> str:
        """Generate a unique tile ID from bounding box coordinates."""
        minx, maxx, miny, maxy = bounds.minx, bounds.maxx, bounds.miny, bounds.maxy
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

        profile.update(
            transform=rasterio.transform.from_bounds(
                bounds.minx,
                bounds.miny,
                bounds.maxx,
                bounds.maxy,
                width=width,
                height=height,
            ),
            crs=self.crs,
        )

        return profile

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
        num_tasks = predictions.shape[1]

        # Process each sample in the batch
        for sample_idx in range(batch_size):
            bounds = bounds_list[sample_idx]
            tile_id = self._generate_tile_id(bounds)

            # Process each task
            for task_idx, (task_name, task_type) in enumerate(
                zip(self.task_names, self.task_types)
            ):
                # Extract prediction for this task: [H, W]
                pred = predictions[sample_idx, task_idx]

                # Move to CPU and convert to numpy
                pred = pred.cpu().numpy()

                # Handle NoData values based on task type
                if task_type == "classification":
                    # Ensure classification values are integers
                    pred = pred.round().astype(self.task_dtypes[task_name])
                else:  # regression
                    # Keep float values, replace NaN with NoData
                    pred = pred.astype(self.task_dtypes[task_name])
                    pred = pred.copy()  # Make writeable
                    pred[pred == self.nodata_values[task_name]] = self.nodata_values[task_name]

                # Generate output filename
                filepath = self.output_dir / f"{tile_id}_{task_name}_{pl_module.__class__.__name__}.tif"

                # Get raster profile
                height, width = pred.shape[-2], pred.shape[-1]
                profile = self._get_profile(task_name, height, width, bounds)

                # Handle cropping if specified
                window = None
                if self.crop > 0:
                    window = Window(
                        self.crop,
                        self.crop,
                        width - self.crop * 2,
                        height - self.crop * 2,
                    )
                    pred = pred[self.crop : height - self.crop, self.crop : width - self.crop]
                    profile.update(
                        width=width - self.crop * 2,
                        height=height - self.crop * 2,
                    )

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
