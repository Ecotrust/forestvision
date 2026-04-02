# Inference Pipeline Guide

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Prerequisites](#3-prerequisites)
4. [Configuration](#4-configuration)
5. [Usage](#5-usage)
6. [Output Format](#6-output-format)
7. [API Reference](#7-api-reference)
8. [Troubleshooting](#8-troubleshooting)
9. [Best Practices](#9-best-practices)

---

## 1. Overview

This document covers the inference pipeline for generating predictions with trained ForestVision models. The pipeline supports generating task-specific predictions (segmentation and regression) and saving them as georeferenced GeoTIFF files. The inference pipeline enables:

- **Single-pass inference**: Generate predictions for multiple tasks simultaneously
- **Task-specific outputs**: Segmentation (semantic classes) and regression (continuous values)
- **Georeferenced outputs**: Cloud-Optimized GeoTIFFs (COG) with proper CRS and bounds
- **Flexible data types**: uint8 for segmentation, float32 for regression
- **Distributed inference**: Compatible with PyTorch Lightning's distributed training

### Supported Models

| Model | Description | Use Case |
|-------|-------------|----------|
| `MultiTaskUNet` | Multi-task U-Net with ResNet backbone | Segmentation + Regression tasks |
| `ResMTUNet` | ResNet-based Multi-Task U-Net | Transfer learning with pretrained weights |

### Key Components

| Component | Purpose | Location |
|-----------|---------|----------|
| `MultiTaskUNet.predict_step()` | Converts model logits to task predictions | `forestvision/trainers/litunet.py` |
| `BaseGeoDataModule.predict_dataloader()` | Loads inference data with geospatial metadata | `forestvision/datamodules/base.py` |
| `MultiTaskPredictionSaver` | Saves predictions to GeoTIFF files | `forestvision/deploy/multitask_writer.py` |
| `demo_predict.py` | Demonstration script for the pipeline | `scripts/demo_predict.py` |

---

## 2. Architecture

### Inference Data Flow

```
Input Tiles (GeoTIFF)
    |
    ▼
┌─────────────────────────────────────────┐
│ predict_dataloader()                    │
│  - Loads imagery with bounds/crs        │
│  - Returns batch dict with "bounds"     │
└─────────────────────────────────────────┘
    |
    ▼
┌─────────────────────────────────────────┐
│ MultiTaskUNet.predict_step()            │
│  - Forward pass → logits               │
│  - Segmentation: softmax + argmax     │
│  - Regression: raw/tanh values          │
│  - Returns {predictions, batch}         │
└─────────────────────────────────────────┘
    |
    ▼
┌─────────────────────────────────────────┐
│ MultiTaskPredictionSaver                │
│  - One GeoTIFF per task per tile        │
│  - Task-specific dtypes/NoData          │
│  - Cloud-Optimized GeoTIFF format       │
└─────────────────────────────────────────┘
    |
    ▼
Output: {tile_id}_{task_name}_MultiTaskUNet.tif
```

### Prediction Processing

The `predict_step()` method handles task-specific post-processing:

```python
def predict_step(self, batch, batch_idx, dataloader_idx=None):
    x = batch["image"]
    y_hat = self(x)  # Raw model output [B, total_channels, H, W]
    
    predictions = []
    channel_offset = 0
    
    for task_type, num_classes in zip(self.task_types, self.num_classes_per_task):
        task_output = y_hat[:, channel_offset:channel_offset + num_classes]
        
        if task_type == "segmentation":
            # Apply softmax + argmax for class predictions
            probs = task_output.softmax(dim=1)
            pred = torch.argmax(probs, dim=1, keepdim=True)
        else:  # regression
            # Keep regression channel as-is
            pred = task_output[:, 0:1]
            
        predictions.append(pred)
        channel_offset += num_classes
    
    return {
        "predictions": torch.cat(predictions, dim=1),  # [B, num_tasks, H, W]
        "batch": batch,  # Preserves bounds, crs metadata
    }
```

---

## 3. Prerequisites

### Model Checkpoint

You need a trained model checkpoint (`.ckpt` file) from PyTorch Lightning training:

```bash
checkpoints/
├── best.ckpt          # Best model based on validation metric
├── last.ckpt          # Last epoch checkpoint
└── epoch_*.ckpt       # Intermediate checkpoints
```

### Input Data

Prepare your inference tiles:

1. **GeoJSON file** with tile boundaries (required)
   ```bash
   data/inference/tiles.geojson
   ```

2. **Imagery tiles** organized by dataset type
   ```
   data/inference/
   ├── sentinel/2021/leafon/*.tif
   ├── dem/2021/*.tif
   └── climatena/*.tif
   ```

### Environment

Ensure your environment is set up:

```bash
# Load environment variables
source .env

# Verify GPU access (optional but recommended)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 4. Configuration

### predict.yaml Template

Create a `configs/predict.yaml` file for use with the TorchGeo CLI:

```yaml
# Prediction configuration for MultiTaskUNet
# Usage: torchgeo predict --config configs/predict.yaml --ckpt_path checkpoints/model.ckpt

# Model configuration - should match your training config
model:
  class_path: forestvision.trainers.litunet.MultiTaskUNet
  init_args:
    in_channels: 15
    task_types:
      - segmentation
      - regression
      - regression
    num_classes_per_task:
      - 14   # forest_type (14 classes)
      - 1    # cancov (regression)
      - 1    # biomass (regression)
    loss: focal
    ignore_index: -2147483648
    # Task band names for output file naming
    task_band_names:
      - forest_type
      - canopy_cover
      - biomass

# Data configuration for inference
data:
  class_path: forestvision.datamodules.base.BaseGeoDataModule
  init_args:
    root: data/
    year: 2021
    batch_size: 16
    num_workers: 4
    patch_size: 120
    # IMPORTANT: Set path to your inference tiles GeoJSON
    predict_tiles_path: inference/tiles.geojson
    # Input datasets configuration (same as training)
    input_configs:
      - dataset_class: forestvision.datasets.geesentinel.GEESentinel2
        path_template: datasets/geesentinel2/{year}/leafon
        bands:
          - B2
          - B3
          - B4
          - B5
          - B6
          - B7
          - B8
          - B8A
          - B11
          - B12
        kwargs:
          season: leafon
          res: 10
    # Target configs can be empty for pure inference
    target_configs: []
    # Input transforms (same normalization as training)
    input_transforms:
      class_path: torchvision.transforms.v2.Compose
      init_args:
        transforms:
          - class_path: forestvision.transforms.Normalize
            init_args:
              on_key: image
              mean: [...]  # From training stats
              std: [...]

# Trainer configuration
trainer:
  accelerator: gpu
  devices: 1
  inference_mode: true
  enable_progress_bar: true
  logger: false  # Disable logging for inference
  
  callbacks:
    - class_path: forestvision.deploy.multitask_writer.MultiTaskPredictionSaver
      init_args:
        output_dir: predictions/
        task_types:
          - segmentation
          - regression
          - regression
        task_names:
          - forest_type
          - canopy_cover
          - biomass
        crs: "EPSG:4326"
        crop: 0  # Number of pixels to crop from edges
        overwrite: true
```

### Task Configuration

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `task_types` | List[str] | Task types in order | `["segmentation", "regression"]` |
| `task_names` | List[str] | Human-readable names | `["forest_type", "agb"]` |
| `task_dtypes` | Dict[str, str] | Rasterio dtype per task | `{"forest_type": "uint8"}` |
| `nodata_values` | Dict[str, Any] | NoData value per task | `{"forest_type": 255}` |

### Default Data Types and NoData Values

If not specified, the following defaults are used:

| Task Type | Default dtype | Default NoData |
|-----------|---------------|----------------|
| segmentation | uint8 | 255 |
| regression | float32 | -9999 |

---

## 5. Usage

### Method 1: Programmatic Usage

For custom inference pipelines:

```python
import torch
from lightning import Trainer
from forestvision.trainers.litunet import MultiTaskUNet
from forestvision.deploy import MultiTaskPredictionSaver
from forestvision.datamodules.base import BaseGeoDataModule

# Load model
model = MultiTaskUNet.load_from_checkpoint("checkpoints/model.ckpt")
model.eval()

# Setup datamodule
datamodule = BaseGeoDataModule(...)
datamodule.setup("predict")

# Create prediction saver
saver = MultiTaskPredictionSaver(
    output_dir="predictions/",
    task_types=model.task_types,
    task_names=model.task_band_names,
    crs="EPSG:4326",
)

# Run inference
trainer = Trainer(
    accelerator="gpu",
    devices=1,
    callbacks=[saver],
)
trainer.predict(model, datamodule=datamodule)
```

### Method 2: Using TorchGeo CLI

If you have a `predict.yaml` configuration:

```bash
torchgeo predict --config configs/predict.yaml --ckpt_path checkpoints/model.ckpt
```

---

## 6. Output Format

### File Naming Convention

```
{tile_id}_{task_name}_{model_class}.tif
```

**Example:**
```
a1b2c3d4_forest_type_MultiTaskUNet.tif
a1b2c3d4_biomass_MultiTaskUNet.tif
```

The `tile_id` is generated from the bounding box coordinates using MD5 hash for uniqueness.

### GeoTIFF Structure

Each output GeoTIFF contains:

| Attribute | Description |
|-----------|-------------|
| `count` | 1 (single band per task) |
| `dtype` | Task-specific (uint8/float32) |
| `crs` | Coordinate reference system |
| `transform` | Affine transformation from bounds |
| `nodata` | Task-specific NoData value |

### Cloud-Optimized GeoTIFF (COG) Features

All outputs are saved as COGs with:
- Tiled organization (256x256 internal tiles)
- LZW compression
- Overviews for efficient visualization at multiple scales

---

## 7. API Reference

### MultiTaskPredictionSaver

```python
class MultiTaskPredictionSaver(BasePredictionWriter):
    def __init__(
        self,
        output_dir: str | Path,           # Output directory
        task_types: List[str],            # ["segmentation", "regression"]
        task_names: Optional[List[str]],  # ["forest_type", "biomass"]
        task_dtypes: Optional[Dict[str, str]],  # {"forest_type": "uint8"}
        write_interval: str = "batch",    # "batch" or "epoch"
        crs: Optional[CRS] = None,        # Coordinate reference system
        crop: int = 0,                    # Edge pixels to crop
        overwrite: bool = False,          # Overwrite existing
        nodata_values: Optional[Dict[str, Any]],  # {"forest_type": 255}
    )
```

### MultiTaskUNet.predict_step

```python
def predict_step(
    self,
    batch: Dict[str, Any],
    batch_idx: int,
    dataloader_idx: Optional[int] = None
) -> Dict[str, Any]:
    """
    Args:
        batch: Dictionary containing "image" tensor and "bounds"
        batch_idx: Index of current batch
        dataloader_idx: Index of current dataloader
        
    Returns:
        Dictionary with:
            - "predictions": Tensor [B, num_tasks, H, W]
            - "batch": Original batch with metadata
    """
```

---

## 8. Troubleshooting

### Common Issues

#### Issue: "Batch must contain 'bounds' for georeferencing"

**Cause:** The DataModule's collate_fn is not including bounds in the batch.

**Solution:** Ensure your DataModule properly handles bounds in `predict_dataloader()`:

```python
def predict_dataloader(self):
    dataset = self._create_predict_dataset()
    return DataLoader(
        dataset,
        batch_size=self.batch_size,
        collate_fn=self._collate_fn,  # Must include bounds
    )
```

#### Issue: Predictions have wrong CRS or are misaligned

**Cause:** CRS not properly passed to `MultiTaskPredictionSaver`.

**Solution:** Explicitly set the CRS when creating the saver:

```python
saver = MultiTaskPredictionSaver(
    output_dir="predictions/",
    task_types=["segmentation"],
    crs="EPSG:4326",  # Match your input data CRS
)
```

#### Issue: CUDA out of memory during inference

**Cause:** Batch size too large for available VRAM.

**Solution:** Reduce batch size or use CPU inference:

```python
trainer = Trainer(
    accelerator="cpu",  # Fallback to CPU
    callbacks=[saver],
)
```

Or use gradient accumulation for larger effective batch sizes.

#### Issue: Segmentation predictions are floats instead of integers

**Cause:** The output dtype is not properly configured.

**Solution:** Ensure task_dtypes specifies uint8 for segmentation:

```python
saver = MultiTaskPredictionSaver(
    task_types=["segmentation", "regression"],
    task_dtypes={
        "forest_type": "uint8",     # Integer classes
        "biomass": "float32",        # Continuous values
    },
)
```

---

## 9. Best Practices

### 1. Preparing Inference Data

- **Tile size**: Use large tiles (e.g. 512x512, 1024x1024).
- **Overlap**: Consider overlap for seamless mosaics (use `crop` parameter to remove borders)
- **CRS consistency**: Ensure all input tiles use the same CRS

### 2. Batch Size Selection

| GPU Memory | Recommended Batch Size | Notes |
|------------|------------------------|-------|
| 8 GB | 2-4 | Use for 512x512 tiles |
| 16 GB | 4-8 | Good balance for most cases |
| 24+ GB | 8-16 | Maximum throughput |

### 3. Output Organization

Organize predictions by model run:

```
predictions/
├── run_2024_03_06/
│   ├── forest_type/
│   └── biomass/
└── run_2024_03_07/
    ├── forest_type/
    └── biomass/
```

### 4. Post-Processing

For creating seamless mosaics from overlapping predictions, use `scripts/create_mosaic.py` with appropriate blending options. See [Create Mosaic Guide](docs/create_mosaic.md) for details.

### 5. Performance Tips

1. **Use GPU inference** when available (10-50x faster than CPU)
2. **Disable logging** during inference (`logger: false` in config)
3. **Use multiple workers** for data loading (`num_workers: 4`)
4. **Enable mixed precision** (`precision: 16`) for faster inference on compatible GPUs

*Last updated: 2026-03-07*
