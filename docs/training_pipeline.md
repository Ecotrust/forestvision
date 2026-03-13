# ForestVision Training Pipeline Guide

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture Overview](#2-architecture-overview)
3. [Data Preparation](#3-data-preparation)
4. [Transform Pipeline](#4-transform-pipeline)
5. [Model Configuration](#5-model-configuration)
6. [Training Execution](#6-training-execution)
7. [Hyperparameter Optimization](#7-hyperparameter-optimization)
8. [Configuration Reference](#8-configuration-reference)
9. [Reproducibility](#9-reproducibility)
10. [Troubleshooting](#10-troubleshooting)
11. [Appendix](#11-appendix)

---

## 1. Overview

This guide covers the ForestVision training pipeline for multi-task geospatial deep learning. ForestVision enables simultaneous prediction of forest type classification and continuous regression targets (canopy cover, biomass) from Sentinel-2 imagery and auxiliary data.

### What This Guide Covers

- Data preparation and sampling
- Multi-source data download from Google Earth Engine
- Transform pipeline configuration
- Model architecture selection and configuration
- Training execution with PyTorch Lightning
- Hyperparameter optimization with Optuna
- Inference and prediction generation
- Reproducibility best practices

---

## 2. Architecture Overview

### 2.1 High-Level Data Flow

```
┌──────────────────┐      ┌──────────────┐      ┌──────────────┐      ┌───────────┐
│  sample_tiles.py │      │prepare_data.py│      │ DataModule   │      │   Model   │
│ (Sampling/Split) │─────▶│ (GEE Download)│─────▶│ (Loading)    │─────▶│ (Training)│
└────────┬─────────┘      └──────┬───────┘      └──────┬───────┘      └─────┬─────┘
     │                       │                     │                    │
     ▼                       ▼                     ▼                    ▼
   ┌───────────┐          ┌──────────┐          ┌──────────┐          ┌──────────┐
   │ GeoJSON   │          │ Cached   │          │ Tile     │          │ Metrics  │
   │ Tile Sets │          │ Rasters  │          │ Sampler  │          │ Artifacts│
   └───────────┘          └──────────┘          └──────────┘          └──────────┘
```

### 2.2 Core Components

| Component | Purpose | Key Classes |
|-----------|---------|-------------|
| **Datasets** | Abstract GEE/local data access | `GEESentinel2`, `GEE3Dep`, `GNNForestAttr` |
| **DataModules** | Manage data loading & transforms | `GNNDataModule`, `BaseGeoDataModule` |
| **Transforms** | Preprocessing & augmentation | `SelectBands`, `Normalize`, `AppendNDVI` |
| **Models** | Neural network architectures | `ResMTUNet`, `MTUNet`, `MultiTaskUNet` |
| **Trainers** | LightningModules for training | `MultiTaskUNet` (LightningModule) |

### 2.3 IntersectionDataset Pattern

ForestVision uses TorchGeo's `IntersectionDataset` to combine multiple input sources with multiple targets while ensuring spatial alignment:

```python
# Combine datasets using & operator
sentinel = GEESentinel2(paths="...", bands=["B2", "B3", "B4"])
dem = GEE3Dep(paths="...", bands=["elevation"])
gnn = GNNForestAttr(paths="...", bands=["fortypba"])

input_dataset = sentinel & dem      # Combines inputs
train_dataset = input_dataset & gnn  # Full training set
```

**Benefits:**
- Automatic alignment: Only returns samples where all datasets have coverage
- Lazy evaluation: No data loaded until requested
- Composable: Easy to add/remove datasets without reprocessing

### 2.4 Path Template System

The DataModule uses flexible path templates with placeholders:

| Placeholder | Description | Example Value |
|-------------|-------------|---------------|
| `{root}` | Base data directory | `"data/fortypba"` |
| `{year}` | Acquisition year | `2021` |
| `{stage}` | Current stage | `"training"` or `"validation"` |

**Examples:**
```yaml
# Shared directory (spatial split)
path_template: "training/sentinel/{year}"

# Separate directories (temporal split)
path_template: "{stage}/sentinel/{year}"
```

---

## 3. Data Preparation

### 3.1 Tile Sampling

Generate balanced training/validation splits with `scripts/sample_tiles.py`:

```bash
# Basic usage (128x128 tiles)
python scripts/sample_tiles.py \
  --output-path data/fortypba/ \
  --sample-size 8000

# For 256x256 tiles
python scripts/sample_tiles.py \
  --output-path data/fortypba/ \
  --tile-size 256 \
  --sample-size 8000

# K-fold cross-validation
python scripts/sample_tiles.py \
  --output-path data/fortypba/ \
  --tile-size 256 \
  --k-folds 5
```

**Balancing Strategies:**

| Strategy | Description |
|----------|-------------|
| `none` | Simple random sampling |
| `equal` | Equal number of tiles per dominant forest type |
| `proportional` | Maintains natural distribution |
| `inverse_freq` | Boosts rare classes using inverse frequency weights |
| `capped` | Pixel-level balancing ensuring sufficient representation |

**Outputs:**
- `{name}_train.geojson` / `{name}_val.geojson`: Tile boundaries
- `{name}_train_frequencies.csv`: Pixel counts per class
- `{name}_report.md`: Sampling summary
- `{name}_weights.csv`: Class weights for loss functions

### 3.2 Data Download

Download imagery from Google Earth Engine with `scripts/prepare_data.py`:

```bash
# Download training data
python scripts/prepare_data.py \
  --config configs/osugnn_best.yaml \
  --on-keys image mask \
  --year 2021
```

**What it does:**
1. Creates GEE download URLs for each tile
2. Downloads Sentinel-2, DEM, and auxiliary data
3. Computes normalization statistics
4. Saves to `data/fortypba/training/`

**Stats file generation:**
Statistics are automatically computed and saved to:
```
data/fortypba/train_stats_2021.pt
```

### 3.3 Data Configuration

Define input and target datasets in your YAML config:

```yaml
data:
  class_path: forestvision.datamodules.GNNDataModule
  init_args:
  root: data/fortypba
  year: 2021
  stats_path: train_stats_2021.pt
  
  train_tiles_path: tiles/train_128p10m.geojson
  val_tiles_path: tiles/val_128p10m.geojson
  
  input_datasets:
    - dataset_class: forestvision.datasets.GEESentinel2
    path_template: "training/geesentinel2/{year}/leafon"
    bands: ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    kwargs:
      season: leafon
      res: 10
    
    - dataset_class: forestvision.datasets.GEE3Dep
    path_template: "training/gee3dep/{year}"
    bands: ["elevation"]
    kwargs:
      res: 10
  
  target_datasets:
    - dataset_class: forestvision.datasets.GNNForestAttr
    path_template: "training/gnn/{year}"
    bands: ["fortypba", "cancov", "qmd_dom"]
    kwargs:
      res: 10
      remap: true
```

---

## 4. Transform Pipeline

### 4.1 5-Stage Pipeline

ForestVision uses a carefully ordered transform pipeline:

```
Raw Data → [Pre-aug] → [Augment] → [Post-aug] → Model
        ↓           ↓            ↓
     input_     train_      post_aug_
     transforms transforms  input_transforms
```

**Critical Principle:** Augmentations must happen BEFORE normalization.

**Why this order matters:**
- Normalization scales data to specific ranges
- Augmentations (brightness, contrast) can push normalized values out of range
- Interpolation artifacts occur on normalized values

### 4.2 Transform Stages

#### Stage 1: `input_transforms` (Pre-augmentation)

Applied to input data before augmentation:

```yaml
input_transforms:
  class_path: torchvision.transforms.v2.Compose
  init_args:
  transforms:
    - class_path: forestvision.transforms.SelectBands
    init_args:
      indices: [0, 1, 2, 3, 6, 10, 11, 12, 13, 14]  # Select specific bands
```

#### Stage 2: `target_transforms` (Pre-augmentation)

Applied to target data before augmentation:

```yaml
target_transforms:
  class_path: torchvision.transforms.v2.Compose
  init_args:
  transforms:
    - class_path: forestvision.transforms.CombineGNNDWMask
```

#### Stage 3: `train_transforms` (Augmentation - Training Only)

Applied only during training:

```yaml
train_transforms:
  class_path: torchvision.transforms.v2.Compose
  init_args:
  transforms:
    - class_path: forestvision.transforms.augmentations.RandomHorizontalFlip
    init_args:
      p: 0.5
    - class_path: forestvision.transforms.augmentations.RandomCropResize
    init_args:
      size: [112, 112]
      scale: [0.8, 1.2]
```

**When it runs:**
- Training: YES
- Validation: NO
- Test: NO

#### Stage 4 & 5: Post-Augmentation Transforms

Applied after augmentation (both train and validation):

```yaml
post_aug_input_transforms:
  class_path: torchvision.transforms.v2.Compose
  init_args:
  transforms:
    - class_path: forestvision.transforms.Normalize
    init_args:
      on_key: "image"
      identity_channels: [10]  # Skip NDVI normalization

post_aug_target_transforms:
  class_path: torchvision.transforms.v2.Compose
  init_args:
  transforms:
    - class_path: forestvision.transforms.Normalize
    init_args:
      on_key: "mask"
      identity_channels: [0]  # Don't normalize class labels
```

### 4.3 Key Configuration Principles

#### Order Matters: SelectBands BEFORE Normalize

```yaml
transforms:
  - class_path: forestvision.transforms.SelectBands
  init_args:
    indices: [0, 2, 4, 6]
  - class_path: forestvision.transforms.Normalize
  init_args:
    mean: null  # Auto-populated from stats
    std: null
```

#### Identity Channels

Skip normalization for specific channels, e.g.:
- **Class labels** (channel 0 of mask): Already integers 0-13
- **Index bands** (NDVI, SAVI): Already in [-1, 1] range

```yaml
identity_channels: [5, 6]  # Indices in SELECTED tensor, not original
```

#### NoData Handling

The `Normalize` transform automatically handles NoData values (< -1e9):

1. Detects large negative values
2. Temporarily replaces with 0 for normalization
3. Applies normalization
4. Restores original NoData values

---

## 5. Model Configuration

### 5.1 Model Architectures

#### ResMTUNet

Multi-Task U-Net with ResNet backbone for transfer learning:

```yaml
model:
  class_path: forestvision.trainers.litunet.MultiTaskUNet
  init_args:
  model: "ResMTUNet"
  backbone: "resnet50"  # resnet18/34/50/101
  pretrained: true
  in_channels: 15
  task_types: ["classification", "regression", "regression"]
  num_classes_per_task: [14, 1, 1]
```

**Backbone Performance:**

| Backbone | Parameters | Speed | Use Case |
|----------|------------|-------|----------|
| ResNet18 | ~25M | Fastest | Quick experimentation |
| ResNet34 | ~30M | Fast | Balance speed/accuracy |
| ResNet50 | ~35M | Medium | Production (default) |
| ResNet101| ~45M | Slower | Maximum accuracy |

**Multispectral Input Handling:**

ResNet expects 3-channel RGB. ForestVision modifies conv1 directly for N-channel input:

```python
# Original conv1: [64, 3, 7, 7]
# Modified conv1: [64, N, 7, 7]

# Initialization: Average RGB weights across all channels
mean_weight = original_weights.mean(dim=1, keepdim=True)
new_conv.weight.copy_(mean_weight.repeat(1, N, 1, 1))
```

### 5.2 Multi-Task Configuration

Configure tasks using `task_types` and `num_classes_per_task`:

```yaml
model:
  init_args:
  task_types: ["classification", "regression", "regression"]
  num_classes_per_task: [14, 1, 1]
  task_band_names: ["forest_type", "canopy_cover", "biomass"]
```

**Task Types:**
- `classification`: Discrete classes (softmax + argmax)
- `regression`: Continuous values (tanh or linear output)

### 5.3 Loss Functions

#### Classification Loss: Focal Loss

Addresses class imbalance by down-weighting easy examples:

```yaml
model:
  init_args:
  focal_alpha: 0.25      # Balance parameter
  focal_gamma: 2.0       # Focusing parameter
  focal_weight: [...]    # Per-class weights (14 values)
```

**Formula:**
```
FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
```

#### Regression Losses

| Loss | Description | Use Case |
|------|-------------|----------|
| `mae` | Mean Absolute Error | Robust to outliers |
| `mse` | Mean Squared Error | Penalizes large errors |
| `huber` | Smooth L1 | Balance MAE/MSE |
| `sharploss` | MAE + gradient edge loss | Preserve boundaries |
| `quantile` | Pinball loss | Predict intervals |

```yaml
model:
  init_args:
  reg_loss: "sharploss"
  sharploss_alpha: 0.6   # Edge emphasis
```

### 5.4 Multi-Task Loss Weighting

#### Uncertainty Weighting (Default)

Automatic task balancing via learnable uncertainty:

```yaml
model:
  init_args:
  loss_weighting: "uncertainty"
  init_log_vars: 0.0
```

**Formula:**
```
L_total = precision_seg * L_seg + log_var_seg +
      precision_reg * L_reg + log_var_reg
```

**Intuition:**
- Easy task (low loss) → lower uncertainty → higher precision → more weight
- Hard task (high loss) → higher uncertainty → lower precision → less weight

#### Other Weighting Strategies

| Strategy | Use Case | Configuration |
|----------|----------|---------------|
| `fixed` | Known task priorities | `seg_loss_weight`, `reg_loss_weights` |
| `equal` | Simple uniform weighting | `loss_weighting: "equal"` |
| `realtime` | Dynamic per-batch | `loss_weighting: "realtime"` |

### 5.5 Transfer Learning

Freeze backbone for transfer learning:

```yaml
model:
  init_args:
  backbone: "resnet50"
  pretrained: true
  freeze_backbone: true  # Only train decoder + heads
  lr: 0.001              # Higher LR for new layers
```

**Best practices:**
1. Start with pretrained weights (faster convergence)
2. For small datasets: freeze backbone, train decoder first
3. Gradually unfreeze for fine-tuning

---

## 6. Training Execution

### 6.1 Basic Training

Run training with `torchgeo fit`:

```bash
# Standard training
torchgeo fit --config configs/osugnn_best.yaml

# With specific epoch count
torchgeo fit --config configs/osugnn_best.yaml --trainer.max_epochs 50

# Multi-GPU training
torchgeo fit \
  --config configs/osugnn_best.yaml \
  --trainer.accelerator gpu \
  --trainer.devices 2 \
  --trainer.strategy ddp
```

### 6.2 Fast Dev Run (Testing)

Quick validation of configuration:

```bash
torchgeo fit --config configs/osugnn_best.yaml --trainer.fast_dev_run true
```

Runs 1 batch of training and validation to catch errors quickly.

### 6.3 Resuming from Checkpoint

Continue training from a saved checkpoint:

```bash
torchgeo fit \
  --config configs/osugnn_best.yaml \
  --ckpt_path checkpoints/last.ckpt
```

**Preserves:**
- Model weights
- Optimizer state
- Learning rate scheduler state
- Current epoch

### 6.4 Monitoring Training

Training metrics are automatically logged to TensorBoard:

```bash
# View logs
tensorboard --logdir lightning_logs/

# Or in VS Code
# Open Command Palette → "Python: Launch TensorBoard"
```

**Logged metrics:**
- `train_loss`, `val_loss`: Total weighted loss
- `train_task_*_raw_loss`: Raw loss per task
- `train_task_*_precision`: Task weights (uncertainty weighting)
- `seg_val_jaccard`, `reg_val_r2`: Validation metrics

---

## 7. Hyperparameter Optimization

### 7.1 Single-Objective HPO

Optimize all hyperparameters with `scripts/optuna_hpo.py`:

```bash
# Basic usage (50 trials, 30 epochs each)
python scripts/optuna_hpo.py --config configs/optuna_base.yaml

# With persistence (resumable)
python scripts/optuna_hpo.py \
  --config configs/optuna_base.yaml \
  --n-trials 100 \
  --storage sqlite:///hpo_study.db \
  --study-name my_experiment
```

**Search Space:**

| Parameter | Type | Range |
|-----------|------|-------|
| `lr` | log float | [1e-5, 1e-2] |
| `dropout` | float | [0.1, 0.7] |
| `seg_loss_weight` | float | [0.3, 0.8] |
| `weight_decay` | log float | [1e-6, 1e-2] |
| `batch_size` | categorical | [16, 24, 32, 36, 48, 64] |
| `focal_alpha` | float | [0.1, 1.0] |
| `focal_gamma` | float | [1.0, 5.0] |

**Features:**
- **Early Pruning**: MedianPruner terminates unpromising trials (~30-40% compute savings)
- **Persistence**: SQLite storage for resumable studies
- **Auto-Export**: Best config saved to YAML

### 7.2 Multi-Objective HPO

Find Pareto front for multiple objectives:

```bash
python scripts/optuna_hpo_multi.py \
  --config configs/optuna_base.yaml \
  --n-trials 100 \
  --sampler nsga2 \
  --storage sqlite:///optuna_multi.db
```

**Default Objectives:**

| Metric | Direction | Description |
|--------|-----------|-------------|
| `seg_val_jaccard` | maximize | Segmentation IoU |
| `reg_val_r2` | maximize | Regression R² |
| `reg_val_mae` | minimize | Regression MAE |

**Outputs:**
- `configs/pareto/pareto_trial_*.yaml`: Individual configurations
- `configs/pareto/pareto_front.html`: 3D interactive visualization

**Selecting a Configuration:**
1. **Best Segmentation**: Trial with highest `seg_val_jaccard`
2. **Best Regression**: Trial with highest `reg_val_r2`
3. **Balanced**: Automatically selected (closest to ideal point)

### 7.3 Focal Loss HPO

Fine-tune focal loss parameters only:

```bash
# HPO mode
python scripts/optuna_hpo_focal.py \
  --config configs/osugnn_v2.yaml \
  --n-trials 50

# Resume best trial for full training
python scripts/optuna_hpo_focal.py \
  --config configs/osugnn_v2.yaml \
  --resume-from-checkpoint optuna_logs/trial_23/checkpoints/best.ckpt \
  --additional-epochs 95
```

**Optimized Parameters:**
- `focal_alpha`: 0.1 - 1.0
- `focal_gamma`: 0.5 - 5.0
- `focal_weight`: 14 per-class weights (0.01 - 5.0 each)

---

## 8. Configuration Reference

### 8.1 Complete Training Configuration Example

```yaml
# configs/osugnn_best.yaml
seed_everything: 42

trainer:
  max_epochs: 50
  accelerator: gpu
  devices: 1
  precision: 16
  logger:
  class_path: pytorch_lightning.loggers.TensorBoardLogger
  init_args:
    save_dir: lightning_logs
    name: osugnn_best
  callbacks:
  - class_path: pytorch_lightning.callbacks.ModelCheckpoint
    init_args:
    monitor: val_loss
    mode: min
    save_top_k: 3
    filename: '{epoch:02d}-{val_loss:.4f}'
  - class_path: pytorch_lightning.callbacks.EarlyStopping
    init_args:
    monitor: val_loss
    patience: 10
    mode: min

model:
  class_path: forestvision.trainers.litunet.MultiTaskUNet
  init_args:
  in_channels: 15
  task_types: ["classification", "regression", "regression"]
  num_classes_per_task: [14, 1, 1]
  task_band_names: ["forest_type", "canopy_cover", "biomass"]
  model: "ResMTUNet"
  backbone: "resnet50"
  pretrained: true
  freeze_backbone: false
  dropout: 0.3
  focal_alpha: 0.25
  focal_gamma: 2.0
  reg_loss: "mae"
  loss_weighting: "uncertainty"
  lr: 1e-4
  weight_decay: 1e-4
  scheduler_patience: 10
  scheduler_factor: 0.5
  ignore_index: -1

data:
  class_path: forestvision.datamodules.GNNDataModule
  init_args:
  root: data/fortypba
  year: 2021
  batch_size: 32
  num_workers: 4
  stats_path: train_stats_2021.pt
  train_tiles_path: tiles/train_128p10m.geojson
  val_tiles_path: tiles/val_128p10m.geojson
  download: false
  
  input_datasets:
    - dataset_class: forestvision.datasets.GEESentinel2
    path_template: "training/geesentinel2/{year}/leafon"
    bands: ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    kwargs:
      season: leafon
      res: 10
    transforms:
      class_path: torchvision.transforms.v2.Compose
      init_args:
      transforms:
        - class_path: forestvision.transforms.AppendNDVI
        init_args:
          index_nir: 6
          index_red: 2
    
    - dataset_class: forestvision.datasets.GEE3Dep
    path_template: "training/gee3dep/{year}"
    bands: ["elevation"]
    kwargs:
      res: 10
    
    - dataset_class: forestvision.datasets.ClimateNA
    path_template: "training/climatena"
    bands: ["AHM", "MAP", "TD"]
  
  target_datasets:
    - dataset_class: forestvision.datasets.GNNForestAttr
    path_template: "training/gnn/{year}"
    bands: ["fortypba", "cancov", "qmd_dom"]
    kwargs:
      res: 10
      remap: true
  
  input_transforms:
    class_path: torchvision.transforms.v2.Compose
    init_args:
    transforms:
      - class_path: forestvision.transforms.SelectBands
      init_args:
        indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
  
  target_transforms:
    class_path: torchvision.transforms.v2.Compose
    init_args:
    transforms:
      - class_path: forestvision.transforms.CombineGNNDWMask
  
  train_transforms:
    class_path: torchvision.transforms.v2.Compose
    init_args:
    transforms:
      - class_path: forestvision.transforms.augmentations.RandomHorizontalFlip
      init_args:
        p: 0.5
      - class_path: forestvision.transforms.augmentations.RandomCropResize
      init_args:
        size: [112, 112]
        scale: [0.8, 1.2]
  
  post_aug_input_transforms:
    class_path: torchvision.transforms.v2.Compose
    init_args:
    transforms:
      - class_path: forestvision.transforms.Normalize
      init_args:
        on_key: "image"
        identity_channels: [10]
  
  post_aug_target_transforms:
    class_path: torchvision.transforms.v2.Compose
    init_args:
    transforms:
      - class_path: forestvision.transforms.Normalize
      init_args:
        on_key: "mask"
        identity_channels: [0]
        nodata: -2147483648

optimizer:
  class_path: torch.optim.Adam
  init_args:
  lr: 1e-4
  weight_decay: 1e-4
```

### 8.2 Key Parameters Quick Reference

#### Model Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `in_channels` | int | Input channels | Required |
| `task_types` | list | ["classification", "regression", ...] | Required |
| `num_classes_per_task` | list | Classes per task | Required |
| `model` | str | "MTUNet", "ResMTUNet" | "MTUNet" |
| `backbone` | str | "resnet18/34/50/101" | "resnet50" |
| `pretrained` | bool | ImageNet weights | true |
| `freeze_backbone` | bool | Freeze for transfer learning | false |
| `dropout` | float | Dropout rate | 0.0 |

#### Loss Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `focal_alpha` | float | Focal loss alpha | null |
| `focal_gamma` | float | Focal loss gamma | 2.0 |
| `focal_weight` | list | Per-class weights | null |
| `reg_loss` | str | "mae", "mse", "sharploss", "quantile" | "mae" |
| `loss_weighting` | str | "uncertainty", "fixed", "equal" | "uncertainty" |

#### Training Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `lr` | float | Learning rate | 1e-4 |
| `weight_decay` | float | Weight decay | 1e-4 |
| `scheduler_patience` | int | LR scheduler patience | 10 |
| `scheduler_factor` | float | LR reduction factor | 0.5 |
| `ignore_index` | int | NoData value | null |

---

## 9. Reproducibility

### 9.1 Environment Specification

**Docker image:** `nvcr.io/nvidia/pytorch:25.11-py3`
- PyTorch 2.5 with CUDA 12.5
- Python 3.11
- GDAL 3.x

**Verify environment:**
```bash
python -c "import torch; import torchgeo; import rasterio; print('All imports successful')"
python -c "import torch; assert torch.cuda.is_available(); print(f'GPU: {torch.cuda.get_device_name(0)}')"
gdalinfo --version
```

### 9.2 Experiment Tracking

**Record for each experiment:**

1. **Git commit hash:**
```bash
git rev-parse HEAD > experiment_commit.txt
```

2. **Complete config:**
```bash
cp configs/osugnn_best.yaml experiment_config.yaml
```

3. **Environment snapshot:**
```bash
pip freeze > experiment_requirements.txt
```

4. **Dataset statistics:**
```bash
cp data/fortypba/train_stats_2021.pt experiment_stats.pt
```

### 9.3 Reproducibility Checklist

Before claiming reproducibility:

- [ ] Docker container with correct image
- [ ] All packages at specified versions
- [ ] Same dataset version
- [ ] Same train/val/test tile boundaries
- [ ] Complete YAML config saved
- [ ] Random seed set (`seed_everything`)
- [ ] Git commit hash recorded
- [ ] Normalization statistics from same data
- [ ] Exact command line documented

---

## 10. Troubleshooting

### Common Issues

#### Issue: CUDA out of memory

**Solutions:**
```bash
# Reduce batch size
torchgeo fit --config config.yaml --data.init_args.batch_size 16

# Use gradient accumulation
torchgeo fit --config config.yaml --trainer.accumulate_grad_batches 2

# Use smaller tile size
# Update config: size: [112, 112] instead of [256, 256]
```

#### Issue: "Normalize transform was not populated with mean"

**Cause:** Stats file not found or path incorrect.

**Solution:**
```yaml
data:
  init_args:
  stats_path: "train_stats_2021.pt"  # Relative to root
```

#### Issue: Shape mismatch in Normalize

**Cause:** `SelectBands` after `Normalize` with mismatched `identity_channels`.

**Solution:** Ensure `SelectBands` comes BEFORE `Normalize` in transform chain.

#### Issue: High validation loss

**Cause:** Validation dataset not normalized.

**Solution:** Ensure `post_aug_input_transforms` and `post_aug_target_transforms` are populated with `Normalize` transforms.

#### Issue: Poor performance after resuming checkpoint

**Cause:** Checkpoint loaded but config differs.

**Solution:**
- Verify checkpoint path exists
- Use same base config for HPO and resume
- Check hyperparameters match the trial

#### Issue: NoData values causing massive loss scores

**Cause:** NoData values being normalized.

**Solution:**
- Use robust masking: `(target == ignore_index) | (target < -1e9)`
- Ensure `Normalize` transform has proper NoData handling
- Verify `ignore_index` is set correctly in model config

---

## 11. Appendix

### 11.1 File Reference

| Script/Path | Purpose |
|-------------|---------|
| `scripts/sample_tiles.py` | Generate balanced tile splits |
| `scripts/prepare_data.py` | Download data from GEE |
| `scripts/optuna_hpo.py` | Single-objective HPO |
| `scripts/optuna_hpo_focal.py` | Focal loss HPO |
| `lightning_logs/` | TensorBoard logs |
| `optuna_logs/` | HPO trial logs |

### 11.2 Metrics Reference

**Classification Metrics:**
- `seg_*_accuracy`: Overall pixel accuracy
- `seg_*_kappa`: Cohen's Kappa (agreement accounting for chance)
- `seg_*_jaccard`: Jaccard Index (IoU for multi-class)

**Regression Metrics:**
- `reg_*_rmse`: Root Mean Squared Error
- `reg_*_mae`: Mean Absolute Error
- `reg_*_r2`: Coefficient of determination (R²)

**Loss Logging:**
- `train_loss` / `val_loss`: Total weighted loss
- `train_task_*_raw_loss`: Raw loss per task
- `train_task_*_weight` / `*_precision`: Task weight
- `train_task_*_log_var`: Uncertainty parameter

### 11.3 Quick Command Reference

```bash
# Environment
docker-compose up -d forestvision
docker-compose exec forestvision bash
source .env

# Data preparation
python scripts/sample_tiles.py --output-path data/fortypba/ --sample-size 8000
python scripts/prepare_data.py --config configs/osugnn_best.yaml

# Training
torchgeo fit --config configs/osugnn_best.yaml
torchgeo fit --config configs/osugnn_best.yaml --trainer.fast_dev_run true
torchgeo fit --config configs/osugnn_best.yaml --ckpt_path checkpoints/last.ckpt

# HPO
python scripts/optuna_hpo.py --config configs/optuna_base.yaml --n-trials 50
python scripts/optuna_hpo_multi.py --config configs/optuna_base.yaml --n-trials 100

# Inference
python scripts/demo_predict.py --checkpoint checkpoints/best.ckpt --output-dir predictions/

# Monitoring
tensorboard --logdir lightning_logs/
nvidia-smi
```

---

*Last updated: 2026-03-07*
