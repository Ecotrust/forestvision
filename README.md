# ForestVision

<div align="center">

<!-- Badges Row -->
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.9](https://img.shields.io/badge/PyTorch-2.9.0-EE4C2C.svg)](https://pytorch.org/)
[![CUDA 13.0](https://img.shields.io/badge/CUDA-13.0-76B900.svg)](https://developer.nvidia.com/cuda-downloads)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker Ready](https://img.shields.io/badge/Docker-ready-2496ED.svg)](docker-compose.yaml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**PyTorch framework for forest monitoring and geospatial deep learning**

</div>

---

ForestVision extends [TorchGeo](https://github.com/microsoft/torchgeo) to enable cloud-native forest analysis using multi-spectral satellite imagery, terrain data, and modern deep learning. It provides a complete pipeline for predicting forest community composition (classification) and structural characteristics (regression) from remote sensing data.

## Project Structure

```
forestvision/
├── forestvision/          # Core source code
│   ├── datamodules/       # DataModule implementations
│   ├── datasets/          # GeoDataset classes
│   ├── models/            # Neural network architectures
│   ├── trainers/          # LightningModule trainers
│   └── transforms/        # Data augmentation & preprocessing
├── configs/               # YAML configuration files
├── data/                  # Data storage (not version controlled)
├── docs/                  # Documentation (you are here!)
├── scripts/               # Utility scripts
├── notebooks/             # Jupyter notebooks for analysis
└── tests/                 # Unit tests
```

## Features

| Feature | Description |
|---------|-------------|
| **Cloud-Native Data** | Seamless Google Earth Engine integration with automatic downloading and caching |
| **Multi-Task Learning** | Joint classification and regression with shared encoders and task-specific decoders |
| **Transfer Learning** | ImageNet-pretrained ResNet backbones (18/34/50/101) for faster convergence |
| **Hyperparameter Optimization** | Built-in Optuna integration for automated HPO |
| **Flexible Data Pipeline** | 5-stage transform pipeline with automatic spatial alignment via IntersectionDataset |
| **Reproducible Research** | Docker-based environment, YAML configurations, and Weights & Biases integration |

## Installation

### Prerequisites

- NVIDIA GPU with appropriate drivers (tested on GB10/Blackwell and NVIDIA RTX A4000 with CUDA 12+)
- Google Earth Engine (GEE) credentials with a configured project
- Git for version control
- Docker installed with NVIDIA Container Toolkit (optional)

### Option 1: Local Installation

```bash
# Clone repository
git clone https://github.com/Ecotrust/forestvision.git
cd forestvision

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Option 2: Docker 

```bash
# Clone and start
git clone https://github.com/Ecotrust/forestvision.git
cd forestvision
docker-compose up -d forestvision
```

### Verify GPU Access

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

Expected output:
```
CUDA available: True
GPU count: 1
```

### Configure Environment Variables

Copy and edit the environment file:

```bash
cp .env.example .env
# Edit .env with your GEE project ID and credentials
```

Required variables:
```bash
GEE_PROJECT_ID=your-project-id
```

## Documentation

- **[Sample Tiles](docs/sample_tiles.md)** - Tile generation for training and inference
- **[Prepare Data](docs/prepare_data.md)** - Data preparation and preprocessing steps
- **[Training Pipeline](docs/training_pipeline.md)** - Model training and evaluation
- **[Inference Pipeline](docs/inference_pipeline.md)** - Running predictions on new data
- **[Create Mosaic](docs/create_mosaic.md)** - Post-processing and mosaic creation
- **[Datasets](docs/datasets.md)** - Overview of supported datasets

## Examples

### Training a Multi-Task Model

```bash
# Prepare dataset with target channel 0 set to identity stats (mean=0, std=1)
python scripts/prepare_data.py --config path/to/focal_best.yaml -tic 0

# Train with default configuration
torchgeo fit --config path/to/focal_best.yaml --trainer.max_epochs 100

# Train with custom hyperparameters
torchgeo fit \
    --config focal_best.yaml \
    --trainer.max_epochs 50 \
    --data.batch_size 16
```

### Running Hyperparameter Optimization

```bash
# Multi-objective HPO (accuracy vs. inference time)
python scripts/optuna_hpo_multi.py --config path/to/focal_best.yaml --n-trials 100

# Single-objective HPO for focal loss parameters
python scripts/optuna_hpo_focal.py --config path/to/focal_best.yaml --n-trials 50
```

### Inference on New Data

```bash
# Run prediction pipeline
torchgeo predict --config path/to/predict_best.yaml --ckpt_path path/to/best_model.ckpt
```

## Citation

If you use ForestVision in your research, please cite:

```bibtex
@software{forestvision2026,
  author = {Ecotrust},
  title = {ForestVision: A PyTorch framework for forest monitoring and geospatial deep learning},
  publisher = {Ecotrust},
  url = {https://github.com/Ecotrust/forestvision},
  version = {0.1.0},
  year = {2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributors

- [Yankuic Galvan](mailto:yankuic@gmail.com) - Lead Developer

## Acknowledgments

- Built on [TorchGeo](https://github.com/microsoft/torchgeo) by Microsoft
- Uses [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://lightning.ai/)
- Geospatial processing powered by [GDAL](https://gdal.org/) and [Rasterio](https://rasterio.readthedocs.io/)

