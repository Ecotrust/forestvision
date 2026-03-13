# ForestVision Dockerfile
# Based on NVIDIA PyTorch container with GDAL and geospatial dependencies

FROM nvcr.io/nvidia/pytorch:25.11-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install GDAL system dependencies
RUN apt-get update && apt-get install -y \
    gdal-bin \
    python3-gdal \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Install Python packages (explicit list as requested)
# Geo
RUN pip install --no-cache-dir \
    earthengine-api \
    geopandas \
    rasterio

# DS
RUN pip install --no-cache-dir \
    scikit-learn \
    numpy \
    pandas

# Misc
RUN pip install --no-cache-dir \
    matplotlib \
    tqdm \
    dotenv

# Hyperparameter Optimization
RUN pip install --no-cache-dir \
    "optuna>=3.6.0" \
    "optuna-integration>=3.6.0"

# Configuration
RUN pip install --no-cache-dir \
    "pyyaml>=6.0"

# Default command
CMD ["/bin/bash"]
