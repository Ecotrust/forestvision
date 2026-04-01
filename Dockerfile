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

# Set GDAL environment variables for proper configuration
ENV GDAL_DATA=/usr/share/gdal
ENV GDAL_DRIVER_PATH=/usr/lib/gdalplugins
ENV CPL_ZIP_ENCODING=UTF-8

# Set working directory
WORKDIR /workspace

# Install Python packages (explicit list as requested)
# Geo
RUN pip install --no-cache-dir \
    gcloud \
    earthengine-api \
    geopandas \
    rasterio \
    shapely>-2.0 \
    rio_cogeo \
    torchgeo==0.7.0

# DS
RUN pip install --no-cache-dir \
    scikit-learn \
    numpy \
    pandas

# Misc
RUN pip install --no-cache-dir \
    matplotlib \
    tqdm \
    dotenv \
    requests \
    retry \
    pyyaml>=0.6 

# Hyperparameter Optimization
RUN pip install --no-cache-dir \
    "optuna>=3.6.0" \
    "optuna-integration>=3.6.0"

# Configuration
RUN pip install --no-cache-dir \
    "pyyaml>=6.0"

USER ubuntu

# Default command
CMD ["/bin/bash"]
