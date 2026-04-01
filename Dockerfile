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
# These ensure rasterio, geopandas, and other geospatial libraries can find GDAL data
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

# Create non-root user matching host UID/GID
# Build with: docker build --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) -t forestvision:latest .
ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create user and set permissions
# Handle case where UID/GID already exists in base image
RUN if ! getent group $USER_GID > /dev/null 2>&1; then \
        groupadd --gid $USER_GID $USERNAME; \
    else \
        existing_group=$(getent group $USER_GID | cut -d: -f1); \
        echo "GID $USER_GID already exists as group $existing_group"; \
    fi && \
    if ! id -u $USER_UID > /dev/null 2>&1; then \
        useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
        echo "$USERNAME" > /tmp/container_user; \
    else \
        existing_user=$(id -un $USER_UID); \
        echo "UID $USER_UID already exists as user $existing_user"; \
        usermod -aG $USER_GID $existing_user 2>/dev/null || true; \
        echo "$existing_user" > /tmp/container_user; \
    fi && \
    mkdir -p /workspace && \
    chown -R $USER_UID:$USER_GID /workspace

# Use the actual user (developer or existing from base image)
USER $USER_UID

# Default command
CMD ["/bin/bash"]
