# Create Mosaic Script Documentation

## Overview

`scripts/create_mosaic.py` generates seamless mosaics from overlapping prediction tiles using GDAL. It provides multiple methods for merging tiles including simple gdalwarp merging, feather blending for smooth transitions, and various clipping options for boundary-constrained outputs.

This script is the final step in the ForestVision inference pipeline, following model prediction on individual tiles.

## Workflow

### 1. Tile Discovery and Validation

The script discovers prediction tiles based on task name patterns:

**Tile Naming Convention**: Tiles must follow the pattern `{id}_{task}_MultiTaskUNet.tif`
- `id`: Unique tile identifier (e.g., hash or geohash)
- `task`: Task name (cancov, qmd_dom, ba_ge_3, fortypba)
- Example: `038aee7b_cancov_MultiTaskUNet.tif`

**Validation Checks**:
- All tiles must have consistent CRS
- Tiles must be readable by GDAL/rasterio
- Single-task validation (no mixing tasks in one mosaic)

### 2. Mosaic Creation Methods

#### Method: gdalwarp (Default)

Uses GDAL's `gdalwarp` command with overlap handling:

```bash
gdalwarp -of COG -r average -co COMPRESS=DEFLATE ... input1.tif input2.tif output.tif
```

**Resampling by Task Type**:
- **Regression tasks** (cancov, qmd_dom, ba_ge_3): Uses `average` resampling for smooth transitions
- **Segmentation** (fortypba): Uses `mode` resampling to preserve discrete class values

**Advantages**:
- Fastest method for most use cases
- Handles large tile counts efficiently
- Automatic overlap blending via averaging

#### Method: feather_blend

Implements custom alpha blending with raised cosine weight profiles:

**Weight Profile (Raised Cosine)**:
```
Weight at edge:     0.1 (minimum to prevent gaps)
Weight at center:   1.0 (full contribution)
Transition:         0.5 * (1 - cos(π * t))  where t ∈ [0,1]
```

**Neighbor Detection**:
- Automatically detects which tile edges overlap with neighbors
- Only applies fading where neighbors exist
- Prevents unnecessary blending at mosaic boundaries

**Blend Distance**: Configurable via `--blend-distance` (default: 15 pixels)
- Should be approximately half the tile overlap
- For 30px overlap, use `--blend-distance 15`

#### Method: python

Uses rasterio's merge function for Python-native merging:

```python
from rasterio.merge import merge
mosaic, transform = merge(src_files)
```

**Use Cases**:
- When GDAL tools are unavailable
- For custom blending logic integration
- Debugging and development

**Limitations**:
- No built-in feather blending
- May have memory issues with large mosaics
- Slower than gdalwarp for large datasets

### 3. Clipping to Boundary

Optional post-processing step to clip mosaic to a GeoJSON or Shapefile boundary:

**Clipping Command**:
```bash
gdalwarp -cutline boundary.geojson -crop_to_cutline -s_srs EPSG:5070 ...
```

**CRS Handling**:
- Automatically reads CRS from mosaic
- Reprojects boundary if necessary
- Preserves NoData outside boundary

## Configuration

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--task` | str | Required | Task to mosaic (cancov, qmd_dom, ba_ge_3, fortypba, all) |
| `--input-dir` | Path | data/inference/predictions | Directory containing prediction tiles |
| `--output-dir` | Path | data/inference/mosaics | Directory for output mosaics |
| `--method` | str | gdalwarp | Mosaic method (gdalwarp, buildvrt, python) |
| `--blend-distance` | int | None | Feather blending distance in pixels (requires method=gdalwarp) |
| `--clip` | Path | None | GeoJSON/Shapefile boundary for clipping |
| `--crs` | str | None | CRS of tiles when metadata missing (e.g., EPSG:5070) |
| `--compression` | str | DEFLATE | Compression (DEFLATE, LZW, ZSTD, NONE) |
| `--overwrite` | flag | False | Overwrite existing output |
| `--verbose` | flag | False | Enable debug logging |

### Default Tasks

The script supports these MultiTaskUNet outputs:

| Task | Type | Description |
|------|------|-------------|
| `cancov` | Regression | Canopy cover percentage |
| `qmd_dom` | Regression | Quadratic mean diameter |
| `ba_ge_3` | Regression | Basal area >= 3 inches |
| `fortypba` | Segmentation | Forest type segmentation |

## Usage Examples

### Basic Mosaic Creation

```bash
# Simple merge for canopy cover
python scripts/create_mosaic.py --task cancov

# Merge all tasks
python scripts/create_mosaic.py --task all

# With custom output directory
python scripts/create_mosaic.py --task cancov --output-dir /path/to/output
```

### Feather Blending

```bash
# Smooth transitions with 15px blend distance
python scripts/create_mosaic.py --task cancov --blend-distance 15

# For 30px overlap tiles, use half as blend distance
python scripts/create_mosaic.py --task qmd_dom --blend-distance 15 --overwrite
```

### Clipping to Boundary

```bash
# Clip to GeoJSON boundary (requires CRS when tiles lack metadata)
python scripts/create_mosaic.py \
  --task cancov \
  --crs EPSG:5070 \
  --clip data/boundaries/region.geojson \
  --overwrite

# Feather blending with clipping
python scripts/create_mosaic.py \
  --task cancov \
  --blend-distance 15 \
  --crs EPSG:5070 \
  --clip data/boundaries/aoi.geojson \
  --overwrite
```

### Alternative Methods

```bash
# Using VRT build (for many tiles)
python scripts/create_mosaic.py --task all --method buildvrt

# Python merge (rasterio-based)
python scripts/create_mosaic.py --task cancov --method python
```

### Compression Options

```bash
# ZSTD compression (faster read, larger file)
python scripts/create_mosaic.py --task cancov --compression ZSTD

# LZW compression (smaller file, slower)
python scripts/create_mosaic.py --task cancov --compression LZW
```

## Output Files

### Directory Structure

```
{output_dir}/
├── {task}_mosaic.tif          # Cloud-Optimized GeoTIFF mosaic
└── {task}_mosaic.tif.ovr      # Internal overviews (if COG)
```

### File Format

All outputs are **Cloud-Optimized GeoTIFFs (COG)** with:
- Tiled internal structure (512x512 blocks)
- DEFLATE compression by default
- Internal overviews for fast visualization
- Original CRS preserved

### Mosaic Properties

| Property | Value |
|----------|-------|
| Format | COG (Cloud-Optimized GeoTIFF) |
| Compression | DEFLATE (configurable) |
| Tiling | 512x512 pixels |
| Overviews | Automatic (nearest for segmentation, bilinear for regression) |
| NoData | Preserved from input tiles |
| Data Type | Float32 (regression), UInt8 (segmentation) |

## Algorithm Details

### Feather Blending Math

For overlapping regions, the output value is computed as:

```
output = Σ(tile_i × weight_i) / Σ(weight_i)
```

**Weight Function** (raised cosine):
```python
t = distance_from_edge / blend_distance
weight = min_weight + (1 - min_weight) × 0.5 × (1 - cos(π × t))
```

Where:
- `min_weight = 0.1` (ensures all tiles contribute at least 10%)
- `t ∈ [0, 1]` across the blend zone
- Weights fade from 0.1 at tile edges to 1.0 in tile centers

### Neighbor Detection

Tiles are analyzed for spatial overlap to determine blending edges:

```python
x_overlap = min(b1.right, b2.right) - max(b1.left, b2.left)
y_overlap = min(b1.top, b2.top) - max(b1.bottom, b2.bottom)

if x_overlap > threshold and y_overlap > threshold:
    # Tiles overlap - determine relative position
    dx = center2.x - center1.x
    dy = center2.y - center1.y
```

## Troubleshooting

### Common Issues

**"Could not determine CRS from input tiles"**
- Tiles lack CRS metadata
- Solution: Use `--crs EPSG:5070` (or appropriate CRS)

**"Cannot compute bounding box of cutline"**
- CRS mismatch between mosaic and boundary
- Solution: Ensure `--crs` matches your boundary file's CRS

**Stitching Effects Visible**
- Model predictions differ significantly at tile edges
- Solutions:
  - Use `--blend-distance 15` for smoother transitions
  - Increase tile overlap during inference (30-50px)
  - Check model for boundary artifacts

**Large File Sizes**
- Use `--compression ZSTD` for better compression
- Remove unnecessary overviews: `--compression NONE` then `gdaladdo -clean`

## See Also

- [`Inference Pipeline`](inference_pipeline.md) - Full inference documentation
- [`Prepare Data`](prepare_data.md) - Data preparation guide
- [GDAL Warp Documentation](https://gdal.org/programs/gdalwarp.html)
- [COG Specification](https://www.cogeo.org/)
