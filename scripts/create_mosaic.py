"""Create seamless mosaics from prediction tiles using GDAL.

This script generates mosaics from overlapping prediction tiles without visible
stitching effects. It uses GDAL's advanced warping capabilities to blend
overlapping regions smoothly.

Usage:
    # Simple merge with average resampling (fast, recommended)
    python scripts/create_mosaic.py --task cancov

    # With feather blending (smoothest results, slower)
    python scripts/create_mosaic.py --task cancov --blend-distance 15

    # All tasks at once
    python scripts/create_mosaic.py --task all

Author: ForestVision
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import rasterio
from osgeo import gdal


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default task names from MultiTaskUNet
DEFAULT_TASKS = ["cancov", "qmd_dom", "ba_ge_3", "fortypba"]


def get_tiles_for_task(input_dir: Path, task_name: str) -> List[Path]:
    """Get all tile files for a specific task.

    Args:
        input_dir: Directory containing prediction tiles
        task_name: Name of the task (e.g., 'cancov', 'qmd_dom')

    Returns:
        List of Path objects for matching tiles
    """
    pattern = f"*_{task_name}_MultiTaskUNet.tif"
    tiles = sorted(input_dir.glob(pattern))
    return tiles


def validate_tiles(tiles: List[Path]) -> tuple[bool, Optional[str]]:
    """Validate that tiles exist and have consistent CRS.

    Args:
        tiles: List of tile paths to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not tiles:
        return False, "No tiles found"

    if len(tiles) == 1:
        return True, None

    # Check CRS consistency
    reference_crs = None
    for tile in tiles:
        try:
            ds = gdal.Open(str(tile))
            if ds is None:
                return False, f"Could not open {tile}"

            crs = ds.GetProjection()
            ds = None  # Close dataset

            if reference_crs is None:
                reference_crs = crs
            elif crs != reference_crs:
                return False, f"CRS mismatch in {tile}"

        except Exception as e:
            return False, f"Error reading {tile}: {e}"

    return True, None


def clip_mosaic_with_boundary(
    input_path: Path,
    output_path: Path,
    boundary_path: Path,
    compression: str = "DEFLATE",
) -> bool:
    """Clip a mosaic to a boundary using gdalwarp.

    Args:
        input_path: Path to input mosaic
        output_path: Path for clipped output
        boundary_path: Path to boundary GeoJSON/Shapefile
        compression: Compression algorithm

    Returns:
        True if successful, False otherwise
    """
    if not boundary_path.exists():
        logger.error(f"Boundary file not found: {boundary_path}")
        return False

    logger.info(f"Clipping mosaic to boundary: {boundary_path}")

    # Read input CRS to specify for clipping
    with rasterio.open(input_path) as src:
        src_crs = src.crs
        if src_crs is not None:
            src_crs_str = src_crs.to_string()
        else:
            # Try to get CRS from GDAL if rasterio doesn't have it
            ds = gdal.Open(str(input_path))
            if ds:
                wkt = ds.GetProjection()
                ds = None
                if wkt:
                    src_crs_str = wkt
                else:
                    logger.error("Could not determine CRS of input mosaic")
                    return False
            else:
                logger.error("Could not open input mosaic with GDAL")
                return False

    cmd = [
        "gdalwarp",
        "-of", "COG",
        "-co", f"COMPRESS={compression}",
        "-co", "BIGTIFF=YES",
        "-cutline", str(boundary_path),
        "-crop_to_cutline",
        "-s_srs", src_crs_str,  # Source CRS of the mosaic
        "-t_srs", src_crs_str,  # Target CRS (same as source)
        "-overwrite",
        str(input_path),
        str(output_path),
    ]

    logger.debug(f"Command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Successfully clipped mosaic: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"gdalwarp clip failed: {e}")
        logger.error(f"stderr: {e.stderr}")
        return False


def create_mosaic_gdalwarp(
    tiles: List[Path],
    output_path: Path,
    blend_distance: Optional[int] = None,
    compression: str = "DEFLATE",
    resampling: str = "bilinear",
    is_classification: bool = False,
    clip_boundary: Optional[Path] = None,
) -> bool:
    """Create mosaic using gdalwarp with optional blending.

    This is the primary method - uses gdalwarp which handles overlapping
    tiles intelligently. For overlapping areas, it uses averaging by default
    which provides smooth transitions between tiles.

    Args:
        tiles: List of input tile paths
        output_path: Path for output mosaic
        blend_distance: Pixel distance for feather blending (experimental)
        compression: Compression algorithm (DEFLATE, LZW, ZSTD)
        resampling: Resampling method for overview generation
        is_classification: Whether this is classification data (uses mode resampling)
        clip_boundary: Optional path to GeoJSON/Shapefile for clipping

    Returns:
        True if successful, False otherwise
    """
    if not tiles:
        logger.error("No tiles provided for mosaic creation")
        return False

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build gdalwarp command
    cmd = [
        "gdalwarp",
        "-of", "COG",  # Cloud-Optimized GeoTIFF
        "-co", f"COMPRESS={compression}",
        "-co", "BIGTIFF=YES",  # Support large outputs
    ]

    # Add overview resampling method
    overview_resampling = "nearest" if is_classification else resampling
    cmd.extend(["-co", f"RESAMPLING={overview_resampling}"])

    # For overlapping areas, use appropriate resampling
    # For regression: average provides smooth transitions
    # For classification: mode keeps discrete values
    overlap_resampling = "mode" if is_classification else "average"
    cmd.extend(["-r", overlap_resampling])

    logger.info(f"Using {overlap_resampling} resampling for overlapping areas")

    # Add clipping if specified
    if clip_boundary and clip_boundary.exists():
        cmd.extend(["-cutline", str(clip_boundary), "-crop_to_cutline"])
        logger.info(f"Clipping to boundary: {clip_boundary}")

    # Add overwrite and input files (output goes last for gdalwarp)
    cmd.append("-overwrite")
    cmd.extend([str(t) for t in tiles])
    cmd.append(str(output_path))

    logger.info(f"Running gdalwarp with {len(tiles)} tiles...")
    logger.debug(f"Command: {' '.join(cmd)}")

    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(f"Successfully created mosaic: {output_path}")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"gdalwarp failed: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        return False


def create_mosaic_vrt_buildvrt(
    tiles: List[Path],
    output_path: Path,
    compression: str = "DEFLATE",
) -> bool:
    """Alternative: Create mosaic using gdalbuildvrt + gdal_translate.

    This method is useful when gdalwarp has issues with many tiles.
    It builds a VRT first, then translates to COG.

    Args:
        tiles: List of input tile paths
        output_path: Path for output mosaic
        compression: Compression algorithm

    Returns:
        True if successful, False otherwise
    """
    if not tiles:
        logger.error("No tiles provided for mosaic creation")
        return False

    vrt_path = output_path.with_suffix(".vrt")

    # Step 1: Build VRT
    buildvrt_cmd = [
        "gdalbuildvrt",
        "-overwrite",
        str(vrt_path),
    ] + [str(t) for t in tiles]

    logger.info(f"Building VRT: {' '.join(buildvrt_cmd)}")

    try:
        subprocess.run(buildvrt_cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"gdalbuildvrt failed: {e}")
        return False

    # Step 2: Translate VRT to COG
    translate_cmd = [
        "gdal_translate",
        "-of", "COG",
        "-co", f"COMPRESS={compression}",
        "-co", "BIGTIFF=YES",
        str(vrt_path),
        str(output_path),
    ]

    logger.info(f"Translating to COG: {' '.join(translate_cmd)}")

    try:
        subprocess.run(translate_cmd, capture_output=True, text=True, check=True)
        logger.info(f"Successfully created mosaic: {output_path}")

        # Clean up VRT
        vrt_path.unlink(missing_ok=True)
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"gdal_translate failed: {e}")
        return False


def create_blend_weights(height: int, width: int, blend_distance: int, 
                         neighbor_info: dict = None) -> np.ndarray:
    """Create blend weights for smooth tile merging using true alpha blending.
    
    Creates complementary fade patterns where overlapping tiles transition smoothly.
    Uses raised cosine for smoother transitions than linear.
    
    Args:
        height: Height of the tile
        width: Width of the tile  
        blend_distance: Number of pixels from edge to blend (should be <= overlap/2)
        neighbor_info: Dict with 'top', 'bottom', 'left', 'right' bools indicating neighbors

    Returns:
        2D numpy array of weights (0.1 to 1.0) - minimum 0.1 prevents gaps
    """
    weights = np.ones((height, width), dtype=np.float32)
    
    if blend_distance <= 0:
        return weights
    
    # Clamp blend_distance
    blend_distance = min(blend_distance, height // 3, width // 3)
    
    if neighbor_info is None:
        neighbor_info = {'top': True, 'bottom': True, 'left': True, 'right': True}
    
    # Minimum weight to prevent gaps (0.1 means tile always contributes at least 10%)
    min_weight = 0.1
    
    # For true alpha blending, we fade OUT toward edges that have neighbors
    # Use raised cosine for smoother transition: 0.5 * (1 + cos(pi * t)) where t goes 0->1
    
    # Top edge: fade from 1.0 to min_weight if there's a neighbor above
    if neighbor_info.get('top', False):
        for i in range(blend_distance):
            t = i / blend_distance  # 0 to 1
            # Raised cosine: smooth transition from min_weight to 1.0
            weight = min_weight + (1.0 - min_weight) * 0.5 * (1 - np.cos(np.pi * t))
            weights[i, :] = weight
    
    # Bottom edge: fade from 1.0 to min_weight if there's a neighbor below  
    if neighbor_info.get('bottom', False):
        for i in range(blend_distance):
            t = i / blend_distance
            weight = min_weight + (1.0 - min_weight) * 0.5 * (1 - np.cos(np.pi * t))
            weights[-(i+1), :] = weight
    
    # Left edge: fade from 1.0 to min_weight if there's a neighbor to the left
    if neighbor_info.get('left', False):
        for j in range(blend_distance):
            t = j / blend_distance
            weight = min_weight + (1.0 - min_weight) * 0.5 * (1 - np.cos(np.pi * t))
            weights[:, j] = np.minimum(weights[:, j], weight)
    
    # Right edge: fade from 1.0 to min_weight if there's a neighbor to the right
    if neighbor_info.get('right', False):
        for j in range(blend_distance):
            t = j / blend_distance
            weight = min_weight + (1.0 - min_weight) * 0.5 * (1 - np.cos(np.pi * t))
            weights[:, -(j+1)] = np.minimum(weights[:, -(j+1)], weight)
    
    return weights


def detect_tile_neighbors(src_files: list, idx: int, all_bounds: list, overlap_threshold: float = 20.0) -> dict:
    """Detect which edges of a tile overlap with neighboring tiles.
    
    Args:
        src_files: List of open rasterio datasets
        idx: Index of current tile
        all_bounds: List of bounds for all tiles
        overlap_threshold: Minimum overlap in pixels to consider as neighbor
        
    Returns:
        Dict with 'top', 'bottom', 'left', 'right' boolean flags
    """
    current_bounds = all_bounds[idx]
    res = src_files[idx].res[0]  # pixel size
    
    neighbors = {'top': False, 'bottom': False, 'left': False, 'right': False}
    
    for other_idx, other_bounds in enumerate(all_bounds):
        if other_idx == idx:
            continue
            
        # Check for overlap
        x_overlap = min(current_bounds.right, other_bounds.right) - max(current_bounds.left, other_bounds.left)
        y_overlap = min(current_bounds.top, other_bounds.top) - max(current_bounds.bottom, other_bounds.bottom)
        
        if x_overlap > overlap_threshold * res and y_overlap > overlap_threshold * res:
            # Determine relative position
            current_cx = (current_bounds.left + current_bounds.right) / 2
            current_cy = (current_bounds.top + current_bounds.bottom) / 2
            other_cx = (other_bounds.left + other_bounds.right) / 2
            other_cy = (other_bounds.top + other_bounds.bottom) / 2
            
            dx = other_cx - current_cx
            dy = other_cy - current_cy
            
            # Determine which edge the neighbor is on
            if abs(dx) > abs(dy):  # Horizontal neighbor
                if dx > 0:
                    neighbors['right'] = True
                else:
                    neighbors['left'] = True
            else:  # Vertical neighbor
                if dy > 0:
                    neighbors['top'] = True  # In image coords, y increases downward
                else:
                    neighbors['bottom'] = True
    
    return neighbors


def create_mosaic_feather_blend(
    tiles: List[Path],
    output_path: Path,
    blend_distance: int = 15,
    compression: str = "DEFLATE",
    is_classification: bool = False,
    crs_override: Optional[str] = None,
) -> bool:
    """Create mosaic using feather blending for smooth transitions.

    This method provides the smoothest results by using distance-weighted
    blending in overlap zones. It's slower than gdalwarp but produces
    better results when tiles have variations at edges.

    Args:
        tiles: List of input tile paths
        output_path: Path for output mosaic
        blend_distance: Pixel distance for feather blending
        compression: Compression algorithm
        is_classification: Whether this is classification data
        crs_override: CRS to use if tiles lack metadata (e.g., "EPSG:5070")

    Returns:
        True if successful, False otherwise
    """
    if not tiles:
        logger.error("No tiles provided for mosaic creation")
        return False

    if len(tiles) == 1:
        # Single tile, just copy it
        logger.info("Only one tile, copying directly")
        import shutil
        shutil.copy(tiles[0], output_path)
        return True

    logger.info(f"Creating feather-blended mosaic with {blend_distance}px blend distance")
    logger.info(f"Processing {len(tiles)} tiles...")

    try:
        # Open all tiles
        src_files = [rasterio.open(t) for t in tiles]

        # Get common CRS and check bounds
        crs = src_files[0].crs
        nodata = src_files[0].nodata
        
        # Use override CRS if tiles lack metadata
        if crs is None and crs_override:
            from rasterio.crs import CRS
            crs = CRS.from_string(crs_override)
            logger.info(f"Using specified CRS: {crs_override}")
        
        # Ensure CRS is properly set
        if crs is None:
            logger.error("Could not determine CRS from input tiles")
            logger.error("Tiles may be missing CRS metadata. Use --crs option (e.g., --crs EPSG:5070)")
            return False

        # Calculate total bounds
        all_bounds = [src.bounds for src in src_files]
        minx = min(b.left for b in all_bounds)
        miny = min(b.bottom for b in all_bounds)
        maxx = max(b.right for b in all_bounds)
        maxy = max(b.top for b in all_bounds)

        # Calculate output dimensions
        res = src_files[0].res[0]  # Assume square pixels
        out_width = int((maxx - minx) / res)
        out_height = int((maxy - miny) / res)

        logger.info(f"Output mosaic size: {out_width}x{out_height} pixels")
        logger.info(f"Output bounds: ({minx}, {miny}, {maxx}, {maxy})")

        # Detect neighbors for all tiles first
        logger.info("Detecting tile neighbors...")
        tile_neighbors = []
        for idx in range(len(src_files)):
            neighbors = detect_tile_neighbors(src_files, idx, all_bounds, overlap_threshold=20.0)
            tile_neighbors.append(neighbors)
            logger.debug(f"Tile {idx}: neighbors={neighbors}")

        # Create output arrays
        output_data = np.zeros((out_height, out_width), dtype=np.float64)
        output_weights = np.zeros((out_height, out_width), dtype=np.float64)

        # Process each tile
        for idx, src in enumerate(src_files):
            logger.info(f"Processing tile {idx+1}/{len(tiles)}: {tiles[idx].name}")

            # Read tile data
            tile_data = src.read(1)
            tile_height, tile_width = tile_data.shape

            # Create blend weights for this tile with neighbor info
            weights = create_blend_weights(tile_height, tile_width, blend_distance, tile_neighbors[idx])

            # Calculate position in output
            tile_bounds = src.bounds
            col_start = int((tile_bounds.left - minx) / res)
            row_start = int((maxy - tile_bounds.top) / res)

            # Handle nodata
            if nodata is not None:
                valid_mask = tile_data != nodata
                tile_data = np.where(valid_mask, tile_data, 0)
                weights = np.where(valid_mask, weights, 0)

            # Add to output (accumulate weighted sum)
            row_end = row_start + tile_height
            col_end = col_start + tile_width

            output_data[row_start:row_end, col_start:col_end] += tile_data * weights
            output_weights[row_start:row_end, col_start:col_end] += weights

        # Normalize by weights
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            output_data = np.where(output_weights > 0, output_data / output_weights, nodata if nodata else 0)

        # Convert to appropriate dtype
        if is_classification:
            output_data = np.round(output_data).astype(np.uint8)
            dtype = rasterio.uint8
        else:
            output_data = output_data.astype(np.float32)
            dtype = rasterio.float32

        # Create output transform
        out_transform = rasterio.Affine.translation(minx, maxy) * rasterio.Affine.scale(res, -res)

        # Write output
        out_meta = {
            "driver": "GTiff",
            "height": out_height,
            "width": out_width,
            "count": 1,
            "dtype": dtype,
            "crs": crs,
            "transform": out_transform,
            "nodata": nodata,
            "compress": compression.lower(),
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
        }

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(output_data, 1)

        # Close sources
        for src in src_files:
            src.close()

        logger.info(f"Successfully created feather-blended mosaic: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Feather blending failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


def create_mosaic_python(
    tiles: List[Path],
    output_path: Path,
    blend_distance: Optional[int] = None,
    compression: str = "DEFLATE",
) -> bool:
    """Create mosaic using rasterio (Python-native approach).

    This method gives more control but may be slower for large datasets.
    Useful for custom blending logic.

    Args:
        tiles: List of input tile paths
        output_path: Path for output mosaic
        blend_distance: Pixel distance for blending (not implemented yet)
        compression: Compression algorithm

    Returns:
        True if successful, False otherwise
    """
    try:
        import rasterio
        from rasterio.merge import merge

        logger.info(f"Merging {len(tiles)} tiles using rasterio...")

        # Open all tiles
        src_files = [rasterio.open(t) for t in tiles]

        # Merge
        mosaic, out_transform = merge(src_files)

        # Get metadata from first file
        out_meta = src_files[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_transform,
            "compress": compression.lower(),
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
        })

        # Write output
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(mosaic)

        # Close sources
        for src in src_files:
            src.close()

        logger.info(f"Successfully created mosaic: {output_path}")
        return True

    except ImportError:
        logger.error("rasterio not available for Python merge method")
        return False
    except Exception as e:
        logger.error(f"Python merge failed: {e}")
        return False


def create_mosaic_for_task(
    input_dir: Path,
    output_dir: Path,
    task_name: str,
    method: str = "gdalwarp",
    blend_distance: Optional[int] = None,
    compression: str = "DEFLATE",
    overwrite: bool = False,
    clip_boundary: Optional[Path] = None,
    crs_override: Optional[str] = None,
) -> bool:
    """Create mosaic for a single task.

    Args:
        input_dir: Directory containing prediction tiles
        output_dir: Directory for output mosaic
        task_name: Name of the task
        method: Mosaic method (gdalwarp, buildvrt, python)
        blend_distance: Feather blending distance in pixels
        compression: Compression algorithm
        overwrite: Overwrite existing output
        clip_boundary: Optional path to GeoJSON/Shapefile for clipping
        crs_override: CRS to use if tiles lack metadata (e.g., "EPSG:5070")

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing task: {task_name}")
    logger.info(f"{'='*60}")

    # Get tiles for this task
    tiles = get_tiles_for_task(input_dir, task_name)
    logger.info(f"Found {len(tiles)} tiles for {task_name}")

    if not tiles:
        logger.warning(f"No tiles found for task '{task_name}'")
        return False

    # Validate tiles
    is_valid, error_msg = validate_tiles(tiles)
    if not is_valid:
        logger.error(f"Validation failed: {error_msg}")
        return False

    logger.info("Tile validation passed")

    # Determine output path
    output_path = output_dir / f"{task_name}_mosaic.tif"

    # Check if output exists
    if output_path.exists() and not overwrite:
        logger.warning(f"Output exists (use --overwrite to replace): {output_path}")
        return False

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine if this is a classification task
    is_classification = task_name == "fortypba"

    # Use feather blending if blend_distance is specified
    if blend_distance is not None and blend_distance > 0:
        logger.info(f"Using feather blending with {blend_distance}px distance")
        success = create_mosaic_feather_blend(
            tiles, output_path, blend_distance, compression,
            is_classification=is_classification,
            crs_override=crs_override
        )
        # Clip after blending if boundary specified
        if success and clip_boundary and clip_boundary.exists():
            temp_path = output_path.with_suffix('.temp.tif')
            output_path.rename(temp_path)
            success = clip_mosaic_with_boundary(temp_path, output_path, clip_boundary, compression)
            temp_path.unlink(missing_ok=True)
        return success

    # Create mosaic based on method
    if method == "gdalwarp":
        return create_mosaic_gdalwarp(
            tiles, output_path, blend_distance, compression,
            is_classification=is_classification,
            clip_boundary=clip_boundary
        )
    elif method == "buildvrt":
        success = create_mosaic_vrt_buildvrt(tiles, output_path, compression)
        if success and clip_boundary and clip_boundary.exists():
            temp_path = output_path.with_suffix('.temp.tif')
            output_path.rename(temp_path)
            success = clip_mosaic_with_boundary(temp_path, output_path, clip_boundary, compression)
            temp_path.unlink(missing_ok=True)
        return success
    elif method == "python":
        success = create_mosaic_python(tiles, output_path, blend_distance, compression)
        if success and clip_boundary and clip_boundary.exists():
            temp_path = output_path.with_suffix('.temp.tif')
            output_path.rename(temp_path)
            success = clip_mosaic_with_boundary(temp_path, output_path, clip_boundary, compression)
            temp_path.unlink(missing_ok=True)
        return success
    else:
        logger.error(f"Unknown method: {method}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Create seamless mosaics from prediction tiles using GDAL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Simple merge for a single task
    python scripts/create_mosaic.py --task cancov

    # With feather blending (smooth edges)
    python scripts/create_mosaic.py --task cancov --blend-distance 15

    # All tasks at once
    python scripts/create_mosaic.py --task all

    # Using alternative method (for many tiles)
    python scripts/create_mosaic.py --task all --method buildvrt
        """,
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/inference/predictions"),
        help="Directory containing prediction tiles (default: data/inference/predictions)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/inference/mosaics"),
        help="Directory for output mosaics (default: data/inference/mosaics)",
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task name to mosaic (cancov, qmd_dom, ba_ge_3, fortypba, or 'all')",
    )

    parser.add_argument(
        "--method",
        type=str,
        choices=["gdalwarp", "buildvrt", "python"],
        default="gdalwarp",
        help="Mosaic creation method (default: gdalwarp)",
    )

    parser.add_argument(
        "--blend-distance",
        type=int,
        default=None,
        help="Feather blending distance in pixels (default: None = no blending)",
    )

    parser.add_argument(
        "--compression",
        type=str,
        choices=["DEFLATE", "LZW", "ZSTD", "NONE"],
        default="DEFLATE",
        help="Compression algorithm (default: DEFLATE)",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )

    parser.add_argument(
        "--clip",
        type=Path,
        default=None,
        help="Path to GeoJSON/Shapefile boundary for clipping mosaic",
    )

    parser.add_argument(
        "--crs",
        type=str,
        default=None,
        help="CRS of input tiles (e.g., EPSG:5070). Required when tiles lack CRS metadata",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate input directory
    if not args.input_dir.exists():
        logger.error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)

    # Validate clip boundary if specified
    if args.clip and not args.clip.exists():
        logger.error(f"Clip boundary file not found: {args.clip}")
        sys.exit(1)

    # Determine tasks to process
    if args.task.lower() == "all":
        tasks = DEFAULT_TASKS
    else:
        tasks = [args.task]

    # Process each task
    results = []
    for task in tasks:
        success = create_mosaic_for_task(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            task_name=task,
            method=args.method,
            blend_distance=args.blend_distance,
            compression=args.compression,
            overwrite=args.overwrite,
            clip_boundary=args.clip,
            crs_override=args.crs,
        )
        results.append((task, success))

    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)

    for task, success in results:
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"  {task}: {status}")

    # Exit with error if any failed
    if not all(s for _, s in results):
        sys.exit(1)

    logger.info(f"\nMosaics saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
