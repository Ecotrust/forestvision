#!/usr/bin/env python3
"""
Generate balanced tile collection for forest type classification.

1. Download state boundaries
2. Generate tile grid
3. Compute class frequencies from GNN data
4. Select balanced subset using configurable strategy
5. Export tiles (single file or k-fold splits)

Usage:
    python scripts/sample_tiles_v2.py --output-path data/fortypba/tiles/
    python scripts/sample_tiles_v2.py --output-path data/fortypba/tiles/ --k-folds 5
"""

# ============================================================================
# SECTION 1: IMPORTS
# ============================================================================
import argparse
import hashlib
import logging
import os
import sys
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Iterator, Protocol

import geopandas as gpd
import pandas as pd
import requests
import torch
from shapely.geometry import box
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torchgeo.datasets import stack_samples
from tqdm import tqdm

from forestvision.datasets import GNNForestAttr
from forestvision.samplers import TileGeoSampler
from forestvision.samplers.utils import roi_to_tiles

os.environ["CPL_LOG"] = "/dev/null"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# SECTION 2: CONSTANTS
# ============================================================================
STATE_FIPS = {"Oregon": "41", "Washington": "53"}
CENSUS_URL = "https://www2.census.gov/geo/tiger/TIGER2024/STATE/tl_2024_us_state.zip"


# ============================================================================
# SECTION 3: CONFIGURATION
# ============================================================================
@dataclass(frozen=True)
class SamplerConfig:
    """Immutable configuration for tile sampling pipeline."""

    output_path: Path
    name: str | None = None
    gnn_path: str = "data/datasets/gnn"
    boundary_cache: str = "data/boundaries"
    states: list[str] = field(default_factory=lambda: ["Oregon", "Washington"])
    tile_size: int = 128
    tile_res: int = 10
    sample_size: int = 20000
    random_state: int = 42
    batch_size: int = 64
    num_workers: int = 10
    input_tiles: str | None = None
    frequency_file: str | None = None
    overwrite_freq: bool = False
    dry_run: bool = False
    k_folds: int | None = None
    val_split: float = 0.1
    balance_strategy: str = "none"
    min_samples_per_class: int = 10

    def __post_init__(self):
        if self.tile_size <= 0:
            raise ValueError(f"tile_size must be positive, got {self.tile_size}")
        if self.k_folds is not None and self.k_folds < 2:
            raise ValueError(f"k_folds must be >= 2, got {self.k_folds}")

    def get_output_path(self) -> Path:
        """Get the full output path including auto-created subdirectory."""
        subdir = f"tiles_{self.tile_size}x{self.tile_size}"
        return self.output_path / subdir

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "SamplerConfig":
        """Create config from parsed CLI arguments."""
        return cls(
            output_path=Path(args.output_path),
            name=args.name,
            gnn_path=args.gnn_path,
            boundary_cache=args.boundary_cache,
            states=args.states,
            tile_size=args.tile_size,
            tile_res=args.tile_res,
            sample_size=args.sample_size,
            random_state=args.random_state,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            input_tiles=args.input_tiles,
            frequency_file=args.frequency_file,
            overwrite_freq=args.overwrite_freq,
            dry_run=args.dry_run,
            k_folds=args.k_folds,
            val_split=args.val_split,
            balance_strategy=args.balance_strategy,
            min_samples_per_class=args.min_samples_per_class,
        )

    def get_base_name(self) -> str:
        """Generate base name for output files."""
        if self.name:
            return self.name
        return f"balanced_{self.tile_size}x{self.tile_size}_{self.tile_res}m"

    def get_all_tiles_path(self) -> Path:
        """Get path for the all tiles GeoJSON file."""
        return self.get_output_path() / f"{self.get_base_name()}_all.geojson"

    def get_frequencies_path(self) -> Path:
        """Get path for the class frequencies CSV file."""
        return self.get_output_path() / f"{self.get_base_name()}_frequencies.csv"


# ============================================================================
# SECTION 4: PIPELINE CONTEXT
# ============================================================================
@dataclass
class SamplingContext:
    """Mutable state container passed between pipeline steps."""

    config: SamplerConfig
    boundary_gdf: gpd.GeoDataFrame | None = None
    tiles_gdf: gpd.GeoDataFrame | None = None
    class_freq_df: pd.DataFrame | None = None
    selected_tiles: pd.DataFrame | None = None


# ============================================================================
# SECTION 5: SELECTION STRATEGIES (Strategy Pattern)
# ============================================================================
class SelectionStrategy(Protocol):
    """Protocol for tile selection strategies."""

    def compute_quotas(
        self, class_distribution: pd.Series, total: int
    ) -> dict[int, int]:
        """Compute target number of tiles per class.

        Args:
            class_distribution: Series mapping class -> tile count
            total: Total number of tiles to sample

        Returns:
            Dictionary mapping class -> quota
        """
        ...


@dataclass
class EqualStrategy:
    """Equal tiles per class strategy."""

    min_samples_per_class: int = 10

    def compute_quotas(
        self, class_distribution: pd.Series, total: int
    ) -> dict[int, int]:
        n_classes = len(class_distribution)
        base = total // n_classes
        remainder = total - (base * n_classes)
        quotas = {}
        for i, cls in enumerate(class_distribution.index):
            quotas[cls] = max(self.min_samples_per_class, base + (1 if i < remainder else 0))
        return quotas


@dataclass
class ProportionalStrategy:
    """Preserve natural class distribution."""

    min_samples_per_class: int = 10

    def compute_quotas(
        self, class_distribution: pd.Series, total: int
    ) -> dict[int, int]:
        total_tiles = class_distribution.sum()
        quotas = {}
        for cls, count in class_distribution.items():
            target = int(total * count / total_tiles)
            quotas[cls] = max(self.min_samples_per_class, target)
        return quotas


@dataclass
class InverseFreqStrategy:
    """Boost rare classes with inverse frequency weighting."""

    min_samples_per_class: int = 10

    def compute_quotas(
        self, class_distribution: pd.Series, total: int
    ) -> dict[int, int]:
        # Inverse frequency weights
        weights = 1.0 / class_distribution
        weights = weights / weights.sum()
        return {
            cls: max(self.min_samples_per_class, int(total * w))
            for cls, w in weights.items()
        }


@dataclass
class CappedStrategy:
    """Capped pixel-level balancing strategy (from notebook).
    
    Per-class max-filtering with weighted sampling. For each ODF class,
    keeps only tiles where that class has the highest pixel count, then
    samples up to a target number. Tiles can be shared between classes.
    
    Matches the implementation in notebooks/fotypba_fetch_data copy.ipynb
    """

    min_samples_per_class: int = 10
    nodata_class: int = -2147483648

    def select_tiles(
        self,
        eligible_tiles: pd.DataFrame,
        class_freq_df: pd.DataFrame,
        sample_size: int,
        random_state: int,
    ) -> pd.DataFrame:
        """Select tiles using weighted capped sampling.
        
        Args:
            eligible_tiles: DataFrame with tile metrics (geohash, etc.)
            class_freq_df: Full class frequency data per tile
            sample_size: Target tiles per class (uses sample_size // num_classes)
            random_state: Random seed for reproducibility
        
        Returns:
            DataFrame with selected tiles (tile sharing allowed between classes)
        """
        # Filter out nodata and -1 from class_freq_df
        valid_df = class_freq_df[
            ~class_freq_df["odf_class"].isin([self.nodata_class, -1])
        ].copy()
        
        if len(valid_df) == 0:
            logger.warning("No valid class data after filtering nodata")
            return eligible_tiles.sample(
                n=min(len(eligible_tiles), sample_size), 
                random_state=random_state
            )
        
        # Calculate target per class from sample_size
        # First, determine how many unique ODF classes we have
        unique_odf_classes = valid_df["odf_class"].unique()
        n_classes = len(unique_odf_classes)
        target_per_class = sample_size  # Each class gets the full target (tiles can be shared)
        
        logger.info(f"Capped weighted sampling: target = {target_per_class:,} tiles per class")
        logger.info(f"Unique ODF classes: {n_classes} (tiles can be shared between classes)")
        
        # Group by geohash and odf_class, sum counts
        # This creates: odf_counts = gnn_cl_counts.groupby(["hashes", "gnn_class"]).sum()
        odf_counts = valid_df.groupby(["geohash", "odf_class"])["gnn_counts"].sum().reset_index()
        
        # Per-class max-filtering with fallback for low-coverage tiles
        
        # First pass: strict max-filtering (class must be dominant)
        tile_assignments = []  # Tracks which tiles got assigned to which class
        
        for odf_class in unique_odf_classes:
            # Filter by odf_class
            group = odf_counts[odf_counts["odf_class"] == odf_class].copy()
            
            if len(group) == 0:
                logger.warning(f"ODF {odf_class}: no tiles found")
                continue
            
            # Get all data for tiles in this group
            tiles_in_group = group["geohash"].unique()
            all_data_for_tiles = odf_counts[odf_counts["geohash"].isin(tiles_in_group)]
            
            # For each tile, find which odf_class has max counts
            idx_max = all_data_for_tiles.groupby("geohash")["gnn_counts"].idxmax()
            max_class_per_tile = all_data_for_tiles.loc[idx_max]
            
            # Keep only tiles where this odf_class has highest counts
            dominant_tiles = max_class_per_tile[max_class_per_tile["odf_class"] == odf_class]["geohash"]
            group_filtered = group[group["geohash"].isin(dominant_tiles)].copy()
            
            if len(group_filtered) > 0:
                tile_assignments.append(group_filtered)
                logger.info(f"ODF {odf_class}: {len(group_filtered)} dominant tiles")
        
        # Combine all assignments and track which tiles were assigned
        if tile_assignments:
            assigned_df = pd.concat(tile_assignments)
            assigned_hashes = set(assigned_df["geohash"].unique())
        else:
            assigned_df = pd.DataFrame(columns=odf_counts.columns)
            assigned_hashes = set()
        
        # Fallback: tiles with low coverage everywhere (no dominant class)
        all_hashes = set(odf_counts["geohash"].unique())
        fallback_hashes = all_hashes - assigned_hashes
        
        if fallback_hashes:
            logger.info(f"Fallback: {len(fallback_hashes)} tiles with no dominant class")
            # For unassigned tiles, take their highest-count class
            fallback_data = odf_counts[odf_counts["geohash"].isin(fallback_hashes)]
            # Sort by geohash then counts descending, take first (max) per geohash
            fallback_data = fallback_data.sort_values(
                ["geohash", "gnn_counts"], ascending=[True, False]
            )
            fallback_assignment = fallback_data.groupby("geohash").first().reset_index()
            logger.info(f"Fallback assigned {len(fallback_assignment)} tiles to their highest class")
            
            # Combine strict assignments with fallback
            combined_df = pd.concat([assigned_df, fallback_assignment])
        else:
            combined_df = assigned_df
        
        # Now sample from combined assignments per class
        balanced_dfs = []
        for odf_class in unique_odf_classes:
            class_data = combined_df[combined_df["odf_class"] == odf_class]
            
            if len(class_data) == 0:
                logger.warning(f"ODF {odf_class}: no tiles after fallback")
                continue
            
            # Sample to target size
            n = min(len(class_data), target_per_class)
            sampled = class_data.sample(n=n, replace=False, random_state=random_state)
            balanced_dfs.append(sampled)
            
            # Report if fallback contributed
            strict_count = len(assigned_df[assigned_df["odf_class"] == odf_class]) if len(assigned_df) > 0 else 0
            fallback_count = len(fallback_assignment[fallback_assignment["odf_class"] == odf_class]) if fallback_hashes and len(fallback_assignment) > 0 else 0
            if fallback_count > 0:
                logger.info(f"ODF {odf_class}: {strict_count} dominant + {fallback_count} fallback = {len(class_data)} total, sampled {n}")
            else:
                logger.info(f"ODF {odf_class}: {strict_count} dominant tiles, sampled {n}")
        
        if not balanced_dfs:
            logger.warning("No tiles selected for any class, falling back to random")
            return eligible_tiles.sample(
                n=min(len(eligible_tiles), sample_size), 
                random_state=random_state
            )
        
        # Concatenate all class samples and shuffle
        remap_bal = pd.concat(balanced_dfs).sample(frac=1, random_state=random_state)
        
        # Get unique geohashes (tiles can be shared, so we deduplicate)
        selected_geohashes = remap_bal["geohash"].unique()
        
        logger.info(f"Total selected: {len(selected_geohashes)} unique tiles")
        
        # Report per-class tile counts
        logger.info("Selected tile distribution:")
        for odf_class in sorted(unique_odf_classes):
            class_tiles = remap_bal[remap_bal["odf_class"] == odf_class]["geohash"].nunique()
            class_pixels = remap_bal[remap_bal["odf_class"] == odf_class]["gnn_counts"].sum()
            logger.info(f"  ODF {odf_class}: {class_tiles:>4} tiles, {class_pixels:>12,} pixels")
        
        # Return selected tiles (may be fewer than requested due to deduplication)
        selected = eligible_tiles[
            eligible_tiles["geohash"].isin(selected_geohashes)
        ].copy()
        
        return selected


# Strategy registry for lookup by name
STRATEGY_REGISTRY: dict[str, type[SelectionStrategy]] = {
    "equal": EqualStrategy,
    "proportional": ProportionalStrategy,
    "inverse_freq": InverseFreqStrategy,
}


def create_strategy(name: str, min_samples: int = 10) -> SelectionStrategy:
    """Factory function to create strategy instances."""
    if name == "capped":
        return CappedStrategy(min_samples_per_class=min_samples)
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {name}. Use: {list(STRATEGY_REGISTRY.keys())} or 'capped'")
    return STRATEGY_REGISTRY[name](min_samples_per_class=min_samples)


# ============================================================================
# SECTION 6: PIPELINE STEPS
# ============================================================================
class PipelineStep(ABC):
    """Abstract base class for pipeline steps."""

    @abstractmethod
    def execute(self, ctx: SamplingContext) -> None:
        """Execute the step, modifying context in place."""
        pass


class LoadOrGenerateTilesStep(PipelineStep):
    """Load existing tiles or generate from state boundaries."""

    def execute(self, ctx: SamplingContext) -> None:
        config = ctx.config

        if config.input_tiles:
            # Explicit CLI override: load user-specified tiles
            logger.info(f"Loading existing tiles from: {config.input_tiles}")
            ctx.tiles_gdf = self._load_tiles(config.input_tiles)
        else:
            # Check for cached tiles file
            all_tiles_path = config.get_all_tiles_path()
            if all_tiles_path.exists():
                logger.info(f"Loading cached tiles from: {all_tiles_path}")
                ctx.tiles_gdf = self._load_tiles(str(all_tiles_path))
            else:
                # Generate from boundaries
                logger.info(f"Generating tiles for states: {', '.join(config.states)}")
                boundary_gdf = self._download_boundaries(config)
                ctx.boundary_gdf = boundary_gdf
                ctx.tiles_gdf = self._generate_grid(boundary_gdf, config)

                # Save generated tiles if not dry-run
                if not config.dry_run:
                    ctx.tiles_gdf.to_file(all_tiles_path, driver="GeoJSON")
                    logger.info(f"Saved {len(ctx.tiles_gdf)} tiles to: {all_tiles_path}")

        if config.dry_run and len(ctx.tiles_gdf) > 10:
            logger.info(f"Dry-run: Limiting to 10 tiles")
            ctx.tiles_gdf = ctx.tiles_gdf.head(10)

    def _load_tiles(self, path: str) -> gpd.GeoDataFrame:
        """Load tiles from GeoJSON, ensuring geohash column exists."""
        gdf = gpd.read_file(path)
        if "geohash" not in gdf.columns:
            gdf["geohash"] = [
                hashlib.md5(geom.bounds.__repr__().encode()).hexdigest()[:10]
                for geom in gdf.geometry
            ]
        return gdf

    def _download_boundaries(self, config: SamplerConfig) -> gpd.GeoDataFrame:
        """Download and filter state boundaries from Census."""
        cache_path = Path(config.boundary_cache)
        cache_path.mkdir(parents=True, exist_ok=True)
        shapefile_path = cache_path / "tl_2024_us_state.shp"

        if not shapefile_path.exists():
            logger.info(f"Downloading state boundaries from {CENSUS_URL}")
            response = requests.get(CENSUS_URL, timeout=120)
            response.raise_for_status()
            with zipfile.ZipFile(BytesIO(response.content)) as z:
                z.extractall(cache_path)

        gdf = gpd.read_file(shapefile_path)
        state_fps = [STATE_FIPS.get(s, s) for s in config.states]
        mask = gdf["NAME"].isin(config.states) | gdf["STATEFP"].isin(state_fps)
        filtered = gdf[mask].copy()

        if len(filtered) == 0:
            raise ValueError(f"No states found matching: {config.states}")

        # Reproject to EPSG:5070
        if filtered.crs is None or filtered.crs.to_epsg() != 5070:
            filtered = filtered.to_crs("EPSG:5070")

        return filtered

    def _generate_grid(
        self, boundary_gdf: gpd.GeoDataFrame, config: SamplerConfig
    ) -> gpd.GeoDataFrame:
        """Generate non-overlapping tile grid within boundaries."""
        from torchgeo.datasets import BoundingBox

        tile_size_m = config.tile_size * config.tile_res
        total_bounds = boundary_gdf.total_bounds

        roi = BoundingBox(
            total_bounds[0],
            total_bounds[2],
            total_bounds[1],
            total_bounds[3],
            0,
            sys.maxsize,
        )

        tiles_bounds = roi_to_tiles(
            roi=roi,
            size=config.tile_size,
            res=config.tile_res,
            stride=config.tile_size,
        )

        geometries = [box(*bounds) for bounds in tiles_bounds]
        tiles_gdf = gpd.GeoDataFrame({"geometry": geometries}, crs="EPSG:5070")

        tiles_gdf["geohash"] = [
            hashlib.md5(geom.bounds.__repr__().encode()).hexdigest()[:10]
            for geom in tiles_gdf.geometry
        ]

        # Filter to tiles with centroids inside boundary
        boundary_union = boundary_gdf.union_all()
        tiles_gdf["centroid"] = tiles_gdf.geometry.centroid
        tiles_gdf = tiles_gdf[tiles_gdf["centroid"].apply(boundary_union.contains)].copy()
        tiles_gdf = tiles_gdf.drop(columns=["centroid"])

        logger.info(f"Generated {len(tiles_gdf)} tiles within boundaries")
        return tiles_gdf


class ComputeClassFrequenciesStep(PipelineStep):
    """Compute class frequencies for each tile from GNN data."""

    def execute(self, ctx: SamplingContext) -> None:
        config = ctx.config
        tiles_gdf = ctx.tiles_gdf

        # Check for user-specified frequency file first
        if config.frequency_file:
            freq_path = Path(config.frequency_file)
            if freq_path.exists():
                logger.info(f"Loading specified frequency file: {freq_path}")
                ctx.class_freq_df = pd.read_csv(freq_path)
                return
            else:
                logger.warning(f"Specified frequency file not found: {freq_path}")
                logger.info("Will compute frequencies from GNN data instead")

        # Check for cached frequencies in output directory
        output_csv = config.get_frequencies_path()
        if not config.overwrite_freq and output_csv.exists():
            logger.info(f"Loading existing class frequencies from: {output_csv}")
            ctx.class_freq_df = pd.read_csv(output_csv)
            return

        logger.info(f"Computing class frequencies from: {config.gnn_path}")
        gnn = GNNForestAttr(
            paths=config.gnn_path,
            bands=["fortypba"],
            remap=False,
            crs="EPSG:5070",
            res=10,
        )

        # Filter tiles to GNN bounds
        from shapely.geometry import box as shapely_box

        bounds = gnn.bounds
        gnn_bbox = shapely_box(bounds.minx, bounds.miny, bounds.maxx, bounds.maxy)
        tiles_gdf = tiles_gdf[tiles_gdf.geometry.intersects(gnn_bbox)].copy()

        if len(tiles_gdf) == 0:
            logger.warning("No tiles intersect with GNN dataset bounds!")
            ctx.class_freq_df = pd.DataFrame(
                columns=["geohash", "gnn_class", "gnn_counts", "frequency", "total_counts"]
            )
            return

        # Sample GNN data
        sampler = TileGeoSampler(gnn, tiles_gdf)
        dataloader = DataLoader(
            gnn,
            sampler=sampler,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            collate_fn=stack_samples,
        )

        ctx.class_freq_df = self._process_batches(dataloader, gnn)

        # Save if not dry-run
        if not config.dry_run:
            ctx.class_freq_df.to_csv(output_csv, index=False)
            logger.info(f"Saved class frequencies to: {output_csv}")

    def _process_batches(
        self, dataloader: DataLoader, gnn
    ) -> pd.DataFrame:
        """Process all batches and compute class frequencies."""
        records = []
        skipped = []
        nodata_value = gnn.nodata

        for batch in tqdm(dataloader, desc="Computing class frequencies"):
            if batch["mask"].shape[-2] != batch["mask"].shape[-1]:
                continue

            raw_values = batch["mask"].type(torch.int)
            batch_hashes = [
                hashlib.md5(str((b.minx, b.miny, b.maxx, b.maxy)).encode()).hexdigest()[:10]
                for b in batch["bounds"]
            ]

            for i, tile_hash in enumerate(batch_hashes):
                tile_data = raw_values[i]

                classes, counts = tile_data.unique(return_counts=True)
                for cls, cnt in zip(classes.tolist(), counts.tolist()):
                    records.append({"geohash": tile_hash, "gnn_class": cls, "gnn_counts": cnt})

        df = pd.DataFrame(records)
        if len(df) == 0:
            return df

        # Compute frequencies
        tile_totals = df.groupby("geohash")["gnn_counts"].sum().reset_index()
        tile_totals.columns = ["geohash", "total_counts"]
        df = df.merge(tile_totals, on="geohash")
        df["frequency"] = df["gnn_counts"] / df["total_counts"]
        df["odf_class"] = df["gnn_class"].replace(gnn.remap_dict)

        return df


class SelectTilesStep(PipelineStep):
    """Select balanced subset of tiles using configured strategy."""

    def __init__(self, strategy: SelectionStrategy | None = None):
        self.strategy = strategy

    def execute(self, ctx: SamplingContext) -> None:
        config = ctx.config
        class_freq_df = ctx.class_freq_df

        if len(class_freq_df) == 0:
            logger.warning("No class frequency data available for selection")
            ctx.selected_tiles = pd.DataFrame()
            return

        # Compute tile metrics
        tile_metrics = self._compute_metrics(class_freq_df)

        # All tiles are eligible (no filtering)
        eligible = tile_metrics

        # Select tiles
        if self.strategy is None:
            # Simple random sampling (original behavior)
            n = min(config.sample_size, len(eligible))
            selected = eligible.sample(n=n, random_state=config.random_state)
        else:
            # Class-aware sampling with strategy
            selected = self._sample_with_strategy(
                eligible, class_freq_df, config.sample_size, config.random_state
            )

        ctx.selected_tiles = selected
        logger.info(f"Selected {len(selected)} tiles")

    def _compute_metrics(self, class_freq_df: pd.DataFrame) -> pd.DataFrame:
        """Compute per-tile metrics for selection."""

        def get_dominant_class(group):
            idxmax = group["frequency"].idxmax()
            return group.loc[idxmax, "odf_class"]

        return (
            class_freq_df.groupby("geohash")
            .agg(
                num_gnn_classes=("gnn_class", "nunique"),
                num_odf_classes=("odf_class", "nunique"),
                max_frequency=("frequency", "max"),
                total_counts=("gnn_counts", "sum"),
                dominant_odf_class=("odf_class", lambda x: get_dominant_class(class_freq_df.loc[x.index])),
            )
            .reset_index()
        )

    def _sample_with_strategy(
        self,
        eligible: pd.DataFrame,
        class_freq_df: pd.DataFrame,
        sample_size: int,
        random_state: int,
    ) -> pd.DataFrame:
        """Sample tiles using the configured strategy."""
        # Check if strategy is capped (has different interface)
        if isinstance(self.strategy, CappedStrategy):
            return self.strategy.select_tiles(eligible, class_freq_df, sample_size, random_state)
        
        # Standard tile-level strategies (equal, proportional, inverse_freq)
        # Get class distribution
        class_dist = eligible["dominant_odf_class"].value_counts().sort_index()
        quotas = self.strategy.compute_quotas(class_dist, sample_size)

        logger.info(f"Per-class quotas: {quotas}")

        selected = []
        for odf_class, quota in quotas.items():
            class_tiles = eligible[eligible["dominant_odf_class"] == odf_class]
            if len(class_tiles) == 0:
                logger.warning(f"No eligible tiles for ODF class {odf_class}")
                continue

            n = min(quota, len(class_tiles))
            if n < quota:
                logger.warning(
                    f"ODF class {odf_class}: only {len(class_tiles)} tiles available (quota: {quota})"
                )

            sampled = class_tiles.sample(n=n, random_state=random_state)
            selected.append(sampled)

        return pd.concat(selected, ignore_index=True)


# ============================================================================
# SECTION 7: EXPORTERS
# ============================================================================
class Exporter(ABC):
    """Abstract base class for exporters."""

    @abstractmethod
    def export(self, ctx: SamplingContext) -> None:
        pass


class SingleFileExporter(Exporter):
    """Export selected tiles to single GeoJSON file."""

    def export(self, ctx: SamplingContext) -> None:
        config = ctx.config
        if config.dry_run:
            return

        output_path = config.get_output_path() / f"{config.get_base_name()}.geojson"
        selected_hashes = ctx.selected_tiles["geohash"].tolist()
        result_gdf = ctx.tiles_gdf[ctx.tiles_gdf["geohash"].isin(selected_hashes)].copy()

        result_gdf.to_file(output_path, driver="GeoJSON")
        logger.info(f"Saved {len(result_gdf)} tiles to: {output_path}")


class KFoldExporter(Exporter):
    """Export tiles as stratified k-fold train/validation splits."""

    def __init__(self, n_folds: int, val_split: float):
        self.n_folds = n_folds
        self.val_split = val_split

    def export(self, ctx: SamplingContext) -> None:
        config = ctx.config
        if config.dry_run:
            return

        logger.info(f"Generating {self.n_folds} stratified folds with {self.val_split:.0%} validation")

        # Create stratification labels
        stratify_df = self._create_stratification_labels(ctx)

        # Split into validation and training pool
        val_hashes, train_pool_df = self._split_validation(stratify_df, config.random_state)

        # Generate k folds from training pool
        fold_indices = self._generate_folds(train_pool_df, config.random_state)

        # Save validation set
        self._save_validation(ctx, val_hashes)

        # Save training folds
        for fold_idx, fold_hashes in enumerate(fold_indices, 1):
            self._save_fold(ctx, fold_idx, fold_hashes)

        logger.info(f"Exported {self.n_folds} folds + validation set")

    def _create_stratification_labels(self, ctx: SamplingContext) -> pd.DataFrame:
        """Create stratification labels combining dominant class with class diversity."""
        df = ctx.selected_tiles.copy()

        # Bin the number of ODF classes
        df["num_odf_classes_bin"] = pd.cut(
            df["num_odf_classes"],
            bins=[0, 2, 4, 6, 8, float("inf")],
            labels=["1-2", "3-4", "5-6", "7-8", "9+"],
        )

        # Combine dominant class with diversity bin
        df["stratify_label"] = (
            df["dominant_odf_class"].astype(str) + "_" + df["num_odf_classes_bin"].astype(str)
        )

        # Combine rare labels
        label_counts = df["stratify_label"].value_counts()
        min_count = max(int(1 / self.val_split), self.n_folds)
        rare_labels = label_counts[label_counts < min_count].index.tolist()

        if rare_labels:
            logger.info(f"Combining {len(rare_labels)} rare stratification groups")
            df["stratify_label"] = df["stratify_label"].apply(
                lambda x: "other" if x in rare_labels else x
            )

        return df

    def _split_validation(
        self, df: pd.DataFrame, random_state: int
    ) -> tuple[list[str], pd.DataFrame]:
        """Split into validation set and training pool."""
        n_splits = int(1 / self.val_split)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        train_idx, val_idx = next(iter(skf.split(df, df["stratify_label"])))

        val_hashes = df.iloc[val_idx]["geohash"].tolist()
        train_pool_df = df.iloc[train_idx].copy()

        logger.info(f"Training pool: {len(train_pool_df)} tiles, Validation: {len(val_hashes)} tiles")

        return val_hashes, train_pool_df

    def _generate_folds(self, train_pool_df: pd.DataFrame, random_state: int) -> Iterator[list[str]]:
        """Generate k folds from training pool."""
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=random_state
        )

        for _, fold_idx in skf.split(train_pool_df, train_pool_df["stratify_label"]):
            yield train_pool_df.iloc[fold_idx]["geohash"].tolist()

    def _save_validation(self, ctx: SamplingContext, val_hashes: list[str]) -> None:
        """Save validation set to files."""
        config = ctx.config
        base_name = config.get_base_name()
        output_path = config.get_output_path()

        val_gdf = ctx.tiles_gdf[ctx.tiles_gdf["geohash"].isin(val_hashes)].copy()
        val_gdf.to_file(output_path / f"{base_name}_val.geojson", driver="GeoJSON")

        val_csv = ctx.class_freq_df[ctx.class_freq_df["geohash"].isin(val_hashes)]
        val_csv.to_csv(output_path / f"{base_name}_val.csv", index=False)

    def _save_fold(self, ctx: SamplingContext, fold_idx: int, fold_hashes: list[str]) -> None:
        """Save a single fold to files."""
        config = ctx.config
        base_name = config.get_base_name()
        output_path = config.get_output_path()

        fold_gdf = ctx.tiles_gdf[ctx.tiles_gdf["geohash"].isin(fold_hashes)].copy()
        fold_gdf.to_file(
            output_path / f"{base_name}_fold_{fold_idx}_train.geojson",
            driver="GeoJSON",
        )

        fold_csv = ctx.class_freq_df[ctx.class_freq_df["geohash"].isin(fold_hashes)]
        fold_csv.to_csv(
            output_path / f"{base_name}_fold_{fold_idx}_train.csv", index=False
        )

        logger.info(f"Fold {fold_idx}: {len(fold_gdf)} tiles")


# ============================================================================
# SECTION 8: REPORTING
# ============================================================================
def compute_inverse_frequency_weights(class_freq_df: pd.DataFrame) -> pd.DataFrame:
    """Compute inverse frequency weights for each ODF class.
    
    Args:
        class_freq_df: DataFrame with class frequency data per tile.
        
    Returns:
        DataFrame with columns: odf_class, pixel_count, frequency, 
        raw_weight, normalized_weight
    """
    if len(class_freq_df) == 0:
        return pd.DataFrame(columns=[
            "odf_class", "pixel_count", "frequency", 
            "raw_weight", "normalized_weight"
        ])
    
    # Aggregate pixel counts by ODF class
    odf_totals = class_freq_df.groupby("odf_class")["gnn_counts"].sum().sort_index()
    
    # Filter out nodata values
    nodata_values = [-2147483648, -1]
    valid_odf_totals = odf_totals[~odf_totals.index.isin(nodata_values)]
    valid_total = valid_odf_totals.sum()
    
    if valid_total == 0 or len(valid_odf_totals) == 0:
        return pd.DataFrame(columns=[
            "odf_class", "pixel_count", "frequency", 
            "raw_weight", "normalized_weight"
        ])
    
    # Calculate frequencies and inverse weights
    frequencies = valid_odf_totals / valid_total
    inv_weights = 1.0 / frequencies
    normalized_weights = inv_weights / inv_weights.sum() * len(valid_odf_totals)
    
    # Create result DataFrame
    weights_df = pd.DataFrame({
        "odf_class": valid_odf_totals.index,
        "pixel_count": valid_odf_totals.values,
        "frequency": frequencies.values,
        "raw_weight": inv_weights.values,
        "normalized_weight": normalized_weights.values,
    })
    
    return weights_df


def save_inverse_frequency_weights(
    weights_df: pd.DataFrame, output_path: Path, base_name: str
) -> Path:
    """Save inverse frequency weights to CSV file.
    
    Args:
        weights_df: DataFrame with inverse frequency weights.
        output_path: Directory to save the file.
        base_name: Base name for the output file.
        
    Returns:
        Path to the saved CSV file.
    """
    if len(weights_df) == 0:
        logger.warning("No inverse frequency weights to save")
        return None
    
    csv_path = output_path / f"{base_name}_inverse_frequency_weights.csv"
    weights_df.to_csv(csv_path, index=False)
    logger.info(f"Saved inverse frequency weights to: {csv_path}")
    return csv_path


def print_class_frequency_report(
    class_freq_df: pd.DataFrame, weights_df: pd.DataFrame | None = None
) -> None:
    """Print formatted class frequency report with inverse frequency weights.
    
    Args:
        class_freq_df: DataFrame with class frequency data per tile.
        weights_df: Optional pre-computed weights DataFrame. If None, weights
            will be computed from class_freq_df.
    """
    if len(class_freq_df) == 0:
        print("\nNo class frequency data available.")
        return
    
    # Compute weights if not provided
    if weights_df is None:
        weights_df = compute_inverse_frequency_weights(class_freq_df)

    print("\n" + "=" * 60)
    print(f"{'CLASS FREQUENCY REPORT':^60}")
    print("=" * 60)

    # Overall distribution
    print("\nOverall ODF Class Distribution:")
    odf_totals = class_freq_df.groupby("odf_class")["gnn_counts"].sum().sort_index()
    total = odf_totals.sum()

    for cls, count in odf_totals.items():
        print(f"  ODF {cls:3d}: {count:12,} pixels ({100*count/total:5.2f}%)")
    print(f"\n  Total:   {total:12,} pixels")

    # Inverse Frequency Weights Section
    print("\n" + "-" * 60)
    print(f"{'INVERSE FREQUENCY WEIGHTS':^60}")
    print("-" * 60)
    print("Formula: weight = 1.0 / frequency, normalized to sum to n_classes")
    print("-" * 60)

    if len(weights_df) > 0:
        print(f"{'ODF Class':<12} {'Pixels':>12} {'Frequency':>12} {'Raw Weight':>12} {'Norm. Weight':>12}")
        print("-" * 60)

        for _, row in weights_df.iterrows():
            print(f"{int(row['odf_class']):<12} {int(row['pixel_count']):>12,} "
                  f"{row['frequency']:>11.4f} {row['raw_weight']:>12.2f} "
                  f"{row['normalized_weight']:>12.2f}")

        print("-" * 60)
        print(f"{'Total':<12} {int(weights_df['pixel_count'].sum()):>12,} "
              f"{weights_df['frequency'].sum():>11.4f} "
              f"{weights_df['raw_weight'].sum():>12.2f} "
              f"{weights_df['normalized_weight'].sum():>12.2f}")
        print(f"\nNote: Normalized weights sum to {len(weights_df):.0f} (number of valid classes)")
        print("      Higher weights indicate rarer classes that receive boost in 'inverse_freq' strategy")
    else:
        print("  No valid class data available for weight calculation")

    # Per-tile stats
    print("\n" + "=" * 60)
    tile_classes = class_freq_df.groupby("geohash")["odf_class"].nunique()
    print(f"Per-Tile Statistics:")
    print(f"  Total tiles: {class_freq_df['geohash'].nunique()}")
    print(f"  Avg classes per tile: {tile_classes.mean():.1f}")

    print("=" * 60)


def print_selection_report(ctx: SamplingContext) -> None:
    """Print final selection summary."""
    config = ctx.config

    print("\n" + "=" * 60)
    if config.k_folds:
        print(f"{'STRATIFIED K-FOLD SUMMARY':^60}")
    else:
        print(f"{'BALANCED TILE SELECTION SUMMARY':^60}")
    print("=" * 60)

    print(f"Total tiles generated:    {len(ctx.tiles_gdf)}")
    if ctx.class_freq_df is not None:
        print(f"Tiles with class data:    {ctx.class_freq_df['geohash'].nunique()}")
    if ctx.selected_tiles is not None:
        print(f"Selected tiles:           {len(ctx.selected_tiles)}")

    if config.balance_strategy != "none":
        print(f"\nBalance strategy:         {config.balance_strategy}")
    else:
        print(f"\nBalance strategy:         random (no class balancing)")

    if not config.dry_run:
        print(f"\nOutput directory: {config.get_output_path()}")

    print("=" * 60 + "\n")

    if config.dry_run:
        print("DRY-RUN COMPLETE: No files were saved.\n")


def print_balanced_data_report(ctx: SamplingContext) -> None:
    """Print pixel/tile report for selected/balanced tiles.

    Shows class distribution within the selected tile set, allowing
    comparison with the overall distribution from print_class_frequency_report().

    For capped strategies, tiles can belong to multiple classes, so pixel
    counts reflect all ODF classes present in selected tiles.

    Args:
        ctx: SamplingContext containing selected_tiles and class_freq_df.
    """
    if ctx.selected_tiles is None or len(ctx.selected_tiles) == 0:
        print("\nNo selected tiles to report.")
        return

    if ctx.class_freq_df is None or len(ctx.class_freq_df) == 0:
        print("\nNo class frequency data available for selected tiles.")
        return

    print("\n" + "=" * 60)
    print(f"{'SELECTED/BALANCED DATA REPORT':^60}")
    print("=" * 60)

    # Get selected geohashes
    selected_geohashes = ctx.selected_tiles["geohash"].unique()

    # Filter class_freq_df to only selected tiles
    selected_freq_df = ctx.class_freq_df[
        ctx.class_freq_df["geohash"].isin(selected_geohashes)
    ].copy()

    if len(selected_freq_df) == 0:
        print("\nNo class frequency data found for selected tiles.")
        print("=" * 60)
        return

    # Total selected tiles
    total_selected_tiles = len(selected_geohashes)
    print(f"\nTotal selected tiles: {total_selected_tiles}")

    # --- Per-class TILE counts (dominant class) ---
    print("\nPer-Class Tile Counts (by dominant ODF class):")
    print("-" * 50)

    # Count tiles where each ODF class is dominant
    if "dominant_odf_class" in ctx.selected_tiles.columns:
        tile_counts = ctx.selected_tiles["dominant_odf_class"].value_counts().sort_index()
    else:
        # Fallback: compute dominant class from selected_freq_df
        def get_dominant_class(group):
            idxmax = group["frequency"].idxmax()
            return group.loc[idxmax, "odf_class"]

        dominant_per_tile = selected_freq_df.groupby("geohash").apply(
            get_dominant_class, include_groups=False
        )
        tile_counts = dominant_per_tile.value_counts().sort_index()

    for cls, count in tile_counts.items():
        pct = 100 * count / total_selected_tiles
        print(f"  ODF {cls:3d}: {count:>6} tiles ({pct:5.2f}%)")
    print(f"\n  Total:   {total_selected_tiles:>6} tiles (100.00%)")

    # --- Per-class PIXEL counts ---
    print("\nPer-Class Pixel Counts (all pixels in selected tiles):")
    print("-" * 50)

    # Sum pixels per ODF class in selected tiles
    # Filter out nodata values
    nodata_values = [-2147483648, -1]
    valid_pixels = selected_freq_df[~selected_freq_df["odf_class"].isin(nodata_values)]

    pixel_counts = valid_pixels.groupby("odf_class")["gnn_counts"].sum().sort_index()
    total_pixels = pixel_counts.sum()

    if total_pixels > 0:
        for cls, count in pixel_counts.items():
            pct = 100 * count / total_pixels
            print(f"  ODF {cls:3d}: {count:>12,} pixels ({pct:5.2f}%)")
        print(f"\n  Total:   {total_pixels:>12,} pixels (100.00%)")
    else:
        print("  No valid pixel data found.")

    # --- Comparison with overall distribution ---
    print("\nComparison with Overall Distribution:")
    print("-" * 50)
    print(f"{'ODF Class':<12} {'Overall %':>12} {'Selected %':>12} {'Change':>10}")
    print("-" * 50)

    # Overall distribution
    overall_pixels = (
        ctx.class_freq_df[~ctx.class_freq_df["odf_class"].isin(nodata_values)]
        .groupby("odf_class")["gnn_counts"]
        .sum()
    )
    overall_total = overall_pixels.sum()

    # Compare distributions
    all_classes = sorted(set(overall_pixels.index) | set(pixel_counts.index))
    for cls in all_classes:
        overall_pct = 100 * overall_pixels.get(cls, 0) / overall_total if overall_total > 0 else 0
        selected_pct = 100 * pixel_counts.get(cls, 0) / total_pixels if total_pixels > 0 else 0
        change = selected_pct - overall_pct
        change_str = f"{change:+6.2f}%"
        print(f"  ODF {cls:<3d}    {overall_pct:>11.2f}% {selected_pct:>11.2f}% {change_str:>10}")

    print("=" * 60)


# ============================================================================
# SECTION 9: PIPELINE ORCHESTRATOR
# ============================================================================
class SamplingPipeline:
    """Orchestrates the tile sampling workflow."""

    def __init__(self, steps: list[PipelineStep], exporter: Exporter):
        self.steps = steps
        self.exporter = exporter

    def run(self, config: SamplerConfig) -> SamplingContext:
        """Execute all pipeline steps."""
        ctx = SamplingContext(config=config)

        for step in self.steps:
            logger.info(f"Executing: {step.__class__.__name__}")
            step.execute(ctx)

        self.exporter.export(ctx)
        return ctx

    @classmethod
    def from_config(cls, config: SamplerConfig) -> "SamplingPipeline":
        """Factory: Build pipeline from configuration."""
        # Determine selection strategy
        strategy = None
        if config.balance_strategy != "none":
            strategy = create_strategy(
                config.balance_strategy,
                min_samples=config.min_samples_per_class,
            )

        # Build steps
        steps: list[PipelineStep] = [
            LoadOrGenerateTilesStep(),
            ComputeClassFrequenciesStep(),
            SelectTilesStep(strategy=strategy),
        ]

        # Choose exporter
        exporter: Exporter
        if config.k_folds:
            exporter = KFoldExporter(n_folds=config.k_folds, val_split=config.val_split)
        else:
            exporter = SingleFileExporter()

        return cls(steps=steps, exporter=exporter)


# ============================================================================
# SECTION 10: CLI
# ============================================================================
def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate balanced tile collection for forest type classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Output
    parser.add_argument("--output-path", type=str, required=True, help="Output directory")
    parser.add_argument("--name", type=str, default=None, help="Base name for output files")

    # Data paths
    parser.add_argument("--gnn-path", type=str, default="data/datasets/gnn")
    parser.add_argument("--boundary-cache", type=str, default="data/datasets/boundaries")
    parser.add_argument("--states", nargs="+", default=["Oregon", "Washington"])

    # Tile parameters
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--tile-res", type=int, default=10)

    # Selection criteria
    parser.add_argument("--sample-size", type=int, default=8000)
    parser.add_argument("--random-state", type=int, default=42)

    # Balancing strategy
    parser.add_argument("--balance-strategy", type=str, default="none",
                       choices=["none", "equal", "proportional", "inverse_freq", "capped"],
                       help="Tile selection strategy: 'none' for random, 'capped' for pixel-level capping")
    parser.add_argument("--min-samples-per-class", type=int, default=10,
                       help="Minimum tiles per class (for class-aware strategies)")

    # Processing
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--input-tiles", type=str, default=None,
                       help="Override: Use custom tiles GeoJSON instead of cached/generated")
    parser.add_argument("--frequency-file", type=str, default=None,
                       help="Override: Use custom frequency CSV instead of cached/computed")
    parser.add_argument("--overwrite-freq", action="store_true",
                       help="Force recomputation of class frequencies")
    parser.add_argument("--dry-run", action="store_true")

    # K-fold
    parser.add_argument("--k-folds", type=int, default=None)
    parser.add_argument("--val-split", type=float, default=0.1)

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    config = SamplerConfig.from_args(args)

    # Ensure output directory exists (including auto-created subdirectory)
    config.get_output_path().mkdir(parents=True, exist_ok=True)

    # Build and run pipeline
    pipeline = SamplingPipeline.from_config(config)
    ctx = pipeline.run(config)

    # Compute inverse frequency weights
    weights_df = compute_inverse_frequency_weights(ctx.class_freq_df)
    
    # Save weights to CSV (unless dry-run)
    if not config.dry_run and len(weights_df) > 0:
        save_inverse_frequency_weights(
            weights_df, 
            config.get_output_path(), 
            config.get_base_name()
        )

    # Print reports
    print_class_frequency_report(ctx.class_freq_df, weights_df)
    print_selection_report(ctx)
    print_balanced_data_report(ctx)


if __name__ == "__main__":
    main()
