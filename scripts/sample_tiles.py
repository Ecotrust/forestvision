#!/usr/bin/env python3
"""
Generate tile collection for forest type classification.

1. Download state boundaries
2. Generate tile grid
3. Compute class frequencies from GNN data
4. Select subset using configurable strategy
5. Export tiles (single file or k-fold splits)

Usage:
    python scripts/sample_tiles.py --output-path data/fortypba/tiles/
    python scripts/sample_tiles.py --output-path data/fortypba/tiles/ --k-folds 5
"""

import argparse
import hashlib
import logging
import os
import sys
import zipfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import box
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader
from torchgeo.datasets import stack_samples
from tqdm import tqdm

from forestvision.datasets import GNNForestAttr
from forestvision.samplers import TileGeoSampler
from forestvision.samplers.utils import roi_to_tiles

# Suppress annoying logs
os.environ["CPL_LOG"] = "/dev/null"
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

STATE_FIPS = {"Oregon": "41", "Washington": "53"}
CENSUS_URL = "https://www2.census.gov/geo/tiger/TIGER2024/STATE/tl_2024_us_state.zip"


@dataclass(frozen=True)
class SamplerConfig:
    output_path: Path
    name: str = None
    gnn_path: str = "data/datasets/gnn"
    boundary_cache: str = "data/boundaries"
    states: list[str] = field(default_factory=lambda: ["Oregon", "Washington"])
    tile_size: int = 128
    tile_res: int = 10
    stride: int = None
    sample_size: int = 8000
    random_state: int = 42
    batch_size: int = 64
    num_workers: int = 10
    input_tiles: str = None
    frequency_file: str = None
    overwrite_freq: bool = False
    dry_run: bool = False
    k_folds: int = None
    val_split: float = 0.15
    test_split: float = 0.15
    balance_strategy: str = "none"
    min_samples: int = 10
    inference: bool = False
    max_nodata: float = 0.3

    def __post_init__(self):
        if self.tile_size <= 0:
            raise ValueError("tile_size must be > 0")
        if self.stride is not None and self.stride <= 0:
            raise ValueError("stride must be > 0")
        if self.k_folds is not None and self.k_folds < 2:
            raise ValueError("k_folds must be >= 2")
        if not 0.0 <= self.max_nodata <= 1.0:
            raise ValueError("max_nodata must be between 0.0 and 1.0")

    def out_dir(self):
        return self.output_path / f"tiles_{self.tile_size}x{self.tile_size}"

    def base(self):
        """Base name for universal files (all tiles, frequencies)."""
        return (
            self.name or f"all_{self.tile_size}x{self.tile_size}_{self.tile_res}m"
        )

    def split_base(self):
        """Base name for split-specific files (train, val, folds, weights, report).
        
        Uses the balance strategy as prefix to distinguish between different
        sampling strategies applied to the same tile set.
        """
        if self.name:
            return self.name
        prefix = self.balance_strategy if self.balance_strategy != "none" else "random"
        return f"{prefix}_{self.tile_size}x{self.tile_size}_{self.tile_res}m"


def get_strat_labels(df, n, v):
    df = df.copy()
    df["bin"] = pd.cut(
        df["num_odf_classes"],
        bins=[0, 2, 4, 6, 8, float("inf")],
        labels=["1-2", "3-4", "5-6", "7-8", "9+"],
    )
    df["strat"] = df["dominant_odf_class"].astype(str) + "_" + df["bin"].astype(str)
    c = df["strat"].value_counts()
    rare = c[c < max(2, int(1 / v), n or 0)].index.tolist()
    if rare:
        df["strat"] = df["strat"].apply(lambda x: "other" if x in rare else x)
    return df


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output-path", required=True)
    p.add_argument("--name")
    p.add_argument("--input-tiles")
    p.add_argument("--frequency-file")
    p.add_argument("--gnn-path", default="data/datasets/gnn")
    p.add_argument("--boundary-cache", default="data/boundaries")
    p.add_argument("--states", nargs="+", default=["Oregon", "Washington"])
    for a, v in [
        ("tile-size", 128),
        ("tile-res", 10),
        ("sample-size", 8000),
        ("random-state", 42),
        ("batch-size", 64),
        ("num-workers", 10),
        ("min-samples", 10),
    ]:
        p.add_argument("--" + a, type=int, default=v)
    p.add_argument(
        "--sampling-strategy",
        default="none",
        choices=["none", "equal", "proportional", "inverse_freq", "capped"],
    )
    p.add_argument("--overwrite-freq", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--k-folds", type=int)
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--test-split", type=float, default=0.15)
    p.add_argument(
        "--inference",
        action="store_true",
        help="Create inference tiles overlapping GNN ROI without computing frequencies",
    )
    p.add_argument(
        "--max-nodata",
        type=float,
        default=0.3,
        help="Maximum fraction of nodata pixels allowed per tile (0.0-1.0, default: 0.3)",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Stride for tile generation (default: None, meaning stride equals tile-size for no overlap)",
    )

    args = p.parse_args()
    # Map sampling_strategy to balance_strategy for SamplerConfig
    args.balance_strategy = args.sampling_strategy
    kw = {
        k: v for k, v in vars(args).items() 
        if k.replace('-', '_') in SamplerConfig.__dataclass_fields__
    }
    kw["output_path"] = Path(kw["output_path"])
    cfg = SamplerConfig(**kw)
    out, base, split_base = cfg.out_dir(), cfg.base(), cfg.split_base()

    # 1. Load Tiles
    if cfg.input_tiles:
        tiles = gpd.read_file(cfg.input_tiles)
    else:
        tp = out / f"{base}_all.geojson"
        if tp.exists():
            tiles = gpd.read_file(str(tp))
        else:
            c = Path(cfg.boundary_cache)
            c.mkdir(parents=True, exist_ok=True)
            s = c / "tl_2024_us_state.shp"
            if not s.exists():
                r = requests.get(CENSUS_URL, timeout=120)
                zipfile.ZipFile(BytesIO(r.content)).extractall(c)
            gdf = gpd.read_file(s)
            fps = [STATE_FIPS.get(st, st) for st in cfg.states]
            b = gdf[gdf["NAME"].isin(cfg.states) | gdf["STATEFP"].isin(fps)].to_crs(
                "EPSG:5070"
            )
            from torchgeo.datasets import BoundingBox

            tb = b.total_bounds
            roi = BoundingBox(tb[0], tb[2], tb[1], tb[3], 0, sys.maxsize)
            tbounds = roi_to_tiles(
                roi=roi, size=cfg.tile_size, res=cfg.tile_res, stride=cfg.stride
            )
            tiles = gpd.GeoDataFrame(
                {"geometry": [box(*bt) for bt in tbounds]}, crs="EPSG:5070"
            )
            state_union = b.union_all()
            tiles = tiles[tiles.geometry.intersects(state_union)].copy()
            if not cfg.dry_run:
                out.mkdir(parents=True, exist_ok=True)
                tiles.to_file(tp, driver="GeoJSON")
    if "geohash" not in tiles.columns:
        tiles["geohash"] = [
            hashlib.md5(x.bounds.__repr__().encode()).hexdigest()[:10]
            for x in tiles.geometry
        ]
    if cfg.dry_run:
        tiles = tiles.head(10)

    # 2. Frequencies (skip in inference mode)
    fp = out / f"{base}_frequencies.csv"
    if cfg.inference:
        # Load GNN dataset to get bounds, filter tiles to GNN ROI
        gnn = GNNForestAttr(
            paths=cfg.gnn_path, bands=["fortypba"], remap=False, crs="EPSG:5070", res=10
        )
        tiles = tiles[
            tiles.geometry.intersects(
                box(gnn.bounds.minx, gnn.bounds.miny, gnn.bounds.maxx, gnn.bounds.maxy)
            )
        ].copy()
        freqs = pd.DataFrame()  # Empty frequencies for inference
        logger.info(f"Inference mode: {len(tiles)} tiles overlap GNN ROI")
    elif cfg.frequency_file and Path(cfg.frequency_file).exists():
        freqs = pd.read_csv(cfg.frequency_file)
    elif not cfg.overwrite_freq and fp.exists():
        freqs = pd.read_csv(fp)
    else:
        gnn = GNNForestAttr(
            paths=cfg.gnn_path, bands=["fortypba"], remap=False, crs="EPSG:5070", res=10
        )
        mask_tg = tiles[
            tiles.geometry.intersects(
                box(gnn.bounds.minx, gnn.bounds.miny, gnn.bounds.maxx, gnn.bounds.maxy)
            )
        ]
        if mask_tg.empty:
            freqs = pd.DataFrame(
                columns=["geohash", "odf_class", "gnn_counts", "frequency"]
            )
        else:
            dl = DataLoader(
                gnn,
                sampler=TileGeoSampler(gnn, mask_tg),
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                collate_fn=stack_samples,
            )
            recs = []
            for b_idx in tqdm(dl, desc="Frequencies"):
                hs = [
                    hashlib.md5(
                        str((bx.minx, bx.miny, bx.maxx, bx.maxy)).encode()
                    ).hexdigest()[:10]
                    for bx in b_idx["bounds"]
                ]
                for i, h in enumerate(hs):
                    mask = b_idx["mask"][i]
                    total_pixels = mask.numel()
                    
                    # Count valid pixels (non-nodata)
                    valid_mask = (mask != -2147483648) & (mask != -1)
                    valid_pixels = valid_mask.sum().item()
                    nodata_pixels = total_pixels - valid_pixels
                    nodata_fraction = nodata_pixels / total_pixels if total_pixels > 0 else 0.0
                    
                    # Skip tiles with too much nodata
                    if nodata_fraction > cfg.max_nodata:
                        continue
                    
                    # Count class frequencies for valid pixels only
                    cl_u, cl_c = mask[valid_mask].unique(return_counts=True)
                    for c, n in zip(cl_u.tolist(), cl_c.tolist()):
                        recs.append({
                            "geohash": h, 
                            "gnn_class": c, 
                            "gnn_counts": n,
                            "total_pixels": valid_pixels,
                            "nodata_pixels": nodata_pixels,
                            "nodata_fraction": nodata_fraction
                        })
            freqs = pd.DataFrame(recs)
            if not freqs.empty:
                freqs = freqs.merge(
                    freqs.groupby("geohash")["gnn_counts"]
                    .sum()
                    .reset_index(name="total"),
                    on="geohash",
                )
                freqs["frequency"] = freqs["gnn_counts"] / freqs["total"]
                freqs["odf_class"] = freqs["gnn_class"].replace(gnn.remap_dict)
            if not cfg.dry_run:
                out.mkdir(parents=True, exist_ok=True)
                freqs.to_csv(fp, index=False)

    # 3. Select
    selected = None
    if not freqs.empty:
        el = (
            freqs.groupby("geohash")
            .agg(
                num_odf_classes=("odf_class", "nunique"),
                dominant_odf_class=(
                    "odf_class",
                    lambda x: freqs.loc[
                        freqs.loc[x.index, "frequency"].idxmax(), "odf_class"
                    ],
                ),
            )
            .reset_index()
        )
        s = cfg.balance_strategy
        # Filter out only invalid classes for analysis, but keep nodata info
        valid_classes = freqs[~freqs["odf_class"].isin([-2147483648, -1])].copy()
        
        if s == "none":
            selected = el.sample(
                n=min(cfg.sample_size, len(el)), random_state=cfg.random_state
            )
        elif s == "capped":
            # Group by geohash and odf_class, sum counts
            summed = valid_classes.groupby(["geohash", "odf_class"])["gnn_counts"].sum().reset_index()
            # Get index of max count per geohash (dominant class per tile)
            max_idx = summed.groupby("geohash")["gnn_counts"].idxmax()
            # Use iloc for positional indexing to avoid index mismatch
            assigned = summed.iloc[max_idx]
            bal = [
                assigned[assigned["odf_class"] == c].sample(
                    n=min(len(assigned[assigned["odf_class"] == c]), cfg.sample_size),
                    random_state=cfg.random_state,
                )
                for c in valid_classes["odf_class"].unique()
                if not assigned[assigned["odf_class"] == c].empty
            ]
            selected = el[
                el["geohash"].isin({h for b_pool in bal for h in b_pool["geohash"]})
            ].copy()
        else:
            dist = el["dominant_odf_class"].value_counts().sort_index()
            if s == "equal":
                base_q, rem = divmod(cfg.sample_size, len(dist))
                q = {
                    c: max(cfg.min_samples, base_q + (1 if i < rem else 0))
                    for i, c in enumerate(dist.index)
                }
            elif s == "proportional":
                q = {
                    c: max(cfg.min_samples, int(cfg.sample_size * ct / dist.sum()))
                    for c, ct in dist.items()
                }
            else:  # inverse_freq
                w_d = 1.0 / dist
                w_d /= w_d.sum()
                q = {
                    c: max(cfg.min_samples, int(cfg.sample_size * wt))
                    for c, wt in w_d.items()
                }
            sel = [
                el[el["dominant_odf_class"] == c].sample(
                    n=min(qv, len(el[el["dominant_odf_class"] == c])),
                    random_state=cfg.random_state,
                )
                for c, qv in q.items()
                if not el[el["dominant_odf_class"] == c].empty
            ]
            selected = pd.concat(sel, ignore_index=True) if sel else pd.DataFrame()

    # 4. Export & Report
    train_h, val_h, test_h = [], [], []
    if cfg.inference:
        # Inference mode: export all tiles without splitting
        if not tiles.empty and not cfg.dry_run:
            out.mkdir(parents=True, exist_ok=True)
            tiles.to_file(out / f"{split_base}_inference.geojson", driver="GeoJSON")
            logger.info(f"Exported {len(tiles)} inference tiles")
    elif selected is not None and not selected.empty and not cfg.dry_run:

        def save_split(df_split, n):
            h_l = df_split["geohash"].tolist()
            tiles[tiles["geohash"].isin(h_l)].to_file(
                out / f"{split_base}_{n}.geojson", driver="GeoJSON"
            )
            freqs[freqs["geohash"].isin(h_l)].to_csv(
                out / f"{split_base}_{n}.csv", index=False
            )

        if cfg.k_folds:
            # First split out test set from selected data
            df_st = get_strat_labels(selected, cfg.k_folds, cfg.val_split)
            tr_val, test = train_test_split(
                df_st,
                test_size=cfg.test_split,
                stratify=df_st["strat"],
                random_state=cfg.random_state,
            )
            test_h = test["geohash"].tolist()
            save_split(test, "test")
            # Now split remaining into train/val for k-fold
            tr_i, vl_i = next(
                StratifiedKFold(
                    n_splits=int(1 / cfg.val_split),
                    shuffle=True,
                    random_state=cfg.random_state,
                ).split(tr_val, tr_val["strat"])
            )
            tr_p, vl_d = tr_val.iloc[tr_i], tr_val.iloc[vl_i]
            train_h, val_h = tr_p["geohash"].tolist(), vl_d["geohash"].tolist()
            save_split(vl_d, "val")
            skf = StratifiedKFold(
                n_splits=cfg.k_folds, shuffle=True, random_state=cfg.random_state
            )
            for i, (_, f_i) in enumerate(skf.split(tr_p, tr_p["strat"]), 1):
                save_split(tr_p.iloc[f_i], f"fold_{i}_train")
        else:
            df_st = get_strat_labels(selected, 0, cfg.val_split)
            # First split out test set
            tr_val, test = train_test_split(
                df_st,
                test_size=cfg.test_split,
                stratify=df_st["strat"],
                random_state=cfg.random_state,
            )
            test_h = test["geohash"].tolist()
            save_split(test, "test")
            # Split remaining into train/val
            # Adjust val_split to account for remaining proportion
            adjusted_val_split = cfg.val_split / (1 - cfg.test_split)
            tr_p, vl_d = train_test_split(
                tr_val,
                test_size=adjusted_val_split,
                stratify=tr_val["strat"],
                random_state=cfg.random_state,
            )
            train_h, val_h = tr_p["geohash"].tolist(), vl_d["geohash"].tolist()
            save_split(tr_p, "train")
            save_split(vl_d, "val")

    if cfg.inference:
        # Inference mode report
        print(f"\nINFERENCE REPORT: {split_base}")
        print("- Mode:                Inference")
        print(f"- GNN Path:            {cfg.gnn_path}")
        print(f"- Total tiles:         {len(tiles):,}")
        print(f"- Output:              {out / f'{split_base}_inference.geojson'}")
        md = [
            f"# Report: {split_base}",
            "\n## Summary",
            "- Mode: Inference",
            f"- GNN Path: `{cfg.gnn_path}`",
            f"- Total Tiles: {len(tiles):,}",
            f"- Output: `{out / f'{split_base}_inference.geojson'}`",
        ]
        if not cfg.dry_run:
            with open(out / f"{split_base}_report.md", "w") as f:
                f.write("\n".join(md))
        print("")
    elif not freqs.empty:
        valid = freqs[~freqs["odf_class"].isin([-2147483648, -1])]
        t_u, f_u = (
            cfg.input_tiles or out / f"{base}_all.geojson",
            cfg.frequency_file or out / f"{base}_frequencies.csv",
        )
        print(f"\nSAMPLING REPORT: {split_base}")
        print(f"- Base tile file:      {t_u}")
        print(f"- Base frequency CSV:  {f_u}")
        print(f"- Total tiles in CSV:  {freqs['geohash'].nunique():,}")
        print(f"- Sampling strategy:   {cfg.balance_strategy}")
        print(f"- Max nodata allowed:  {cfg.max_nodata:.1%}")
        
        # Calculate nodata statistics
        if 'nodata_fraction' in freqs.columns:
            nodata_stats = freqs['nodata_fraction'].describe()
            print(f"- Mean nodata:         {nodata_stats['mean']:.1%}")
            print(f"- Median nodata:       {nodata_stats['50%']:.1%}")
            print(f"- Max nodata:          {nodata_stats['max']:.1%}")
        
        selected_count = len(selected) if selected is not None else 0
        md = [
            f"# Report: {split_base}",
            "\n## Summary",
            f"- Strategy: {cfg.balance_strategy}",
            f"- Max Nodata Allowed: {cfg.max_nodata:.1%}",
            f"- Total Tiles Input: {freqs['geohash'].nunique():,}",
            f"- Selected Tiles: {selected_count:,}",
            f"- Base Tile File: `{t_u}`",
            f"- Base Freq File: `{f_u}`",
            f"- Output: `{out}`",
        ]
        if train_h and val_h:
            tp, vp, tep = (
                valid[valid["geohash"].isin(train_h)]
                .groupby("odf_class")["gnn_counts"]
                .sum(),
                valid[valid["geohash"].isin(val_h)]
                .groupby("odf_class")["gnn_counts"]
                .sum(),
                valid[valid["geohash"].isin(test_h)]
                .groupby("odf_class")["gnn_counts"]
                .sum(),
            )
            ts, vs, tes = tp.sum(), vp.sum(), tep.sum()
            print("- Frequency stats for train/val/test split (Pixels %):")
            md.extend(
                ["\n## Split Stats", "| ODF | Train % | Val % | Test % |", "|:---|---:|---:|---:|"]
            )
            for c in sorted(set(tp.index) | set(vp.index) | set(tep.index)):
                tr_pct = 100 * tp.get(c, 0) / ts if ts else 0
                vl_pct = 100 * vp.get(c, 0) / vs if vs else 0
                te_pct = 100 * tep.get(c, 0) / tes if tes else 0
                print(f"  ODF {int(c):3d}: Train {tr_pct:5.1f}%, Val {vl_pct:5.1f}%, Test {te_pct:5.1f}%")
                md.append(f"| {int(c)} | {tr_pct:.1f}% | {vl_pct:.1f}% | {te_pct:.1f}% |")
        if not cfg.dry_run:
            dist_v = valid.groupby("odf_class")["gnn_counts"].sum().sort_index()
            w_df = pd.DataFrame(
                {"pixels": dist_v, "freq": dist_v / dist_v.sum() if dist_v.sum() else 0}
            )
            if not w_df.empty:
                inv_f = 1.0 / w_df["freq"].replace(0, float("inf"))
                w_df["weight"] = (inv_f / inv_f.sum() * len(w_df)).round(2)
                w_df.to_csv(out / f"{split_base}_weights.csv")
                md.extend(
                    [
                        "\n## Distribution",
                        "| ODF | Pixels | Freq | Weight |",
                        "|:---|---:|---:|---:|",
                    ]
                )
                for c, r in w_df.iterrows():
                    md.append(
                        f"| {int(c)} | {int(r.pixels):,} | {r.freq:.4f} | {r.weight:.2f} |"
                    )
            with open(out / f"{split_base}_report.md", "w") as f:
                f.write("\n".join(md))
        print("")


if __name__ == "__main__":
    import traceback

    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
