#!/usr/bin/env python3
"""
Generic script to prepare data for forestvision dataloaders.

This script:
1. Accepts a config.yaml file as argument
2. Extracts data preparation parameters from the config
3. Downloads required data
4. Computes and saves statistics for each dataset individually using DatasetStats

Usage:
    python scripts/prepare_data.py --config data/fortypba/conf/fortypes.yaml

    # Compute only image statistics
    python scripts/prepare_data.py --config data/fortypba/conf/fortypes.yaml --on-keys image

    # Compute only mask statistics  
    python scripts/prepare_data.py --config data/fortypba/conf/fortypes.yaml --on-keys mask

    # Skip download (data already exists)
    python scripts/prepare_data.py --config data/fortypba/conf/fortypes.yaml --skip-download

    # Set identity channels per dataset using semicolon separator
    # Format: "dataset1_ch1 dataset1_ch2; dataset2_ch1; dataset3_ch1 dataset3_ch2"
    python scripts/prepare_data.py --config data/fortypba/conf/fortypes.yaml \
        --input-identity-channels "10 11; 0; 1 2 3" \
        --target-identity-channels "0"

    # Example with 3 input datasets:
    # - Dataset 1 (GEESentinel2): identity channels 10, 11 (e.g., NDVI, NDWI)
    # - Dataset 2 (GEE3Dep): identity channel 0
    # - Dataset 3 (ClimateNA): identity channels 1, 2, 3
    python scripts/prepare_data.py --config config.yaml \
        --input-identity-channels "10 11; 0; 1 2 3"
    
"""

import os
import argparse
import logging
import pydoc
import inspect
from typing import List, Dict, Any, Optional

import yaml
import json
import ee
import dotenv 

import torch
from tqdm import tqdm

from forestvision.samplers import TileGeoSampler
from forestvision.datasets.utils import DatasetStats
from forestvision.datasets import GPDFeatureCollection

os.environ["CPL_LOG"] = "/dev/null"


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_identity_channels(transforms_cfg: Any, key: str = "image") -> List[int]:
    """Extract identity_channels for a specific key from transform configuration."""
    if not transforms_cfg:
        return []

    def find_identity(obj):
        if isinstance(obj, dict):
            cls_path = obj.get("class_path", "")
            if "Normalize" in cls_path:
                init_args = obj.get("init_args", {})
                if init_args.get("on_key") == key:
                    return init_args.get("identity_channels", [])
            
            # Recurse into all dictionary values
            for v in obj.values():
                res = find_identity(v)
                if res:
                    return res
        elif isinstance(obj, list):
            for item in obj:
                res = find_identity(item)
                if res:
                    return res
        return []

    return find_identity(transforms_cfg)


def resolve_class(class_path: str):
    """Dynamically resolve a class from a module path string."""
    cls = pydoc.locate(class_path)
    if cls is None:
        raise ImportError(f"Could not locate class: {class_path}")
    return cls


def instantiate_transforms(transforms_cfg: Any, dataset: Any = None) -> Any:
    """Instantiate transforms from configuration, skipping Normalize."""
    if transforms_cfg is None:
        return None

    def instantiate(obj, ds=None):
        if isinstance(obj, dict) and "class_path" in obj:
            cls_path = obj["class_path"]
            # Skip Normalize during preparation
            if cls_path == "forestvision.transforms.Normalize":
                return None
                
            cls = pydoc.locate(cls_path)
            if cls is None:
                raise ImportError(f"Could not locate class: {cls_path}")

            init_args = obj.get("init_args", {}).copy()
            
            # Check if the class accepts a 'dataset' argument
            sig = inspect.signature(cls.__init__)
            if "dataset" in sig.parameters and ds is not None:
                init_args["dataset"] = ds

            # Recursively instantiate arguments
            resolved_args = {k: instantiate(v, ds=ds) for k, v in init_args.items()}
            # Filter out None results from recursion (skipped transforms)
            resolved_args = {k: v for k, v in resolved_args.items() if v is not None}
            
            return cls(**resolved_args)
        elif isinstance(obj, list):
            items = [instantiate(item, ds=ds) for item in obj]
            # Filter out None (skipped transforms)
            items = [item for item in items if item is not None]
            if not items:
                return None
            # If it was a list intended for Compose, we'll need to handle it
            return items
        elif isinstance(obj, dict):
            return {k: instantiate(v, ds=ds) for k, v in obj.items()}
        return obj

    res = instantiate(transforms_cfg, ds=dataset)
    
    # If the top level was a Compose (often the case), and we filtered its list
    if isinstance(res, list) and len(res) > 0:
        from torchvision.transforms import v2
        return v2.Compose(res)
    
    return res


def instantiate_dataset(
    cfg_dict: Dict[str, Any],
    root: str,
    year: int,
    stage: str = "training",
    roi: Optional[Any] = None,
    download: bool = False,
):
    """Instantiate a dataset from configuration dictionary."""
    cls_path = cfg_dict.get("dataset_class")
    if isinstance(cls_path, str):
        dataset_class = resolve_class(cls_path)
    else:
        dataset_class = cls_path

    path_template = cfg_dict.get("path_template", "")
    path = path_template.format(root=root, year=year, stage=stage)
    if not os.path.isabs(path):
        path = os.path.join(root, path)

    bands = cfg_dict.get("bands", [])
    kwargs = cfg_dict.get("kwargs", {}).copy()

    # Handle constructor arguments
    sig = inspect.signature(dataset_class.__init__)
    if "paths" in sig.parameters:
        kwargs["paths"] = path
    elif "path" in sig.parameters:
        kwargs["path"] = path
        
    if "year" in sig.parameters and "year" not in kwargs:
        kwargs["year"] = year

    if "roi" in sig.parameters:
        kwargs["roi"] = roi
    
    if "download" in sig.parameters:
        kwargs["download"] = download

    # Create dataset instance
    ds = dataset_class(bands=bands, transforms=None, **kwargs)
    
    # Apply transforms but EXCLUDE Normalize
    # This ensures DatasetStats sees the final channels (post-Append and post-SelectBands)
    transforms_cfg = cfg_dict.get("transforms")
    if transforms_cfg:
        ds.transforms = instantiate_transforms(transforms_cfg, dataset=ds)
        
    return ds


def parse_per_dataset_identity(identity_str: Optional[str]) -> List[List[int]]:
    """
    Parse per-dataset identity channels from semicolon-separated string.
    
    Format: "10 11; 0; 1 2 3" means:
        - Dataset 1: channels 10, 11
        - Dataset 2: channel 0
        - Dataset 3: channels 1, 2, 3
    
    Args:
        identity_str: Semicolon-separated string of space-separated channel indices
        
    Returns:
        List of lists, where each inner list contains channel indices for a dataset
    """
    if not identity_str:
        return []
    
    result = []
    # Split by semicolon to get per-dataset groups
    groups = identity_str.split(";")
    
    for group in groups:
        # Strip whitespace and split by spaces to get individual indices
        indices = [int(x) for x in group.strip().split() if x.strip()]
        result.append(indices)
    
    return result


def prepare_data(
    config_path: str,
    on_keys: List[str] = None,
    skip_download: bool = False,
    overwrite: bool = False,
    download_all_bands: bool = False,
    input_identity_channels: Optional[str] = None,
    target_identity_channels: Optional[str] = None,
) -> None:
    """Prepare data by downloading and computing statistics using DatasetStats."""
    tqdm_handler = TqdmLoggingHandler()
    tqdm_handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(tqdm_handler)
    root_logger.setLevel(logging.INFO)

    if on_keys is None:
        on_keys = ["image", "mask"]

    config = load_config(config_path)
    if "data" not in config or "init_args" not in config["data"]:
        raise ValueError("Config must contain 'data.init_args' section")
    
    data_args = config["data"]["init_args"]
    root = data_args.get("root", ".")
    year = data_args.get("year")
    train_tiles_path = data_args.get("train_tiles_path")
    
    if not train_tiles_path:
        raise ValueError("Config must contain 'train_tiles_path'")
    
    train_tiles = GPDFeatureCollection(os.path.join(root, train_tiles_path))
    
    stats_path = data_args.get("stats_path", f"train_stats_{year}.json")
    if not os.path.isabs(stats_path):
        stats_path = os.path.join(root, stats_path)

    # Ensure .json extension
    if stats_path.endswith('.pt'):
        stats_path = stats_path[:-3] + '.json'

    # Check overwrite
    if os.path.exists(stats_path) and not overwrite:
        logging.info(f"Statistics file already exists at {stats_path}. Use --overwrite to recompute.")
        return
    elif os.path.exists(stats_path):
        logging.info(f"Overwriting existing stats file: {stats_path}")

    # Extract dataset configs
    input_datasets_cfg = data_args.get("input_datasets", [])
    target_datasets_cfg = data_args.get("target_datasets", [])

    all_stats = {
        "input_stats": [],
        "target_stats": {},
        "year": year,
    }

    roi = train_tiles.bounds if hasattr(train_tiles, "bounds") else None

    # Retrieve Validation Data if needed
    val_tiles_path = data_args.get("val_tiles_path")
    if not skip_download and val_tiles_path:
        logging.info("Retrieving validation data...")
        val_tiles = GPDFeatureCollection(os.path.join(root, val_tiles_path))
        val_roi = val_tiles.bounds if hasattr(val_tiles, "bounds") else None
        
        all_ds_cfgs = (input_datasets_cfg or []) + (target_datasets_cfg or [])
        for cfg in all_ds_cfgs:
            ds = instantiate_dataset(
                cfg, root, year, stage="validation", roi=val_roi, download=not skip_download
            )
            if hasattr(ds, "download") or hasattr(ds, "_download"):
                logging.info(f"Downloading validation data for {ds.__class__.__name__}...")
                sampler = TileGeoSampler(ds, val_tiles.data)
                loader = torch.utils.data.DataLoader(
                    ds, 
                    sampler=sampler, 
                    batch_size=15, 
                    num_workers=5, 
                    collate_fn=lambda x: x 
                )
                for _ in tqdm(loader, desc=f"Downloading val {ds.__class__.__name__}", leave=False):
                    pass

    # Parse per-dataset identity channels from CLI arguments
    input_identity_per_dataset = parse_per_dataset_identity(input_identity_channels)
    target_identity_per_dataset = parse_per_dataset_identity(target_identity_channels)
    
    # Validate number of datasets matches
    if input_identity_per_dataset and len(input_identity_per_dataset) != len(input_datasets_cfg):
        logging.warning(
            f"Number of input identity groups ({len(input_identity_per_dataset)}) "
            f"does not match number of input datasets ({len(input_datasets_cfg)}). "
            f"Using per-dataset mapping where available."
        )
    
    if target_identity_per_dataset and len(target_identity_per_dataset) != len(target_datasets_cfg):
        logging.warning(
            f"Number of target identity groups ({len(target_identity_per_dataset)}) "
            f"does not match number of target datasets ({len(target_datasets_cfg)}). "
            f"Using per-dataset mapping where available."
        )

    # Process Input Datasets
    if "image" in on_keys:
        logging.info("Computing statistics for input datasets...")
        for i, cfg in enumerate(input_datasets_cfg):
            ds = instantiate_dataset(cfg, root, year, roi=roi, download=not skip_download)
            
            # Get dataset name for logging
            ds_name = ds.__class__.__name__
            
            # Get identity channels for this specific dataset
            ds_identity = []
            if input_identity_per_dataset and i < len(input_identity_per_dataset):
                ds_identity = input_identity_per_dataset[i]
                if ds_identity:
                    logging.info(f"Identity channels for input dataset {ds_name}: {ds_identity}")
            
            # Download if requested
            if not skip_download and hasattr(ds, "download"):
                logging.info(f"Downloading {ds_name}...")
                sampler = TileGeoSampler(ds, train_tiles.data)
                loader = torch.utils.data.DataLoader(
                    ds, 
                    sampler=sampler, 
                    batch_size=15, 
                    num_workers=5, 
                    collate_fn=lambda x: x
                )
                for _ in tqdm(loader, desc=f"Downloading {ds_name}", leave=False):
                    pass

            # Compute stats
            sampler = TileGeoSampler(ds, train_tiles.data)
            
            # Determine actual channel count by taking a single sample
            # This accounts for appended bands from transforms
            sample = ds[next(iter(sampler))]
            data_key = "image" if not hasattr(ds, "is_image") or ds.is_image else "mask"
            actual_channels = sample[data_key].shape[0]
            logging.info(f"Actual channels after transforms: {actual_channels}")

            stats_calculator = DatasetStats(
                dataset=ds,
                sampler=sampler,
                batch_size=15,
                num_workers=5,
                channels=actual_channels,
            )
            ds_stats = stats_calculator.compute()

            # Force identity stats if requested for this dataset
            mean_list = ds_stats["mean"].tolist()
            std_list = ds_stats["std"].tolist()
            
            for idx in ds_identity:
                if 0 <= idx < len(mean_list):
                    logging.info(f"Forcing identity stats for channel {idx} in {ds_name}")
                    mean_list[idx] = 0.0
                    std_list[idx] = 1.0

            formatted_stats = {
                "dataset_class": ds_name,
                "mean": mean_list,
                "std": std_list,
                "min": ds_stats["min"].tolist(),
                "max": ds_stats["max"].tolist(),
                "config_index": i,
            }

            if ds_stats.get("nodata") is not None:
                formatted_stats["nodata_info"] = {
                    "value": ds_stats["nodata"],
                    "pixels": ds_stats["nodata_pixels"],
                }

            all_stats["input_stats"].append(formatted_stats)

    # Process Target Datasets
    if "mask" in on_keys:
        logging.info("Computing statistics for target datasets...")
        
        # Instantiate all target datasets
        target_datasets = []
        for i, cfg in enumerate(target_datasets_cfg):
            ds = instantiate_dataset(cfg, root, year, roi=roi, download=not skip_download)
            
            # Download target datasets before combining and computing stats
            if not skip_download and hasattr(ds, "download"):
                logging.info(f"Downloading target {ds.__class__.__name__}...")
                sampler = TileGeoSampler(ds, train_tiles.data)
                loader = torch.utils.data.DataLoader(
                    ds, 
                    sampler=sampler, 
                    batch_size=15, 
                    num_workers=5, 
                    collate_fn=lambda x: x
                )
                for _ in tqdm(loader, desc=f"Downloading target {ds.__class__.__name__}", leave=False):
                    pass
            
            target_datasets.append(ds)
        
        # Combine target datasets via IntersectionDataset (like datamodule does)
        if len(target_datasets) == 1:
            combined_target_ds = target_datasets[0]
        else:
            combined_target_ds = target_datasets[0]
            for ds in target_datasets[1:]:
                combined_target_ds &= ds
        
        # Apply target transforms (including CombineGNNDWMask) to get final channel count
        # BUT skip Normalize since stats haven't been computed yet
        target_transforms_cfg = data_args.get("target_transforms")
        if target_transforms_cfg:
            from forestvision.datamodules.base import BaseGeoDataModule
            # Create a temporary datamodule instance to instantiate transforms
            temp_dm = BaseGeoDataModule(
                root=root,
                year=year,
                input_configs=[],
                target_configs=[],
            )
            # Instantiate transforms but skip Normalize
            instantiated_transforms = temp_dm._instantiate_transforms(
                target_transforms_cfg, combined_target_ds
            )
            
            # Filter out Normalize transforms for stats computation
            def filter_normalize(transform):
                """Recursively filter out Normalize transforms."""
                from forestvision.transforms import Normalize
                
                if isinstance(transform, Normalize):
                    return None
                elif hasattr(transform, 'transforms'):  # Compose or similar
                    filtered = []
                    for t in transform.transforms:
                        ft = filter_normalize(t)
                        if ft is not None:
                            filtered.append(ft)
                    if not filtered:
                        return None
                    transform.transforms = filtered
                    return transform
                elif isinstance(transform, list):
                    filtered = [filter_normalize(t) for t in transform]
                    return [t for t in filtered if t is not None]
                return transform
            
            combined_target_ds.transforms = filter_normalize(instantiated_transforms)
        
        # Get combined dataset name for logging
        ds_name = "CombinedTarget"
        
        # Get identity channels - flatten all target identity channels
        ds_identity = []
        if target_identity_per_dataset:
            # Flatten all identity channel lists
            for identity_list in target_identity_per_dataset:
                ds_identity.extend(identity_list)
            if ds_identity:
                logging.info(f"Identity channels for combined target: {ds_identity}")

        sampler = TileGeoSampler(combined_target_ds, train_tiles.data)
        
        # Determine actual channel count after transforms
        sample = combined_target_ds[next(iter(sampler))]
        actual_channels = sample["mask"].shape[0]
        logging.info(f"Target channels after transforms: {actual_channels}")
        
        # Set is_image attribute for IntersectionDataset compatibility
        if not hasattr(combined_target_ds, 'is_image'):
            combined_target_ds.is_image = False
        
        # Set nodata attribute so DatasetStats can exclude nodata values
        # Use the nodata value from config (default to -2147483648)
        combined_target_ds.nodata = -2147483648
        
        stats_calculator = DatasetStats(
            dataset=combined_target_ds,
            sampler=sampler,
            batch_size=15,
            num_workers=5,
            channels=actual_channels,
        )
        ds_stats = stats_calculator.compute()

        # Force identity stats if requested
        mean_list = ds_stats["mean"].tolist()
        std_list = ds_stats["std"].tolist()
        
        for idx in ds_identity:
            if 0 <= idx < len(mean_list):
                logging.info(f"Forcing identity stats for channel {idx}")
                mean_list[idx] = 0.0
                std_list[idx] = 1.0

        target_stats = {
            "mean": mean_list,
            "std": std_list,
            "min": ds_stats["min"].tolist(),
            "max": ds_stats["max"].tolist(),
        }
        all_stats["target_stats"] = target_stats

        if ds_stats.get("nodata") is not None:
            all_stats["target_nodata_info"] = {
                "value": ds_stats["nodata"],
                "pixels": ds_stats["nodata_pixels"],
            }

    # Save to JSON
    output_dir = os.path.dirname(stats_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)

    logging.info(f"Statistics saved to: {stats_path}")

    # Print summary report
    print(f"\n{'='*60}")
    print(f"{'DATA PREPARATION SUMMARY':^60}")
    print(f"{'='*60}")
    print(f"Stats Path: {stats_path}")
    print(f"Year:       {year}")
    
    if all_stats.get("input_stats"):
        print("\nInput Datasets:")
        for entry in all_stats["input_stats"]:
            name = entry["dataset_class"]
            channels = len(entry["mean"])
            print(f"  - {name:<15} | Channels: {channels}")
            mean_str = ", ".join([f"{m:.2f}" for m in entry["mean"][:5]])
            if channels > 5:
                mean_str += " ..."
            print(f"    Mean: [{mean_str}]")
            
            if "nodata_info" in entry:
                nd = entry["nodata_info"]
                print(f"    Nodata: Value={nd['value']}, Pixels={nd['pixels']}")

    if all_stats.get("target_stats"):
        target = all_stats["target_stats"]
        channels = len(target["mean"])
        print("\nTarget Dataset:")
        print(f"  - Combined        | Channels: {channels}")
        mean_str = ", ".join([f"{m:.2f}" for m in target["mean"][:5]])
        if channels > 5:
            mean_str += " ..."
        print(f"    Mean: [{mean_str}]")
            
    print(f"{'='*60}\n")

    root_logger.removeHandler(tqdm_handler)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare data for forestvision dataloaders"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument(
        "--on-keys",
        nargs="+",
        choices=["image", "mask"],
        default=["image", "mask"],
        help="Which sample types to compute statistics for (default: image mask)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip data download (assume data already exists)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing statistics file"
    )
    parser.add_argument(
        "--download-all-bands",
        action="store_true",
        help="Download all available bands instead of just the selected bands from config",
    )
    parser.add_argument(
        "--input-identity-channels", "-iic",
        type=str,
        default=None,
        help="Per-dataset identity channels for input data (mean=0, std=1). "
             "Format: 'dataset1_ch1 ch2; dataset2_ch1; dataset3_ch1 ch2'. "
             "E.g., --input-identity-channels '10 11; 0; 1 2 3' "
             "sets identity for channels 10,11 in dataset1, channel 0 in dataset2, "
             "and channels 1,2,3 in dataset3.",
    )
    parser.add_argument(
        "--target-identity-channels", "-tic",
        type=str,
        default=None,
        help="Per-dataset identity channels for target data (mean=0, std=1). "
             "Format: 'dataset1_ch1; dataset2_ch1 ch2'. "
             "E.g., --target-identity-channels '0' sets identity for channel 0 in the first target dataset.",
    )
    args = parser.parse_args()

    # Join list to string if multiple arguments were passed (for backward compatibility)
    input_identity_str = args.input_identity_channels
    if isinstance(input_identity_str, list):
        input_identity_str = " ".join(map(str, input_identity_str))
    
    target_identity_str = args.target_identity_channels
    if isinstance(target_identity_str, list):
        target_identity_str = " ".join(map(str, target_identity_str))

    prepare_data(
        config_path=args.config,
        on_keys=args.on_keys,
        skip_download=args.skip_download,
        overwrite=args.overwrite,
        download_all_bands=args.download_all_bands,
        input_identity_channels=input_identity_str,
        target_identity_channels=target_identity_str,
    )


if __name__ == "__main__":
    # Load env for GEE
    dotenv.load_dotenv('.')
    ee_project = os.getenv("GEE_PROJECT_NAME")
    try:
        ee.Initialize(project=ee_project, opt_url='https://earthengine-highvolume.googleapis.com')
    except Exception as e:
        print(f"EE Init failed: {e}. Ensure you are authenticated.")
    main()
