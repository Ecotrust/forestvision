#!/usr/bin/env python3
"""
Generic script to prepare data for forestvision dataloaders.

This script:
1. Accepts a config.yaml file as argument
2. Extracts data preparation parameters from the config
3. Downloads required data
4. Computes and saves statistics for specified sample keys

Usage:
    python scripts/prepare_data.py data/fortypba/conf/fortypes.yaml

    # Compute only image statistics
    python scripts/prepare_data.py data/fortypba/conf/fortypes.yaml --on-keys image

    # Compute only mask statistics
    python scripts/prepare_data.py data/fortypba/conf/fortypes.yaml --on-keys mask

    # Skip download (data already exists)
    python scripts/prepare_data.py data/fortypba/conf/fortypes.yaml --skip-download
"""

import os
import argparse
import logging
import importlib
from pathlib import Path
from typing import List, Dict, Any

import yaml
import ee
import dotenv 

import torch
from torchgeo.datasets import GeoDataset, IntersectionDataset
from geopandas import GeoDataFrame

from tqdm import tqdm
from forestvision.samplers import TileGeoSampler
from forestvision.datasets import DatasetStats

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
    """Load configuration from YAML file.

    Args:
        config_path: Path to config YAML file

    Returns:
        Dictionary containing configuration
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def extract_datamodule_args(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract datamodule arguments from config."""
    if "data" not in config:
        raise ValueError("Config must contain 'data' section")

    if "class_path" not in config["data"]:
        raise ValueError("Config must contain 'data.class_path'")

    class_path = config["data"]["class_path"]

    if "init_args" not in config["data"]:
        raise ValueError("Config must contain 'data.init_args' section")

    data_args = config["data"]["init_args"].copy()
    data_args["class_path"] = class_path

    required_keys = ["root", "year", "train_tiles_path"]
    for key in required_keys:
        if key not in data_args:
            raise ValueError(f"Missing required config key: {key}")

    return data_args


def import_class(class_path: str):
    """Dynamically import a class from a module path.

    Args:
        class_path: Full path to class (e.g., 'forestvision.datamodules.ForTypesDataModule')

    Returns:
        The imported class
    """
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_stats(
    dataset: GeoDataset,
    tiles: GeoDataFrame,
    nodata: int | None = None,
    overwrite: bool = False,
) -> dict:
    """Compute dataset statistics including mean and standard deviation."""
    sampler = TileGeoSampler(dataset, tiles=tiles)

    # Disable transforms temporarily to compute raw stats
    original_transforms = dataset.transforms
    dataset.transforms = None

    try:
        # Check actual tensor shape from a sample to ensure consistency with DatasetStats.
        # This accounts for any active transforms in the tree.
        sample = dataset[next(iter(sampler))]
        key = "image" if dataset.is_image else "mask"
        channels = sample[key].shape[0]

        if nodata is None:
            nodata = dataset.nodata

        stats_calculator = DatasetStats(
            dataset,
            sampler,
            path=None,
            batch_size=5,
            num_workers=5,
            channels=channels,
            nodata=nodata,
            overwrite=overwrite,
        )
        return stats_calculator.compute()
    finally:
        # Restore transforms
        dataset.transforms = original_transforms


def prepare_data(
    config_path: str,
    on_keys: List[str] = None,
    skip_download: bool = False,
    overwrite: bool = False,
    target_identity_channels: List[int] = None,
    download_all_bands: bool = False,
) -> None:
    """Prepare data by downloading and computing statistics."""
    tqdm_handler = TqdmLoggingHandler()
    tqdm_handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(tqdm_handler)
    root_logger.setLevel(logging.INFO)

    if on_keys is None:
        on_keys = ["image", "mask"]

    config = load_config(config_path)
    data_args = extract_datamodule_args(config)

    datamodule_class = import_class(data_args["class_path"])

    root = data_args["root"]
    year = data_args["year"]
    train_tiles_path = data_args["train_tiles_path"]
    stats_path = data_args.get("stats_path", f"train_stats_{year}.pt")

    if not os.path.isabs(stats_path):
        stats_path = os.path.join(root, stats_path)

    if os.path.exists(stats_path) and not overwrite:
        print(f"Statistics file already exists at {stats_path}")
        return

    output_dir = os.path.dirname(stats_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for split in ["training", "validation", "test"]:
        split_dir = os.path.join(root, split)
        if not os.path.exists(split_dir):
            os.makedirs(split_dir, exist_ok=True)

    datamodule = datamodule_class(**data_args)

    if not skip_download:
        # Lower logging level to WARNING during download phase to mute INFO:root:Downloading GEE...
        original_level = root_logger.getEffectiveLevel()
        root_logger.setLevel(logging.WARNING)
        
        datamodule.setup("fit")
        datamodule.prepare_data(
            overwrite=overwrite,
            download_all_bands=download_all_bands
        )
        
        # Restore logging level
        root_logger.setLevel(original_level)
    else:
        datamodule.setup("prepare")

    def extract_datasets(dataset):
        if isinstance(dataset, IntersectionDataset):
            result = []
            if hasattr(dataset, "datasets"):
                for ds in dataset.datasets:
                    result.extend(extract_datasets(ds))
            else:
                result.extend(extract_datasets(dataset.dataset1))
                result.extend(extract_datasets(dataset.dataset2))
            return result
        else:
            return [dataset]

    all_datasets = extract_datasets(datamodule.train_dataset)
    train_tiles = datamodule.train_tiles

    input_datasets = []
    target_dataset = None

    target_class_names = []
    for cfg in datamodule.target_configs:
        cls = cfg.dataset_class
        name = cls.__name__ if hasattr(cls, "__name__") else str(cls).split(".")[-1]
        target_class_names.append(name)

    for ds in all_datasets:
        ds_name = ds.__class__.__name__
        if ds_name in target_class_names:
            target_dataset = ds
        else:
            input_datasets.append(ds)

    if not target_dataset:
        raise ValueError("Could not identify target dataset in tree.")

    def create_placeholder_stats():
        return {
            "mean": torch.tensor([0.0]),
            "std": torch.tensor([1.0]),
            "min": torch.tensor([0.0]),
            "max": torch.tensor([1.0]),
            "nodata": 0,
            "nodata_pixels": "0 (0.00%)",
            "sample_size": 0,
        }

    all_input_stats = []
    for i, ds in enumerate(input_datasets):
        if "image" in on_keys:
            input_stats = get_stats(
                ds,
                train_tiles.data,
                overwrite=overwrite,
            )
        else:
            input_stats = create_placeholder_stats()
        
        # Add metadata to track which config this corresponds to
        input_stats['dataset_class'] = ds.__class__.__name__
        input_stats['config_index'] = i
        input_stats['path'] = str(getattr(ds, 'path', getattr(ds, 'paths', 'unknown')))
        all_input_stats.append(input_stats)

    if "mask" in on_keys:
        target_stats = get_stats(
            target_dataset,
            train_tiles.data,
            nodata=getattr(target_dataset, "nodata", None),
            overwrite=overwrite,
        )

        # Apply identity normalization (mean=0, std=1) to specified channels
        # (e.g. classification labels) so they remain unscaled in the datamodule.
        if target_identity_channels:
            num_channels = len(target_stats["mean"])
            for idx in target_identity_channels:
                if 0 <= idx < num_channels:
                    logging.info(f"Setting identity stats for target channel {idx}")
                    target_stats["mean"][idx] = 0.0
                    target_stats["std"][idx] = 1.0
                else:
                    logging.warning(f"Target identity index {idx} out of range (max={num_channels-1})")
    else:
        target_stats = create_placeholder_stats()

    _serialize_stats = getattr(datamodule_class, "_serialize_stats", None)
    if _serialize_stats is None:
        raise AttributeError(
            f"Datamodule class {data_args['class_path']} must have a _serialize_stats method"
        )

    serialized_input_stats = _serialize_stats(all_input_stats)
    serialized_target_stats = _serialize_stats(target_stats)

    stats_dict = {
        "input_stats": serialized_input_stats,
        "target_stats": serialized_target_stats,
        "year": year,
        "inputs_class": [ds.__class__.__name__ for ds in input_datasets],
        "target_class": target_dataset.__class__.__name__,
    }

    torch.save(stats_dict, stats_path)

    root_logger.removeHandler(tqdm_handler)

    print(f"\n{'='*50}")
    print(f"Data Preparation Summary")
    print(f"{'='*50}")
    print(f"Stats saved to: {stats_path}")
    print(f"Year: {year}")
    
    if "image" in on_keys:
        print(f"\nInput Statistics (Combined Splits):")
        for stats in all_input_stats:
            ds_name = stats.get('dataset_class', 'Unknown')
            ds_path = stats.get('path', 'Unknown')
            print(f"\n  Dataset: {ds_name} (Index: {stats.get('config_index', '?')})")
            print(f"    Path: {ds_path}")
            print(f"    Bands: {len(stats['mean'])}")
            print(f"    Mean: {stats['mean'].tolist()}")
            print(f"    Std:  {stats['std'].tolist()}")
            print(f"    Min:  {stats['min'].tolist()}")
            print(f"    Max:  {stats['max'].tolist()}")
            print(f"    NoData Pixels: {stats['nodata_pixels']}")
            print(f"    Sample Size: {stats['sample_size']}")
    
    if "mask" in on_keys:
        print(f"\nTarget Statistics ({target_dataset.__class__.__name__}):")
        print(f"    Mean: {target_stats['mean'].tolist()}")
        print(f"    Std:  {target_stats['std'].tolist()}")
        print(f"    Min:  {target_stats['min'].tolist()}")
        print(f"    Max:  {target_stats['max'].tolist()}")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare data for forestvision dataloaders"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument(
        "--on-keys",
        nargs="+",
        choices=["image", "mask"],
        default=["image"],
        help="Which sample types to compute statistics for (default: image)",
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
        "--target-identity-channels",
        nargs="+",
        type=int,
        default=[0],
        help="Target channel indices to set to identity normalization (mean=0, std=1). Default: [0]",
    )
    parser.add_argument(
        "--download-all-bands",
        action="store_true",
        help="Download all available bands instead of just the selected bands from config",
    )

    args = parser.parse_args()

    prepare_data(
        config_path=args.config,
        on_keys=args.on_keys,
        skip_download=args.skip_download,
        overwrite=args.overwrite,
        target_identity_channels=args.target_identity_channels,
        download_all_bands=args.download_all_bands,
    )


if __name__ == "__main__":
    ee_project = dotenv.load_dotenv('.')
    ee.Authenticate()
    ee.Initialize(project=ee_project, opt_url='https://earthengine-highvolume.googleapis.com')
    main()
