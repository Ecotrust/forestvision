import os
import logging
import time
from typing import Any, Dict, List, Optional, Union
import warnings

import ee
from dotenv import load_dotenv

import torch
from torchvision.transforms import v2

from forestvision.datamodules.base import BaseGeoDataModule, DatasetConfig
from forestvision.datasets import (
    GNNForestAttr,
    GEESentinel2,
    GEE3Dep,
)
from forestvision.transforms import (
    Normalize,
    ReplaceNodataVal,
)
from forestvision.deploy import AnyRasterDataset

# Mute warnings
os.environ["CPL_LOG"] = "/dev/null"
warnings.filterwarnings("ignore")

# Load from .env file
load_dotenv()
GEE_PROJECT_NAME = os.getenv("GEE_PROJECT_NAME")
TARGET_PATH = os.getenv("TARGET_PATH")

# fmt: off
REMAP = {
    # Forest type 9
    966: 9,   968: 9,   969: 9,   975: 9,
    
    # Forest type 10
    125: 10,  129: 10,  134: 10,  136: 10,  170: 10,  188: 10,  189: 10,
    190: 10,  191: 10,  192: 10,  193: 10,  196: 10,  197: 10,  200: 10,
    202: 10,  204: 10,  206: 10,  210: 10,  211: 10,  215: 10,  218: 10,
    219: 10,  220: 10,  221: 10,  231: 10,  234: 10,  238: 10,  254: 10,
    256: 10,  259: 10,  260: 10,  261: 10,  262: 10,  263: 10,  265: 10,
    266: 10,  269: 10,  270: 10,  271: 10,  272: 10,  282: 10,  284: 10,
    286: 10,  293: 10,  306: 10,  319: 10,  322: 10,  368: 10,  426: 10,
    427: 10,  488: 10,  498: 10,  518: 10,  535: 10,  545: 10,  546: 10,
    581: 10,  598: 10,  601: 10,  607: 10,  614: 10,  619: 10,  645: 10,
    654: 10,  717: 10,  815: 10,  818: 10,  855: 10,  886: 10,  890: 10,
    891: 10,  893: 10,  896: 10,  897: 10,  898: 10,  899: 10,  900: 10,
    902: 10,  906: 10,
    
    # Forest type 11
    33: 11,   112: 11,  113: 11,  115: 11,  123: 11,  124: 11,  126: 11,
    127: 11,  128: 11,  130: 11,  131: 11,  132: 11,  133: 11,  135: 11,
    148: 11,  165: 11,  177: 11,  182: 11,  184: 11,  186: 11,  199: 11,
    346: 11,  425: 11,  543: 11,  565: 11,  568: 11,  569: 11,  571: 11,
    580: 11,  597: 11,  599: 11,  600: 11,  602: 11,  603: 11,  604: 11,
    605: 11,  606: 11,  608: 11,  610: 11,  621: 11,  622: 11,  624: 11,
    625: 11,  634: 11,  647: 11,  653: 11,  667: 11,  668: 11,  669: 11,
    670: 11,  672: 11,  673: 11,  674: 11,  676: 11,  677: 11,  679: 11,
    681: 11,  683: 11,  684: 11,  685: 11,  689: 11,  690: 11,  698: 11,
    701: 11,  702: 11,  703: 11,  704: 11,  705: 11,  706: 11,  708: 11,
    714: 11,  718: 11,  719: 11,  720: 11,  721: 11,  723: 11,  726: 11,
    728: 11,  743: 11,  748: 11,  752: 11,  767: 11,  776: 11,  791: 11,
    794: 11,  836: 11,  838: 11,  839: 11,  840: 11,  841: 11,  844: 11,
    852: 11,  887: 11,  888: 11,  889: 11,  892: 11,  894: 11,  895: 11,
    901: 11,  907: 11,  908: 11,  909: 11,  910: 11,  911: 11,  915: 11,
    916: 11,  917: 11,  918: 11,  919: 11,  921: 11,  926: 11,  930: 11,
    931: 11,  932: 11,  935: 11,  942: 11,  943: 11,  944: 11,  945: 11,
    946: 11,  947: 11,  948: 11,

    # Set NF to nodata
    # 0: -2147483648

}
# fmt: on

# GNNForestAttr.remap_dict.update(REMAP)


class ClimateNA(AnyRasterDataset):
    all_bands = ["AHM", "MAP", "TD"]
    _cmap = None
    rgb_bands = ["AHM", "MAP", "TD"]
    instrument = "ClimateNA"
    nodata = -9999


class ForTypesDataModule(BaseGeoDataModule):
    """LightningDataModule for forest type classification, refactored to use BaseGeoDataModule."""

    def __init__(
        self,
        root: str,
        year: int,
        stats_path: Optional[str] = None,
        batch_size: int = 1,
        patch_size: Union[int, tuple[int, int]] = 256,
        num_workers: int = 0,
        train_tiles_path: Optional[str] = None,
        val_tiles_path: Optional[str] = None,
        test_tiles_path: Optional[str] = None,
        predict_tiles_path: Optional[str] = None,
        input_datasets: Optional[List[Dict[str, Any]]] = None,
        target_datasets: Optional[List[Dict[str, Any]]] = None,
        ee_project: Optional[str] = None,
        hparams: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        # Initialize Earth Engine
        ee_project = ee_project or GEE_PROJECT_NAME
        try:
            ee.Initialize(project=ee_project)
            logging.info(f"Earth Engine initialized with project: {ee_project}")
        except Exception as e:
            logging.warning(f"Failed to initialize Earth Engine: {e}")

        # Define default datasets if none provided
        if input_datasets is None:
            input_datasets = [
                {
                    "dataset_class": GEESentinel2,
                    "path_template": "{stage}/geesentinel2/{year}",
                    "bands": GEESentinel2.all_bands,
                    "kwargs": {"download": True}
                },
                {
                    "dataset_class": GEE3Dep,
                    "path_template": "{stage}/gee3dep/{year}",
                    "bands": ["elevation"],
                    "kwargs": {"download": True, "res": 10}
                },
                {
                    "dataset_class": ClimateNA,
                    "path_template": "training/climatena",
                    "bands": ClimateNA.all_bands,
                    "kwargs": {"glob": "*.tif", "res": 10, "is_image": True, "nodata": -9999}
                }
            ]

        if target_datasets is None:
            target_path = os.getenv("TARGET_PATH", "targets")
            target_datasets = [
                {
                    "dataset_class": GNNForestAttr,
                    "path_template": target_path,
                    "bands": ["fortypba", "cancov", "qmd_dom", "ba_ge_3"],
                    "kwargs": {"res": 10}
                }
            ]

        super().__init__(
            root=root,
            year=year,
            input_configs=input_datasets,
            target_configs=target_datasets,
            batch_size=batch_size,
            patch_size=patch_size,
            num_workers=num_workers,
            train_tiles_path=train_tiles_path,
            val_tiles_path=val_tiles_path,
            test_tiles_path=test_tiles_path,
            predict_tiles_path=predict_tiles_path,
            stats_path=stats_path,
            hparams=hparams,
            **kwargs,
        )

        # Stat loading and transform setup (kept for compatibility with current workflow)
        self.input_stats = None
        self.target_stats = None
        self._transforms_applied = False

    def setup(self, stage: str, year: Optional[int] = None) -> None:
        """Setup datasets and load stats from JSON file."""
        # Load stats BEFORE calling super().setup() so they're available during dataset creation
        if self.stats_path and os.path.exists(self.stats_path):
            self._load_stats_from_file()
        else:
            logging.warning(f"Stats file not found at {self.stats_path}. Using identity normalization (mean=0, std=1).")
            self._set_identity_stats()
        
        super().setup(stage, year)

    def _set_identity_stats(self):
        """Set identity stats (mean=0, std=1) for all datasets when no stats file exists."""
        # For input datasets, we need to know the channel count
        # We'll set identity stats - they can be overridden later
        for cfg in self.input_configs:
            # Use placeholder identity stats - actual channel count determined during data loading
            cfg.mean = [0.0]  # Will be expanded to match actual channels
            cfg.std = [1.0]
        
        for cfg in self.target_configs:
            cfg.mean = [0.0]
            cfg.std = [1.0]

    def _load_stats_from_file(self):
        """Load JSON stats and populate transforms."""
        import json

        try:
            logging.info(f"Loading statistics from {self.stats_path}")

            with open(self.stats_path, "r") as f:
                stats = json.load(f)

            # 1. Aggregated Input Stats
            all_means = []
            all_stds = []
            for entry in stats.get("input_stats", []):
                all_means.extend(entry.get("mean", []))
                all_stds.extend(entry.get("std", []))

            self.input_stats = {
                "mean": torch.tensor(all_means),
                "std": torch.tensor(all_stds),
            }

            # 2. Aggregated Target Stats
            target_json = stats.get("target_stats", {})
            self.target_stats = {
                "mean": torch.tensor(target_json.get("mean", [0.0])),
                "std": torch.tensor(target_json.get("std", [1.0])),
            }

            # 3. Populate top-level combined transforms with these stats
            # This also handles subsetting if SelectBands is present
            if self.input_transforms:
                self.input_transforms = self._populate_normalize_stats(
                    self.input_transforms, all_means, all_stds
                )
                
                # Update aggregated stats to match final selected bands for model hparams
                # Search for Normalize transform in the chain (handles both dict and objects)
                def get_norm_stats(obj):
                    if isinstance(obj, dict):
                        if "Normalize" in obj.get("class_path", ""):
                            ia = obj.get("init_args", {})
                            return ia.get("mean"), ia.get("std")
                        for v in obj.values():
                            res = get_norm_stats(v)
                            if res: return res
                    elif isinstance(obj, list):
                        for item in obj:
                            res = get_norm_stats(item)
                            if res: return res
                    elif hasattr(obj, "mean") and hasattr(obj, "std"):
                        return obj.mean, obj.std
                    elif hasattr(obj, "transforms"):
                        for t in obj.transforms:
                            res = get_norm_stats(t)
                            if res: return res
                    return None
                
                stats_pair = get_norm_stats(self.input_transforms)
                if stats_pair and stats_pair[0] is not None:
                    self.input_stats["mean"] = torch.tensor(stats_pair[0])
                    self.input_stats["std"] = torch.tensor(stats_pair[1])

            if self.target_transforms:
                self.target_transforms = self._populate_normalize_stats(
                    self.target_transforms, target_json.get("mean"), target_json.get("std")
                )
                
                # Same for target stats
                stats_pair = get_norm_stats(self.target_transforms)
                if stats_pair and stats_pair[0] is not None:
                    self.target_stats["mean"] = torch.tensor(stats_pair[0])
                    self.target_stats["std"] = torch.tensor(stats_pair[1])

            self.hparams["input_stats"] = self._serialize_stats(self.input_stats)
            self.hparams["target_stats"] = self._serialize_stats(self.target_stats)

            logging.info(f"Successfully loaded statistics: {len(all_means)} input channels")
            
        except Exception as e:
            logging.error(f"Failed to load statistics: {e}")
            raise

    def setup_transforms(self):
        """Hook for future use. Normalization is now handled per-dataset via transforms config."""
        pass

    def train_dataloader(self):
        return super().train_dataloader()

    def val_dataloader(self):
        return super().val_dataloader()
