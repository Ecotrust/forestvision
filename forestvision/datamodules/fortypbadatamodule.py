import os
import logging
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
warnings.filterwarnings('ignore')

# Load from .env file
load_dotenv()
GEE_PROJECT_NAME = os.getenv("GEE_PROJECT_NAME")
TARGET_PATH = os.getenv("TARGET_PATH")

REMAP = {
    33: 11, 112: 11, 128: 11, 165: 11, 182: 11, 198: 11, 306: 10, 346: 11,
    113: 11, 115: 11, 123: 11, 124: 11, 125: 10, 126: 11, 127: 11, 129: 10,
    130: 11, 131: 11, 132: 11, 133: 11, 134: 10, 135: 11, 136: 10, 148: 11,
    170: 10, 177: 11, 184: 11, 186: 11, 188: 10, 189: 10,
    190: 10, 191: 10, 192: 10, 193: 10, 196: 10, 197: 10, 199: 11,
    200: 10, 202: 10, 204: 10, 206: 10, 210: 10, 211: 10, 215: 10, 218: 10,
    219: 10, 220: 10, 221: 10, 231: 10, 234: 10, 238: 10, 254: 10, 256: 10,
    259: 10, 260: 10, 261: 10, 262: 10, 263: 10, 265: 10, 266: 10, 269: 10,
    270: 10, 271: 10, 272: 10, 282: 10, 284: 10, 286: 10, 293: 10, 319: 10,
    322: 10, 368: 10, 425: 11, 426: 10, 427: 10, 488: 10, 498: 10,
    518: 10, 535: 10, 543: 11, 545: 10, 546: 10, 565: 11, 568: 11, 569: 11,
    571: 11, 580: 11, 581: 10, 597: 11, 598: 10, 599: 11, 600: 11, 601: 10,
    602: 11, 603: 11, 604: 11, 605: 11, 606: 11, 607: 10, 608: 11, 610: 11,
    614: 10, 619: 10, 621: 11, 622: 11, 624: 11, 625: 11, 634: 11, 645: 10,
    647: 11, 653: 11, 654: 10, 667: 11, 668: 11, 669: 11, 670: 11, 672: 11,
    673: 11, 674: 11, 676: 11, 677: 11, 679: 11, 681: 11, 683: 11, 684: 11,
    685: 11, 689: 11, 690: 11, 698: 11, 701: 11, 702: 11, 703: 11, 704: 11,
    705: 11, 706: 11, 708: 11, 714: 11, 717: 10, 718: 11, 719: 11, 720: 11,
    721: 11, 723: 11, 726: 11, 728: 11, 743: 11, 748: 11, 752: 11, 767: 11,
    776: 11, 791: 11, 794: 11, 815: 10, 818: 10, 836: 11, 838: 11, 839: 11,
    840: 11, 841: 11, 844: 11, 852: 11, 855: 10, 886: 10, 887: 11, 888: 11,
    889: 11, 890: 10, 891: 10, 892: 11, 893: 10, 894: 11, 895: 11, 896: 10,
    897: 10, 898: 10, 899: 10, 900: 10, 901: 11, 902: 10, 906: 10, 907: 11,
    908: 11, 909: 11, 910: 11, 911: 11, 915: 11, 916: 11, 917: 11, 918: 11,
    919: 11, 921: 11, 926: 11, 930: 11, 931: 11, 932: 11, 935: 11, 942: 11,
    943: 11, 944: 11, 945: 11, 946: 11, 947: 11, 948: 11, 966: 9, 968: 9,
    969: 9, 975: 9,
}

GNNForestAttr.remap_dict.update(REMAP)

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
            **kwargs
        )
        
        # Stat loading and transform setup (kept for compatibility with current workflow)
        self.input_stats = None
        self.target_stats = None
        self._transforms_applied = False

    def setup(self, stage: str, year: Optional[int] = None) -> None:
        super().setup(stage, year)
        
        # Load stats if available
        if self.input_stats is None and self.stats_path and os.path.exists(self.stats_path):
            self._load_stats_from_file()
            
        if stage != "prepare":
            self.setup_transforms()

    def _load_stats_from_file(self):
        try:
            logging.info(f"Loading statistics from {self.stats_path}")
            stats_dict = torch.load(self.stats_path)
            if "input_stats" in stats_dict and "target_stats" in stats_dict:
                raw_input_stats = stats_dict["input_stats"]
                
                # Handle list-based stats (new format)
                if isinstance(raw_input_stats, list):
                    all_means, all_stds = [], []
                    
                    # Match stats by config index
                    for i, cfg in enumerate(self.input_configs):
                        if i >= len(raw_input_stats):
                            logging.error(f"Config index {i} exceeds available stats (len={len(raw_input_stats)})")
                            continue
                        
                        ds_stats = self._deserialize_stats(raw_input_stats[i])
                        ds_name = cfg.dataset_class.__name__
                        
                        # Verify class names match for safety
                        stored_class = ds_stats.get('dataset_class', 'Unknown')
                        if stored_class != ds_name:
                            logging.warning(
                                f"Config index {i}: Expected {ds_name}, found {stored_class} in stats. "
                                "Using stats anyway (order-based matching)."
                            )
                        
                        # Subset stats if a custom band list is requested
                        if len(ds_stats.get("mean", [])) != len(cfg.bands):
                            try:
                                # Use all_bands from class or instance metadata
                                band_ref = getattr(cfg.dataset_class, "all_bands", None)
                                indices = [band_ref.index(b) for b in cfg.bands]
                                all_means.extend([ds_stats["mean"][i] for i in indices])
                                all_stds.extend([ds_stats["std"][i] for i in indices])
                                logging.info(f"Subsetted stats for {ds_name} (config index {i})")
                            except (AttributeError, ValueError, IndexError) as e:
                                logging.warning(f"Could not subset stats for {ds_name}: {e}. Using all stats.")
                                all_means.extend(ds_stats.get("mean", []))
                                all_stds.extend(ds_stats.get("std", []))
                        else:
                            all_means.extend(ds_stats.get("mean", []))
                            all_stds.extend(ds_stats.get("std", []))
                    
                    if all_means:
                        self.input_stats = {
                            "mean": torch.tensor(all_means),
                            "std": torch.tensor(all_stds)
                        }
                        logging.info(f"Successfully loaded list-based input statistics ({len(all_means)} channels)")
                    else:
                        logging.error("No input statistics could be loaded from list format")
                        return
                else:
                    # Legacy dict-based format - reject it
                    raise ValueError(
                        "Stats file uses legacy dict-based format. "
                        "Please regenerate stats with: "
                        "python scripts/prepare_data.py --config <your_config.yaml> --overwrite"
                    )
                
                # Update datamodule properties for trainer access
                self.hparams["input_stats"] = self._serialize_stats(self.input_stats)
                
                # Handle target stats
                raw_target_stats = self._deserialize_stats(stats_dict["target_stats"])
                
                # Align target stats with requested target bands
                all_target_means, all_target_stds = [], []
                for cfg in self.target_configs:
                    target_band_ref = getattr(cfg.dataset_class, "all_bands", None)
                    if target_band_ref:
                        try:
                            indices = [target_band_ref.index(b) for b in cfg.bands]
                            all_target_means.extend([raw_target_stats["mean"][i] for i in indices])
                            all_target_stds.extend([raw_target_stats["std"][i] for i in indices])
                            logging.info(f"Subsetted target stats for {cfg.dataset_class.__name__}")
                        except (ValueError, IndexError):
                            all_target_means.extend(raw_target_stats["mean"])
                            all_target_stds.extend(raw_target_stats["std"])
                    else:
                        all_target_means.extend(raw_target_stats["mean"])
                        all_target_stds.extend(raw_target_stats["std"])

                if all_target_means:
                    self.target_stats = {
                        "mean": torch.tensor(all_target_means),
                        "std": torch.tensor(all_target_stds)
                    }
                else:
                    self.target_stats = raw_target_stats
                
                self.hparams["target_stats"] = self._serialize_stats(self.target_stats)
                logging.info("Successfully loaded and aligned statistics from file")
                
        except Exception as e:
            logging.error(f"Failed to load statistics: {e}")
            raise

    def setup_transforms(self):
        if self.input_stats is None or self.target_stats is None:
            return

        # Target transforms applied to the final combined dataset
        target_transforms = v2.Compose([
            ReplaceNodataVal(nodata=-2147483648, new_nodata=-1),
            Normalize(mean=self.target_stats["mean"], std=self.target_stats["std"], on_key="mask", nodata=-1),
        ])

        # We apply target transforms to the combined training/val datasets
        if self.train_dataset:
            self.train_dataset.transforms = target_transforms
        if self.val_dataset:
            self.val_dataset.transforms = target_transforms

        # For input datasets, we apply normalization PER-DATASET to avoid shape mismatches
        # during broadcast in IntersectionDataset.
        stats_file = torch.load(self.stats_path)
        input_stats_list = stats_file["input_stats"]
        
        def apply_input_transforms(dataset):
            from torchgeo.datasets import IntersectionDataset
            
            # Track which config index we're at as we traverse
            idx_counter = [0]
            
            def traverse_and_apply(ds):
                if isinstance(ds, IntersectionDataset):
                    if hasattr(ds, "datasets"):
                        for child in ds.datasets:
                            traverse_and_apply(child)
                    else:
                        traverse_and_apply(ds.dataset1)
                        traverse_and_apply(ds.dataset2)
                else:
                    # Leaf dataset - check if it's a target dataset
                    is_target = any(isinstance(ds, cfg.dataset_class) for cfg in self.target_configs)
                    
                    if not is_target and idx_counter[0] < len(input_stats_list):
                        # This is an input dataset - apply its stats
                        stats = self._deserialize_stats(input_stats_list[idx_counter[0]])
                        ds_name = ds.__class__.__name__
                        
                        # Verify class name matches for safety
                        stored_class = stats.get('dataset_class', 'Unknown')
                        if stored_class != ds_name:
                            logging.warning(
                                f"Config index {idx_counter[0]}: Expected {ds_name}, "
                                f"found {stored_class} in stats"
                            )
                        
                        # Ensure mean/std match the number of base bands currently requested.
                        if len(stats["mean"]) == len(ds.bands):
                            m = torch.tensor(stats["mean"])
                            s = torch.tensor(stats["std"])
                        else:
                            # Mismatch: try to subset stats based on band names
                            # Prioritize instance metadata (all_bands) which might be synced from collection.json
                            band_ref = getattr(ds, "all_bands", None) or getattr(ds.__class__, "all_bands", None)
                            if band_ref:
                                try:
                                    # Map currently requested bands to indices in the statistical baseline (all_bands)
                                    indices = [band_ref.index(b) for b in ds.bands]
                                    
                                    # Safety check: ensure indices are within the stats array bounds
                                    max_stat_idx = len(stats["mean"]) - 1
                                    valid_indices = [i for i in indices if i <= max_stat_idx]
                                    
                                    if len(valid_indices) != len(indices):
                                        missing = [ds.bands[i] for i, idx in enumerate(indices) if idx > max_stat_idx]
                                        logging.warning(f"Statistics for {ds_name} missing bands: {missing}")
                                        idx_counter[0] += 1
                                        return

                                    m = torch.tensor(stats["mean"])[indices]
                                    s = torch.tensor(stats["std"])[indices]
                                    logging.info(f"Successfully subsetted statistics for {ds_name} using metadata")
                                except (ValueError, IndexError) as e:
                                    logging.warning(f"Could not subset stats for {ds_name}: {e}")
                                    idx_counter[0] += 1
                                    return
                            else:
                                logging.warning(
                                    f"Skipping normalization for {ds_name}: Stats channel count ({len(stats['mean'])}) "
                                    f"does not match requested bands ({len(ds.bands)}) and no metadata available."
                                )
                                idx_counter[0] += 1
                                return

                        # Create normalization transform
                        norm = Normalize(mean=m, std=s, on_key="image")
                        
                        # Use a robust wrapper to handle extra bands added by transforms (like NDVI)
                        def robust_norm(sample):
                            img = sample["image"]
                            num_stats_channels = len(m)
                            
                            if img.shape[-3] > num_stats_channels:
                                # Normalize only base bands, leave extra bands (indices) as they are
                                base_img = img[..., :num_stats_channels, :, :]
                                extra_img = img[..., num_stats_channels:, :, :]
                                
                                sample["image"] = base_img
                                sample = norm(sample)
                                
                                sample["image"] = torch.cat([sample["image"], extra_img], dim=-3)
                            else:
                                sample = norm(sample)
                            return sample

                        # Preserve existing transforms and append normalization
                        existing_transforms = ds.transforms
                        if existing_transforms:
                            ds.transforms = v2.Compose([existing_transforms, robust_norm])
                        else:
                            ds.transforms = robust_norm
                            
                        logging.info(f"Applied robust per-dataset normalization to {ds_name} (config index {idx_counter[0]}, {len(m)} base bands)")
                        idx_counter[0] += 1
            
            traverse_and_apply(dataset)

        # Apply to all input datasets in the tree
        for stage_attr in ["train_dataset", "val_dataset"]:
            ds = getattr(self, stage_attr, None)
            if ds:
                apply_input_transforms(ds)

    def train_dataloader(self):
        if self.input_stats is not None and not self._transforms_applied:
            self.setup_transforms()
            self._transforms_applied = True
        return super().train_dataloader()

    def val_dataloader(self):
        if self.input_stats is not None and not self._transforms_applied:
            self.setup_transforms()
            self._transforms_applied = True
        return super().val_dataloader()
