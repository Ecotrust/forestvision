import os
import logging
import inspect
import pydoc
import warnings
from typing import Any, Dict, List, Optional, Type, Union, Callable

from tqdm import tqdm
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader
from torchgeo.datasets import IntersectionDataset, stack_samples, RasterDataset
from torchgeo.datasets.errors import DatasetNotFoundError
from torchvision.transforms import v2

from forestvision.datamodules.clouddatamodule import CloudDataModule
from forestvision.samplers import TileGeoSampler
from forestvision.datasets import GPDFeatureCollection

@dataclass
class DatasetConfig:
    """Configuration for a single dataset within the DataModule.
    
    Attributes:
        dataset_class: The class of the dataset to instantiate.
        path_template: String template for the data path, supports {root}, {year}, and {stage}.
        bands: List of bands to load.
        transforms: Optional transforms to apply to THIS dataset before intersection.
        kwargs: Additional arguments for the dataset constructor.
    """
    dataset_class: Type[RasterDataset]
    path_template: str
    bands: List[str]
    transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)

class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

class BaseGeoDataModule(CloudDataModule):
    """Universal GeoDataModule for ForestVision.
    
    Handles multiple input and target datasets through IntersectionDatasets,
    supports per-dataset transforms, and manages statistics.
    """
    
    def __init__(
        self,
        root: str,
        year: int,
        input_configs: List[Union[DatasetConfig, Dict[str, Any]]],
        target_configs: List[Union[DatasetConfig, Dict[str, Any]]],
        batch_size: int = 1,
        patch_size: Union[int, tuple[int, int]] = 256,
        num_workers: int = 0,
        train_tiles_path: Optional[str] = None,
        val_tiles_path: Optional[str] = None,
        test_tiles_path: Optional[str] = None,
        predict_tiles_path: Optional[str] = None,
        stats_path: Optional[str] = None,
        hparams: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        # We pass a dummy dataset_class to satisfy GeoDataModule, 
        # as we handle multiple datasets via IntersectionDataset internally.
        super().__init__(dataset_class=RasterDataset, **kwargs)
        self.root = root
        self.year = year
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.train_tiles_path = train_tiles_path
        self.val_tiles_path = val_tiles_path
        self.test_tiles_path = test_tiles_path
        self.predict_tiles_path = predict_tiles_path
        self.stats_path = os.path.join(self.root, stats_path) if stats_path else None
        self.hparams_dict = hparams or {}

        # Convert dict configs to DatasetConfig objects if necessary
        self.input_configs = [self._to_config(c) for c in input_configs]
        self.target_configs = [self._to_config(c) for c in target_configs]

        # Initialize placeholders
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None
        
        self.train_tiles = None
        self.val_tiles = None
        self.test_tiles = None
        self.predict_tiles = None

    def _to_config(self, config: Union[DatasetConfig, Dict[str, Any]]) -> DatasetConfig:
        if isinstance(config, DatasetConfig):
            return config

        # Recursively instantiate class_path/init_args dictionaries
        def instantiate(obj):
            if isinstance(obj, dict) and "class_path" in obj:
                cls_path = obj["class_path"]
                cls = pydoc.locate(cls_path)
                if cls is None:
                    raise ImportError(f"Could not locate class: {cls_path}")

                init_args = obj.get("init_args", {})
                # Recursively instantiate arguments
                resolved_args = {k: instantiate(v) for k, v in init_args.items()}
                return cls(**resolved_args)
            elif isinstance(obj, list):
                return [instantiate(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: instantiate(v) for k, v in obj.items()}
            return obj

        # Handle dict-based config (e.g. from YAML/CLI)
        cls = config.get("dataset_class")
        if isinstance(cls, str):
            resolved_cls = pydoc.locate(cls)
            if resolved_cls is None:
                raise ImportError(f"Could not locate dataset class: {cls}")
            config["dataset_class"] = resolved_cls

        if "transforms" in config:
            config["transforms"] = instantiate(config["transforms"])

        return DatasetConfig(**config)

    @staticmethod
    def _serialize_stats(stats_dict: Dict[str, Any]) -> Dict[str, Any]:
        serialized = {}
        for key, value in stats_dict.items():
            if isinstance(value, torch.Tensor):
                serialized[key] = value.tolist()
            elif isinstance(value, dict):
                serialized[key] = BaseGeoDataModule._serialize_stats(value)
            else:
                serialized[key] = value
        return serialized

    @staticmethod
    def _deserialize_stats(stats_dict: Dict[str, Any]) -> Dict[str, Any]:
        deserialized = {}
        for key, value in stats_dict.items():
            if isinstance(value, list):
                deserialized[key] = torch.tensor(value)
            elif isinstance(value, dict):
                deserialized[key] = BaseGeoDataModule._deserialize_stats(value)
            else:
                deserialized[key] = value
        return deserialized

    def _instantiate_combined_dataset(
        self, configs: List[DatasetConfig], stage: str, roi: Optional[Any] = None
    ) -> IntersectionDataset:
        datasets = []
        for cfg in configs:
            path = cfg.path_template.format(root=self.root, year=self.year, stage=stage)

            if not os.path.isabs(path):
                path = os.path.join(self.root, path)
            
            # Prepare arguments, prioritizing explicit year if required by dataset
            kwargs = cfg.kwargs.copy()
            sig = inspect.signature(cfg.dataset_class.__init__)
            
            # Handle path/paths variation in datasets
            if "paths" in sig.parameters:
                kwargs["paths"] = path
            elif "path" in sig.parameters:
                kwargs["path"] = path
                
            if "year" in sig.parameters and "year" not in kwargs:
                kwargs["year"] = self.year

            if "roi" in sig.parameters:
                kwargs["roi"] = roi
            
            try:
                # Mute UserWarnings from TorchGeo during instantiation if fallback available
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning, message=".*relevant files.*")
                    ds = cfg.dataset_class(
                        bands=cfg.bands,
                        transforms=cfg.transforms,
                        **kwargs
                    )
                datasets.append(ds)
            except (DatasetNotFoundError, UserWarning, Exception) as e:
                # Resolve path error types
                is_path_error = "Dataset not found" in str(e) or "relevant files" in str(e)
                
                if is_path_error:
                    # Attempt fallback: remove path/paths and let dataset defaults take over
                    fallback_kwargs = kwargs.copy()
                    fallback_kwargs.pop("path", None)
                    fallback_kwargs.pop("paths", None)
                    
                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=UserWarning, message=".*relevant files.*")
                            ds = cfg.dataset_class(
                                bands=cfg.bands,
                                transforms=cfg.transforms,
                                **fallback_kwargs
                            )
                        datasets.append(ds)
                        logging.info(f"Fell back to default directory for {cfg.dataset_class.__name__}")
                        continue
                    except Exception as fallback_e:
                        logging.warning(f"Could not instantiate {cfg.dataset_class.__name__} at {path} or default.")

                    # If it's not downloadable and fallback failed, skip it
                    is_gee = "GEE" in cfg.dataset_class.__name__
                    if not is_gee:
                         continue
                else:
                    raise

        if not datasets:
            return None
            
        combined = datasets[0]
        for ds in datasets[1:]:
            combined &= ds
        return combined

    def setup(self, stage: str, year: Optional[int] = None) -> None:
        if stage in ["fit", "validate", "prepare"]:
            if self.train_tiles_path:
                self.train_tiles = GPDFeatureCollection(os.path.join(self.root, self.train_tiles_path))
            if self.val_tiles_path:
                self.val_tiles = GPDFeatureCollection(os.path.join(self.root, self.val_tiles_path))
            
            train_roi = self.train_tiles.bounds if self.train_tiles else None
            val_roi = self.val_tiles.bounds if self.val_tiles else None

            input_ds = self._instantiate_combined_dataset(self.input_configs, "training", train_roi)
            target_ds = self._instantiate_combined_dataset(self.target_configs, "training", train_roi)
            if input_ds and target_ds:
                self.train_dataset = target_ds & input_ds

            val_input_ds = self._instantiate_combined_dataset(self.input_configs, "validation", val_roi)
            val_target_ds = self._instantiate_combined_dataset(self.target_configs, "validation", val_roi)
            if val_input_ds and val_target_ds:
                self.val_dataset = val_target_ds & val_input_ds

        elif stage == "test":
            if self.test_tiles_path:
                self.test_tiles = GPDFeatureCollection(os.path.join(self.root, self.test_tiles_path))
            test_roi = self.test_tiles.bounds if self.test_tiles else None
            
            input_ds = self._instantiate_combined_dataset(self.input_configs, "test", test_roi)
            target_ds = self._instantiate_combined_dataset(self.target_configs, "test", test_roi)
            if input_ds and target_ds:
                self.test_dataset = target_ds & input_ds

        elif stage == "predict":
            p_year = year or self.year
            if self.predict_tiles_path:
                self.predict_tiles = GPDFeatureCollection(os.path.join(self.root, self.predict_tiles_path))
            predict_roi = self.predict_tiles.bounds if self.predict_tiles else None
            
            # Predict might use different templates or year handling
            self.predict_dataset = self._instantiate_combined_dataset(self.input_configs, "predict", predict_roi)

    def _collate_fn(self, batch):
        collated = stack_samples(batch)
        # Standard ForestVision batch structure
        return {
            "mask": collated.get("mask"),
            "image": collated.get("image"),
            "crs": collated.get("crs"),
            "bounds": collated.get("bbox"),
        }

    def train_dataloader(self) -> DataLoader:
        sampler = TileGeoSampler(self.train_dataset, self.train_tiles.data, shuffle=True)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        sampler = TileGeoSampler(self.val_dataset, self.val_tiles.data, shuffle=False)
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        sampler = TileGeoSampler(self.test_dataset, self.test_tiles.data, shuffle=False)
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )

    def prepare_data(self, overwrite: bool = False, download_all_bands: bool = False) -> None:
        """Download data for all configured datasets.
        
        Args:
            overwrite: If True, overwrite existing files during download
            download_all_bands: If True, download all available bands (dataset.all_bands)
                               instead of just the selected bands (dataset.bands). This is useful
                               for downloading the full dataset once, then experimenting with
                               different band combinations without re-downloading.
        """
        # Try to setup all stages, but catch errors if ROI/tiles are missing for some
        for stage in ["prepare", "test"]:
            try:
                self.setup(stage)
            except Exception as e:
                logging.warning(f"Could not initialize stage {stage} during prepare_data: {e}")
        
        from torch.utils.data import DataLoader

        # Add tqdm-aware logging handler
        tqdm_handler = TqdmLoggingHandler()
        tqdm_handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        root_logger = logging.getLogger()
        root_logger.addHandler(tqdm_handler)

        def download_dataset(dataset, tiles, download_all_bands=False):
            if not tiles:
                return
                
            # Most GEE datasets in forestvision have a private _download or public download attribute
            # that triggers download during __getitem__
            if hasattr(dataset, "download"):
                dataset.download = True
            elif hasattr(dataset, "_download"):
                dataset._download = True
            else:
                return # Not a downloadable dataset
            
            # Store original state to restore after download
            original_transforms = dataset.transforms
            original_bands = dataset.bands
            
            # Temporarily switch to all_bands if requested
            if download_all_bands and hasattr(dataset, 'all_bands'):
                dataset.bands = dataset.all_bands
                logging.info(
                    f"Downloading ALL bands for {dataset.__class__.__name__}: "
                    f"{dataset.all_bands} (normally uses {original_bands})"
                )
            
            # Disable transforms during download
            dataset.transforms = None

            try:
                # Use a sampler to iterate over tiles and trigger downloads
                sampler = TileGeoSampler(dataset, tiles.data)
                
                # Create a simple collate function that doesn't try to stack
                # (we only care about triggering downloads, not batching the data)
                def download_collate(batch):
                    return batch  # Just return the list, don't stack
                
                dataloader = DataLoader(
                    dataset,
                    sampler=sampler,
                    batch_size=1,  # Process one at a time to avoid stacking issues
                    num_workers=10,
                    collate_fn=download_collate,
                )

                logging.info(f"Downloading {dataset.__class__.__name__} to {getattr(dataset, 'path', getattr(dataset, 'paths', 'unknown'))}")
                for _ in tqdm(dataloader, desc=f"Downloading {dataset.__class__.__name__}", leave=False):
                    pass
            finally:
                # Restore original state
                dataset.transforms = original_transforms
                dataset.bands = original_bands

        def get_all_datasets(dataset):
            if isinstance(dataset, IntersectionDataset):
                res = []
                if hasattr(dataset, "datasets"):
                    for ds in dataset.datasets:
                        res.extend(get_all_datasets(ds))
                else:
                    res.extend(get_all_datasets(dataset.dataset1))
                    res.extend(get_all_datasets(dataset.dataset2))
                return res
            return [dataset]

        try:
            # Prepare a list of (dataset, tiles) to download
            tasks = []
            if self.train_dataset and self.train_tiles:
                tasks.append((self.train_dataset, self.train_tiles))
            if self.val_dataset and self.val_tiles:
                tasks.append((self.val_dataset, self.val_tiles))
            if self.test_dataset and self.test_tiles:
                tasks.append((self.test_dataset, self.test_tiles))

            # Track unique paths and class to avoid redundant downloads
            seen_keys = set()

            for top_ds, tiles in tasks:
                for ds in get_all_datasets(top_ds):
                    # We use path + class to distinguish datasets
                    path = getattr(ds, 'path', getattr(ds, 'paths', None))
                    ds_name = ds.__class__.__name__
                    key = (ds_name, str(path))
                    if path and key not in seen_keys:
                        download_dataset(ds, tiles, download_all_bands=download_all_bands)
                        seen_keys.add(key)
        finally:
            # Always remove the handler
            root_logger.removeHandler(tqdm_handler)

    def cleanup(self) -> None:
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None
