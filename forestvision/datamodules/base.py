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
        mean: Optional mean values for normalization (populated from stats file).
        std: Optional std values for normalization (populated from stats file).
    """
    dataset_class: Type[RasterDataset]
    path_template: str
    bands: List[str]
    transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)
    mean: Optional[List[float]] = None
    std: Optional[List[float]] = None

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
        input_transforms: Optional[Any] = None,
        target_transforms: Optional[Any] = None,
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
        
        self.input_transforms = input_transforms
        self.target_transforms = target_transforms

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

    def _instantiate_transforms(self, transforms_config: Any, dataset: Any) -> Any:
        """Instantiate transforms with dataset reference for metadata synchronization.
        
        Args:
            transforms_config: Transform configuration (dict with class_path or already instantiated)
            dataset: Dataset instance to pass to transforms that support it
            
        Returns:
            Instantiated transform(s)
        """
        def instantiate(obj, ds=None):
            if isinstance(obj, dict) and "class_path" in obj:
                cls_path = obj["class_path"]
                cls = pydoc.locate(cls_path)
                if cls is None:
                    raise ImportError(f"Could not locate class: {cls_path}")

                init_args = obj.get("init_args", {}).copy()
                
                # Check if the class accepts a 'dataset' argument
                sig = inspect.signature(cls.__init__)
                if "dataset" in sig.parameters and ds is not None:
                    init_args["dataset"] = ds
                    logging.debug(f"Passing dataset to {cls.__name__}")
                else:
                    if "dataset" not in sig.parameters:
                        logging.debug(f"{cls.__name__} does not have dataset parameter")
                    if ds is None:
                        logging.debug(f"Dataset is None for {cls.__name__}")

                # Recursively instantiate arguments
                resolved_args = {k: instantiate(v, ds=ds) for k, v in init_args.items()}
                result = cls(**resolved_args)
                
                # Debug: Check if bands were modified
                if ds is not None and hasattr(ds, "bands"):
                    logging.debug(f"After {cls.__name__}: dataset.bands = {ds.bands}")
                
                return result
            elif isinstance(obj, list):
                return [instantiate(item, ds=ds) for item in obj]
            elif isinstance(obj, dict):
                return {k: instantiate(v, ds=ds) for k, v in obj.items()}
            return obj
        
        return instantiate(transforms_config, ds=dataset)

    def _to_config(self, config: Union[DatasetConfig, Dict[str, Any]]) -> DatasetConfig:
        if isinstance(config, DatasetConfig):
            # Ensure transforms are preserved if they are dicts
            return config

        # Handle dict-based config (e.g. from YAML/CLI)
        # Store original transforms dict for prep script or other metadata needs
        original_transforms = config.get("transforms")

        cls = config.get("dataset_class")
        if isinstance(cls, str):
            resolved_cls = pydoc.locate(cls)
            if resolved_cls is None:
                raise ImportError(f"Could not locate dataset class: {cls}")
            config["dataset_class"] = resolved_cls

        # Note: transforms are NOT instantiated here - they're instantiated later with dataset ref
        # in _instantiate_combined_dataset via _instantiate_transforms
        if "transforms" in config:
            # Keep transforms as dict for now, will be instantiated later
            pass

        config_obj = DatasetConfig(**config)
        # Keep original transforms dict attached for inspection if needed
        config_obj._original_transforms = original_transforms
        return config_obj

    @staticmethod
    def _serialize_stats(stats_obj: Any) -> Any:
        """Recursively serialize stats, converting tensors to lists."""
        if isinstance(stats_obj, torch.Tensor):
            return stats_obj.tolist()
        elif isinstance(stats_obj, dict):
            return {
                k: BaseGeoDataModule._serialize_stats(v) for k, v in stats_obj.items()
            }
        elif isinstance(stats_obj, list):
            return [BaseGeoDataModule._serialize_stats(v) for v in stats_obj]
        return stats_obj

    @staticmethod
    def _deserialize_stats(stats_obj: Any) -> Any:
        """Recursively deserialize stats, converting lists of numbers to tensors."""
        if isinstance(stats_obj, list):
            # If it's a list of numbers, convert to tensor
            if len(stats_obj) > 0 and all(
                isinstance(x, (int, float)) for x in stats_obj
            ):
                return torch.tensor(stats_obj)
            # Otherwise, it's a list of items to deserialize
            return [BaseGeoDataModule._deserialize_stats(v) for v in stats_obj]
        elif isinstance(stats_obj, dict):
            return {
                k: BaseGeoDataModule._deserialize_stats(v) for k, v in stats_obj.items()
            }
        return stats_obj

    def _populate_normalize_stats(self, transforms, mean: Optional[List[float]], std: Optional[List[float]]):
        """Inject mean/std into Normalize transforms, accounting for SelectBands.
        Works on both instantiated objects and configuration dictionaries.
        
        Args:
            transforms: Transform object or configuration dictionary/list
            mean: Mean values to inject
            std: Std values to inject
            
        Returns:
            Transforms with populated stats
        """
        current_indices = None
        
        def find_and_populate(obj):
            nonlocal current_indices
            
            # Handle Dictionary Configs
            if isinstance(obj, dict):
                cls_path = obj.get("class_path", "")
                if "SelectBands" in cls_path:
                    current_indices = obj.get("init_args", {}).get("indices")
                    return obj
                
                if "Normalize" in cls_path:
                    init_args = obj.setdefault("init_args", {})
                    m, s = mean, std
                    if current_indices is not None:
                        if m is not None:
                            m = [m[i] for i in current_indices if i < len(m)]
                        if s is not None:
                            s = [s[i] for i in current_indices if i < len(s)]
                    
                    if init_args.get("mean") is None and m is not None:
                        init_args["mean"] = m
                    if init_args.get("std") is None and s is not None:
                        init_args["std"] = s
                    return obj
                
                # Recurse into dict values (e.g. Compose transforms list)
                for k, v in obj.items():
                    obj[k] = find_and_populate(v)
                return obj

            # Handle Lists
            if isinstance(obj, list):
                return [find_and_populate(item) for item in obj]

            # Handle Instantiated Objects
            from forestvision.transforms import Normalize, SelectBands
            if isinstance(obj, SelectBands):
                current_indices = obj.indices
                return obj

            if isinstance(obj, Normalize):
                m, s = mean, std
                if current_indices is not None:
                    if m is not None:
                        m = [m[i] for i in current_indices if i < len(m)]
                    if s is not None:
                        s = [s[i] for i in current_indices if i < len(s)]
                
                if obj.mean is None and m is not None:
                    obj.mean = m
                if obj.std is None and s is not None:
                    obj.std = s
                return obj
            
            if hasattr(obj, 'transforms'):
                for j, t in enumerate(obj.transforms):
                    obj.transforms[j] = find_and_populate(t)
            
            return obj
        
        return find_and_populate(transforms)

    def _instantiate_combined_dataset(
        self, configs: List[DatasetConfig], stage: str, roi: Optional[Any] = None, transforms: Optional[Any] = None
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
                    warnings.filterwarnings(
                        "ignore", category=UserWarning, message=".*relevant files.*"
                    )

                    # Create dataset instance first
                    ds = cfg.dataset_class(
                        bands=cfg.bands,
                        transforms=None,  # Transforms handled separately to pass dataset ref
                        **kwargs
                    )
                    # Explicitly synchronize dataset bands with configuration
                    ds.bands = cfg.bands.copy() if cfg.bands else []

                    # Instantiate transforms with dataset reference
                    if cfg.transforms is not None:
                        ds.transforms = self._instantiate_transforms(cfg.transforms, ds)

                        # Auto-populate Normalize transforms with stats from config
                        if cfg.mean is not None or cfg.std is not None:
                            # Re-verify alignment before populating
                            # If stats were filtered in prep script, they should match the final channel count
                            # BUT we must handle the case where stats are raw and transform appends bands
                            from forestvision.transforms import Normalize

                            def check_and_fix_stats(obj):
                                if isinstance(obj, Normalize):
                                    # Normalize expects mean/std to match the input it receives AT THAT POINT in the chain
                                    # BaseGeoDataModule handles per-dataset normalization.
                                    pass
                                if hasattr(obj, "transforms"):
                                    for t in obj.transforms:
                                        check_and_fix_stats(t)

                            ds.transforms = self._populate_normalize_stats(
                                ds.transforms, cfg.mean, cfg.std
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
            
        # Apply top-level transforms (e.g. SelectBands, Normalize for the whole stack)
        if transforms:
            combined.transforms = self._instantiate_transforms(transforms, combined)
            
        return combined

    def setup(self, stage: str, year: Optional[int] = None) -> None:
        if stage in ["fit", "validate", "prepare"]:
            if self.train_tiles_path:
                self.train_tiles = GPDFeatureCollection(os.path.join(self.root, self.train_tiles_path))
            if self.val_tiles_path:
                self.val_tiles = GPDFeatureCollection(os.path.join(self.root, self.val_tiles_path))
            
            train_roi = self.train_tiles.bounds if self.train_tiles else None
            val_roi = self.val_tiles.bounds if self.val_tiles else None

            input_ds = self._instantiate_combined_dataset(self.input_configs, "training", train_roi, self.input_transforms)
            target_ds = self._instantiate_combined_dataset(self.target_configs, "training", train_roi, self.target_transforms)
            if input_ds and target_ds:
                self.train_dataset = target_ds & input_ds

            val_input_ds = self._instantiate_combined_dataset(self.input_configs, "validation", val_roi, self.input_transforms)
            val_target_ds = self._instantiate_combined_dataset(self.target_configs, "validation", val_roi, self.target_transforms)
            if val_input_ds and val_target_ds:
                self.val_dataset = val_target_ds & val_input_ds

        elif stage == "test":
            if self.test_tiles_path:
                self.test_tiles = GPDFeatureCollection(os.path.join(self.root, self.test_tiles_path))
            test_roi = self.test_tiles.bounds if self.test_tiles else None
            
            input_ds = self._instantiate_combined_dataset(self.input_configs, "test", test_roi, self.input_transforms)
            target_ds = self._instantiate_combined_dataset(self.target_configs, "test", test_roi, self.target_transforms)
            if input_ds and target_ds:
                self.test_dataset = target_ds & input_ds

        elif stage == "predict":
            p_year = year or self.year
            if self.predict_tiles_path:
                self.predict_tiles = GPDFeatureCollection(os.path.join(self.root, self.predict_tiles_path))
            predict_roi = self.predict_tiles.bounds if self.predict_tiles else None
            
            # Predict might use different templates or year handling
            self.predict_dataset = self._instantiate_combined_dataset(self.input_configs, "predict", predict_roi, self.input_transforms)

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
        if not getattr(self, 'download', True):
            logging.info("Skipping data download (download=False)")
            return
        
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
            
            # Temporarily switch to all_bands if requested
            if download_all_bands and hasattr(dataset, 'all_bands'):
                dataset.bands = dataset.all_bands
                logging.info(
                    f"Downloading ALL bands for {dataset.__class__.__name__}: "
                    f"{dataset.all_bands}"
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
                # Restore original transforms (bands are left as-is since transforms may have added derived bands)
                dataset.transforms = original_transforms

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
            elif self.train_dataset and self.val_tiles:
                # When using shared paths (no {stage} in path_template), 
                # validation datasets may not be created separately.
                # Use training datasets but download for validation tiles.
                tasks.append((self.train_dataset, self.val_tiles))
            if self.test_dataset and self.test_tiles:
                tasks.append((self.test_dataset, self.test_tiles))

            # Track unique paths and class to avoid redundant downloads
            seen_keys = set()

            for top_ds, tiles in tasks:
                for ds in get_all_datasets(top_ds):
                    # We use path + class + tiles count to distinguish datasets
                    # This ensures we download for different tile sets (train vs val)
                    path = getattr(ds, 'path', getattr(ds, 'paths', None))
                    ds_name = ds.__class__.__name__
                    tiles_count = len(tiles.data) if hasattr(tiles, 'data') else 0
                    key = (ds_name, str(path), tiles_count)
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
