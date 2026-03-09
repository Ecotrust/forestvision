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
        train_transforms: Optional[Any] = None,
        post_aug_input_transforms: Optional[Any] = None,
        post_aug_target_transforms: Optional[Any] = None,
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
        self.train_transforms = train_transforms
        self.post_aug_input_transforms = post_aug_input_transforms
        self.post_aug_target_transforms = post_aug_target_transforms

        self.input_configs = [self._to_config(c) for c in input_configs]
        self.target_configs = [self._to_config(c) for c in target_configs]

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.train_tiles = None
        self.val_tiles = None
        self.test_tiles = None
        self.predict_tiles = None

    @property
    def _shape_str(self) -> str:
        """Convert patch_size to shape string for path template.
        
        Returns:
            Shape string like "128x128" or "256x256"
        """
        if isinstance(self.patch_size, int):
            return f"{self.patch_size}x{self.patch_size}"
        elif isinstance(self.patch_size, (tuple, list)) and len(self.patch_size) >= 2:
            return f"{self.patch_size[0]}x{self.patch_size[1]}"
        return "128x128"

    def _instantiate_transforms(self, transforms_config: Any, dataset: Any) -> Any:
        def instantiate(obj, ds=None):
            if isinstance(obj, dict) and "class_path" in obj:
                cls_path = obj["class_path"]
                cls = pydoc.locate(cls_path)
                if cls is None:
                    raise ImportError(f"Could not locate class: {cls_path}")

                init_args = obj.get("init_args", {}).copy()
                sig = inspect.signature(cls.__init__)
                if "dataset" in sig.parameters and ds is not None:
                    init_args["dataset"] = ds
                    logging.debug(f"Passing dataset to {cls.__name__}")

                resolved_args = {k: instantiate(v, ds=ds) for k, v in init_args.items()}
                result = cls(**resolved_args)

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
            return config

        original_transforms = config.get("transforms")

        cls = config.get("dataset_class")
        if isinstance(cls, str):
            resolved_cls = pydoc.locate(cls)
            if resolved_cls is None:
                raise ImportError(f"Could not locate dataset class: {cls}")
            config["dataset_class"] = resolved_cls

        config_obj = DatasetConfig(**config)
        config_obj._original_transforms = original_transforms
        return config_obj

    @staticmethod
    def _serialize_stats(stats_obj: Any) -> Any:
        if isinstance(stats_obj, torch.Tensor):
            return stats_obj.tolist()
        elif isinstance(stats_obj, dict):
            return {k: BaseGeoDataModule._serialize_stats(v) for k, v in stats_obj.items()}
        elif isinstance(stats_obj, list):
            return [BaseGeoDataModule._serialize_stats(v) for v in stats_obj]
        return stats_obj

    @staticmethod
    def _deserialize_stats(stats_obj: Any) -> Any:
        if isinstance(stats_obj, list):
            if len(stats_obj) > 0 and all(isinstance(x, (int, float)) for x in stats_obj):
                return torch.tensor(stats_obj)
            return [BaseGeoDataModule._deserialize_stats(v) for v in stats_obj]
        elif isinstance(stats_obj, dict):
            return {k: BaseGeoDataModule._deserialize_stats(v) for k, v in stats_obj.items()}
        return stats_obj

    def _populate_normalize_stats(
        self,
        transforms,
        mean: Optional[List[float]],
        std: Optional[List[float]],
        nodata: Optional[int] = None,
    ):
        # print(f"\n[DEBUG _populate_normalize_stats] Called with mean={mean}, std={std}, nodata={nodata}")
        current_indices = None

        def find_and_populate(obj):
            nonlocal current_indices

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
                    if init_args.get("nodata") is None and nodata is not None:
                        init_args["nodata"] = nodata
                    return obj

                for k, v in obj.items():
                    obj[k] = find_and_populate(v)
                return obj

            if isinstance(obj, list):
                return [find_and_populate(item) for item in obj]

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

                # print(f"[DEBUG _populate_normalize_stats] Found Normalize transform")
                # print(f"[DEBUG _populate_normalize_stats] Before: mean={obj.mean}, std={obj.std}, nodata={obj.nodata}")

                if obj.mean is None and m is not None:
                    obj.mean = m
                if obj.std is None and s is not None:
                    obj.std = s
                if obj.nodata is None and nodata is not None:
                    obj.nodata = nodata

                # print(f"[DEBUG _populate_normalize_stats] After: mean={obj.mean}, std={obj.std}, nodata={obj.nodata}")
                return obj

            if hasattr(obj, "transforms"):
                for j, t in enumerate(obj.transforms):
                    obj.transforms[j] = find_and_populate(t)

            return obj

        return find_and_populate(transforms)

    def _instantiate_combined_dataset(
        self,
        configs: List[DatasetConfig],
        stage: str,
        roi: Optional[Any] = None,
        transforms: Optional[Any] = None,
    ) -> IntersectionDataset:
        datasets = []
        for cfg in configs:
            path = cfg.path_template.format(root=self.root, year=self.year, stage=stage, shape=self._shape_str)

            if not os.path.isabs(path):
                path = os.path.join(self.root, path)

            kwargs = cfg.kwargs.copy()
            sig = inspect.signature(cfg.dataset_class.__init__)

            if "paths" in sig.parameters:
                kwargs["paths"] = path
            elif "path" in sig.parameters:
                kwargs["path"] = path

            if "year" in sig.parameters and "year" not in kwargs:
                kwargs["year"] = self.year

            if "roi" in sig.parameters:
                kwargs["roi"] = roi

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning, message=".*relevant files.*")

                    ds = cfg.dataset_class(
                        bands=cfg.bands,
                        transforms=None,
                        **kwargs,
                    )
                    ds.bands = cfg.bands.copy() if cfg.bands else []

                    if cfg.transforms is not None:
                        ds.transforms = self._instantiate_transforms(cfg.transforms, ds)

                        if cfg.mean is not None or cfg.std is not None:
                            from forestvision.transforms import Normalize

                            ignore_index = self.hparams_dict.get("ignore_index")

                            ds.transforms = self._populate_normalize_stats(
                                ds.transforms, cfg.mean, cfg.std, nodata=ignore_index
                            )

                datasets.append(ds)
            except (DatasetNotFoundError, UserWarning, Exception) as e:
                is_path_error = "Dataset not found" in str(e) or "relevant files" in str(e)

                if is_path_error:
                    fallback_kwargs = kwargs.copy()
                    fallback_kwargs.pop("path", None)
                    fallback_kwargs.pop("paths", None)

                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=UserWarning, message=".*relevant files.*")
                            ds = cfg.dataset_class(
                                bands=cfg.bands,
                                transforms=cfg.transforms,
                                **fallback_kwargs,
                            )
                        datasets.append(ds)
                        logging.info(f"Fell back to default directory for {cfg.dataset_class.__name__}")
                        continue
                    except Exception as fallback_e:
                        logging.warning(f"Could not instantiate {cfg.dataset_class.__name__} at {path} or default.")

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

            # print(f"\n[DEBUG setup] Creating datasets with stage={stage}")
            # print(f"[DEBUG setup] post_aug_input_transforms={self.post_aug_input_transforms}")
            # print(f"[DEBUG setup] post_aug_target_transforms={self.post_aug_target_transforms}")

            train_input_ds = self._instantiate_combined_dataset(
                self.input_configs, "training", train_roi, transforms=None
            )
            train_target_ds = self._instantiate_combined_dataset(
                self.target_configs, "training", train_roi, transforms=None
            )

            if train_input_ds and train_target_ds:
                self.train_dataset = train_target_ds & train_input_ds

                transform_chain = []

                if self.input_transforms:
                    instantiated = self._instantiate_transforms(self.input_transforms, self.train_dataset)
                    if not isinstance(instantiated, list):
                        instantiated = [instantiated]
                    transform_chain.extend(instantiated)

                if self.target_transforms:
                    instantiated = self._instantiate_transforms(self.target_transforms, self.train_dataset)
                    if not isinstance(instantiated, list):
                        instantiated = [instantiated]
                    transform_chain.extend(instantiated)

                if self.train_transforms:
                    instantiated_aug = self._instantiate_transforms(self.train_transforms, self.train_dataset)
                    if not isinstance(instantiated_aug, list):
                        instantiated_aug = [instantiated_aug]
                    transform_chain.extend(instantiated_aug)

                if self.post_aug_input_transforms:
                    instantiated = self._instantiate_transforms(self.post_aug_input_transforms, self.train_dataset)
                    if not isinstance(instantiated, list):
                        instantiated = [instantiated]
                    transform_chain.extend(instantiated)

                if self.post_aug_target_transforms:
                    instantiated = self._instantiate_transforms(self.post_aug_target_transforms, self.train_dataset)
                    if not isinstance(instantiated, list):
                        instantiated = [instantiated]
                    transform_chain.extend(instantiated)

                # print(f"[DEBUG setup] Final transform_chain length: {len(transform_chain)}")
                # for i, t in enumerate(transform_chain):
                #     print(f"[DEBUG setup] Transform {i}: {type(t).__name__} - {t}")

                if transform_chain:
                    from forestvision.transforms.augmentations import ComposeAugmentations
                    self.train_dataset.transforms = ComposeAugmentations(transform_chain)

            val_input_ds = self._instantiate_combined_dataset(
                self.input_configs, "validation", val_roi, transforms=None
            )
            val_target_ds = self._instantiate_combined_dataset(
                self.target_configs, "validation", val_roi, transforms=None
            )
            if val_input_ds and val_target_ds:
                self.val_dataset = val_target_ds & val_input_ds

                val_transform_chain = []

                if self.input_transforms:
                    instantiated = self._instantiate_transforms(self.input_transforms, self.val_dataset)
                    if not isinstance(instantiated, list):
                        instantiated = [instantiated]
                    val_transform_chain.extend(instantiated)

                if self.target_transforms:
                    instantiated = self._instantiate_transforms(self.target_transforms, self.val_dataset)
                    if not isinstance(instantiated, list):
                        instantiated = [instantiated]
                    val_transform_chain.extend(instantiated)

                if self.post_aug_input_transforms:
                    instantiated = self._instantiate_transforms(self.post_aug_input_transforms, self.val_dataset)
                    if not isinstance(instantiated, list):
                        instantiated = [instantiated]
                    val_transform_chain.extend(instantiated)

                if self.post_aug_target_transforms:
                    instantiated = self._instantiate_transforms(self.post_aug_target_transforms, self.val_dataset)
                    if not isinstance(instantiated, list):
                        instantiated = [instantiated]
                    val_transform_chain.extend(instantiated)

                if val_transform_chain:
                    from forestvision.transforms.augmentations import ComposeAugmentations
                    self.val_dataset.transforms = ComposeAugmentations(val_transform_chain)

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

            self.predict_dataset = self._instantiate_combined_dataset(self.input_configs, "predict", predict_roi, self.input_transforms)

    def _collate_fn(self, batch):
        collated = stack_samples(batch)
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

    def predict_dataloader(self) -> DataLoader:
        """DataLoader for prediction/inference.

        Returns:
            DataLoader configured for the prediction dataset.

        Raises:
            RuntimeError: If predict_dataset is not set up or predict_tiles is not provided.
        """
        if self.predict_dataset is None:
            raise RuntimeError(
                "predict_dataset is not initialized. "
                "Ensure setup('predict') is called before predict_dataloader()."
            )
        if self.predict_tiles is None:
            raise RuntimeError(
                "predict_tiles is not set. "
                "Ensure predict_tiles_path is provided in constructor."
            )

        sampler = TileGeoSampler(self.predict_dataset, self.predict_tiles.data, shuffle=False)
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )

    def prepare_data(self) -> None:
        """Prepare data - called only on the main process in distributed training.

        This method is called before setup() and should be used for:
        - Downloading data
        - Creating caches
        - Any one-time preparation that shouldn't be done by all processes
        """
        # Data preparation is handled in setup() for this datamodule
        pass

    def cleanup(self) -> None:
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

