import os
import logging
from typing import Any, Dict, Optional

import ee
from dotenv import load_dotenv

import torch
from kornia.enhance import Denormalize
from torch.utils.data import DataLoader
from torchgeo.datasets import stack_samples, IntersectionDataset
from torchvision.transforms import v2

from forestvision.datasets import (
    GNNForestAttr,
    GEESentinel2,
    GEE3Dep,
    GPDFeatureCollection,
)
from forestvision.datamodules import CloudDataModule
from forestvision.transforms import (
    Normalize,
    RemapFortypba,
    MinMaxScaler,
    InverseMinMaxScaler,
)
from forestvision.samplers import TileGeoSampler
from forestvision.deploy import AnyRasterDataset

torch.set_float32_matmul_precision("medium")


# Load from .env file
load_dotenv()
GEE_PROJECT_NAME = os.getenv("GEE_PROJECT_NAME")
TARGET_PATH = os.getenv("TARGET_PATH")


class ClimateNA(AnyRasterDataset):
    all_bands = ["AHM", "MAP", "TD"]
    _cmap = None
    rgb_bands = ["AHM", "MAP", "TD"]
    instrument = "ClimateNA"
    nodata = -9999


class ForTypesDataModule(CloudDataModule):
    """LightningDataModule implementation for forest type classification.

    This data module supports integrated serialization with auto-loading of statistics
    from hyperparameters, enabling efficient resume training and inference without
    recomputing dataset statistics.

    Key Features:
    - Computes input and target dataset statistics (mean, std) for normalization
    - Serializes statistics to YAML-compatible format for logging
    - Auto-loads statistics from hparams during resume/inference
    - Maintains backward compatibility with existing workflows

    Usage Examples:

    First Training Run (computes and logs stats):
        >>> datamodule = ForTypesDataModule(
        ...     root="data",
        ...     year=2020,
        ...     train_tiles_path="tiles/train.geojson",
        ...     val_tiles_path="tiles/val.geojson"
        ... )
        >>> datamodule.prepare_data()  # Computes and logs stats to trainer
        >>> trainer.fit(model, datamodule)

    Resume Training or Inference (loads from hparams):
        >>> # Load hparams from checkpoint or previous run
        >>> hparams = {"datamodule": {"input_stats": {...}, "target_stats": {...}}}
        >>> datamodule = ForTypesDataModule(
        ...     root="data",
        ...     year=2020,
        ...     train_tiles_path="tiles/train.geojson",
        ...     val_tiles_path="tiles/val.geojson",
        ...     hparams=hparams  # Stats loaded from hparams, no recomputation
        ... )
        >>> datamodule.prepare_data()  # Skips computation, uses hparams stats
        >>> trainer.fit(model, datamodule)

    Manual Stat Loading:
        >>> # Load stats from file and pass as hparams
        >>> import torch
        >>> input_stats = torch.load("input_stats.pt")
        >>> target_stats = torch.load("target_stats.pt")
        >>> hparams = {"datamodule": {"input_stats": input_stats, "target_stats": target_stats}}
        >>> datamodule = ForTypesDataModule(..., hparams=hparams)
    """

    input_stats = None
    target_stats = None

    def __init__(
        self,
        root: str,
        year: int,
        stats_path: str,
        batch_size: int | None = None,
        patch_size: int | tuple[int, int] = None,
        epoch_length: int | None = None,
        num_workers: int | None = None,
        target_path: str | None = None,
        test_tiles_path: str | None = None,
        val_tiles_path: str | None = None,
        predict_tiles_path: str | None = None,
        train_tiles_path: str | None = None,
        ee_project: str | None = None,
        hparams: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a new ForTypesDataModule instance.

        Args:
            root (str): Root directory for data storage.
            year (int): Year of training, validation, and test data.
            batch_size (int): Size of each mini-batch.
            patch_size (int | tuple[int, int]): Size of each patch, either ``size`` or ``(height, width)``.
            epoch_length (int | None): Length of each training epoch.
            num_workers (int): Number of workers for parallel data loading.
            target_path (str | None): Path to target dataset. If None, uses default GNNForestAttr path.
            test_tiles_path (str | None): Path to test tiles GeoJSON file.
            val_tiles_path (str | None): Path to validation tiles GeoJSON file.
            predict_tiles_path (str | None): Path to prediction tiles GeoJSON file.
            train_tiles_path (str | None): Path to training tiles GeoJSON file.
            ee_project (str | None): Google Earth Engine project name for initialization.
            hparams (Optional[Dict[str, Any]]): Optional hyperparameters dictionary containing
                precomputed statistics. If provided, stats will be loaded from hparams instead
                of being recomputed. Expected format:
                hparams["datamodule"]["input_stats"] and hparams["datamodule"]["target_stats"]
            **kwargs: Additional arguments passed to parent class.

        Example:
            >>> # First run - computes and logs stats
            >>> datamodule = ForTypesDataModule(root="data", year=2020)
            >>>
            >>> # Resume/inference run - loads from hparams
            >>> datamodule = ForTypesDataModule(
            ...     root="data",
            ...     year=2020,
            ...     hparams={"datamodule": {"input_stats": {...}, "target_stats": {...}}}
            ... )
        """
        # Initialize Earth Engine if project name is provided
        ee_project = ee_project or GEE_PROJECT_NAME

        try:
            ee.Initialize(project=ee_project)
            logging.info(f"Earth Engine initialized with project: {ee_project}")
        except Exception as e:
            logging.warning(f"Failed to initialize Earth Engine: {e}")
            raise

        self.year = year
        self.root = root
        self.target_path = target_path or TARGET_PATH
        self.hparams_dict = hparams or {}
        self.stats_path = stats_path
        self.train_tiles_path = train_tiles_path
        self.val_tiles_path = val_tiles_path
        self.test_tiles_path = test_tiles_path
        self.predict_tiles_path = predict_tiles_path

        # Initialize datasets to None - will be created in setup()
        self.input_dataset = None
        self.val_input_dataset = None
        self.test_input_dataset = None
        self.predict_inputs_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.predict_dataset = None

        # Initialize samplers to None - will be created in *_dataloader methods
        self.train_sampler = None
        self.val_sampler = None
        self.predict_sampler = None

        # Initialize tiles to None - will be loaded in setup()
        self.train_tiles = None
        self.val_tiles = None
        self.test_tiles = None
        self.predict_tiles = None

        # Track if stats were loaded from hparams
        self.stats_from_hparams = False

        # Track if transforms have been applied (for lazy application)
        self._transforms_applied = False

        self.satimagery_class = GEESentinel2
        self.dem_class = GEE3Dep
        self.climate_class = ClimateNA
        self.init_transforms = None  # ReplaceNodataVal(-32768, 0, on_key="image")

        super().__init__(
            dataset_class=self.satimagery_class,
            batch_size=batch_size,
            patch_size=patch_size,
            length=epoch_length,
            num_workers=num_workers,
            **kwargs,
        )

    @staticmethod
    def _serialize_stats(stats_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively serialize torch tensors in statistics dictionary to YAML-compatible types.

        Converts torch tensors to lists or primitive values for safe YAML serialization.

        Args:
            stats_dict: Dictionary containing statistics with torch tensors

        Returns:
            Dictionary with torch tensors converted to lists/primitives

        Example:
            >>> stats = {"mean": torch.tensor([1.0, 2.0]), "std": torch.tensor([0.1, 0.2])}
            >>> serialized = ForTypesDataModule._serialize_stats(stats)
            >>> print(serialized)
            {'mean': [1.0, 2.0], 'std': [0.1, 0.2]}
        """
        serialized = {}
        for key, value in stats_dict.items():
            if isinstance(value, torch.Tensor):
                serialized[key] = value.tolist()
            elif isinstance(value, dict):
                serialized[key] = ForTypesDataModule._serialize_stats(value)
            else:
                serialized[key] = value
        return serialized

    @staticmethod
    def _deserialize_stats(stats_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively deserialize statistics dictionary from YAML-compatible types to torch tensors.

        Converts lists and primitive values back to torch tensors for use in normalization.

        Args:
            stats_dict: Dictionary containing serialized statistics

        Returns:
            Dictionary with lists/primitives converted to torch tensors

        Example:
            >>> serialized = {'mean': [1.0, 2.0], 'std': [0.1, 0.2]}
            >>> deserialized = ForTypesDataModule._deserialize_stats(serialized)
            >>> print(deserialized)
            {'mean': tensor([1., 2.]), 'std': tensor([0.1000, 0.2000])}
        """
        deserialized = {}
        for key, value in stats_dict.items():
            if isinstance(value, list):
                deserialized[key] = torch.tensor(value)
            elif isinstance(value, dict):
                deserialized[key] = ForTypesDataModule._deserialize_stats(value)
            else:
                deserialized[key] = value

        return deserialized

    def _load_stats_from_file(self) -> bool:
        """Load statistics from the specified stats_path file.

        Returns:
            bool: True if stats were successfully loaded, False otherwise
        """
        if not self.stats_path:
            logging.info("No stats_path provided, skipping file loading")
            return False

        if not os.path.exists(self.stats_path):
            logging.warning(f"Stats file not found at {self.stats_path}")
            return False

        def extract_datasets(dataset):
            if isinstance(dataset, IntersectionDataset):
                result = []
                for ds in dataset.datasets:
                    result.extend(extract_datasets(ds))
                return result
            else:
                return [dataset]

        try:
            logging.info(f"Loading statistics from {self.stats_path}")
            stats_dict = torch.load(self.stats_path)

            # Check if stats_dict contains both input and target stats
            if "input_stats" in stats_dict and "target_stats" in stats_dict:
                # If input_datasets is an IntersectionDataset, merge stats from all datasets
                if isinstance(self.input_dataset, IntersectionDataset):
                    all_input_stats = {"mean": [], "std": [], "min": [], "max": []}
                    input_stats = stats_dict["input_stats"]
                    for key in all_input_stats.keys():
                        all_input_stats[key] = [
                            i
                            for sbl in [
                                input_stats.get(ds.__class__.__name__).get(key)
                                for ds in extract_datasets(self.input_dataset)
                            ]
                            for i in sbl
                        ]
                    stats_dict["input_stats"] = all_input_stats
                self.input_stats = self._deserialize_stats(stats_dict["input_stats"])
                self.target_stats = self._deserialize_stats(stats_dict["target_stats"])
                logging.info(
                    "Successfully loaded input and target statistics from file"
                )
                return True
            else:
                logging.warning(
                    f"Stats file missing required keys: {stats_dict.keys()}"
                )
                return False

        except Exception as e:
            logging.error(f"Failed to load statistics from {self.stats_path}: {e}")
            return False

    def _collate_fn(self, batch):
        """Custom collate function that removes frozen dataclasses before GPU transfer.

        This function removes BoundingBox and CRS objects from the batch to prevent
        PyTorch Lightning from trying to move them to the GPU, which would cause
        errors with frozen dataclasses.

        Args:
            batch: List of samples from the dataset

        Returns:
            dict: Collated batch with only tensor data
        """
        # Use the standard stack_samples to collate tensors
        collated = stack_samples(batch)

        # Debug: Check mask values to verify transforms are applied
        mask_unique = torch.unique(collated["mask"])
        logging.info(f"Mask unique values in collate: {mask_unique}")

        # Verify that mask values are in the expected range [0-13, -1]
        # expected_values = set(range(0, 14)) | {-1}
        # actual_values = set(mask_unique.tolist())

        # unexpected_values = actual_values - expected_values
        # if unexpected_values:
        #     logging.warning(f"Unexpected mask values detected: {unexpected_values}")
        #     logging.warning(f"Expected range: {expected_values}, Got: {actual_values}")
        # else:
        #     logging.info("Mask values are properly remapped to [0-13, -1]")

        # prep mask
        return {
            "mask": collated["mask"],
            "image": collated["image"],
            "crs": collated["crs"],
            "bounds": collated["bbox"],
        }

    def setup_transforms(self) -> None:
        """Setup transforms for datasets.

        Configures normalization transforms for both input and target datasets
        based on precomputed statistics. First checks if stats are available in hparams,
        otherwise falls back to computed stats.

        Transform Strategy:
        - Target transforms (RemapFortypba) are applied to COMBINED datasets to ensure
          mask remapping happens regardless of dataset composition
        - Input transforms (MinMaxScaler) are applied to INDIVIDUAL input datasets
          since they need to be normalized before being combined

        Workflow:
        1. Check if hparams contains precomputed stats
        2. If yes, deserialize and use them (skip computation)
        3. If no, use existing computed stats or compute new ones
        4. Apply target transforms to combined datasets
        5. Apply input transforms to individual input datasets
        """
        # Check if stats are available in hparams (highest priority)
        if (
            self.hparams_dict
            and "datamodule" in self.hparams_dict
            and "input_stats" in self.hparams_dict["datamodule"]
            and "target_stats" in self.hparams_dict["datamodule"]
        ):

            logging.info("Loading statistics from hparams...")
            try:
                # Deserialize stats from hparams
                self.input_stats = self._deserialize_stats(
                    self.hparams_dict["datamodule"]["input_stats"]
                )
                self.target_stats = self._deserialize_stats(
                    self.hparams_dict["datamodule"]["target_stats"]
                )
                self.stats_from_hparams = True
                logging.info("Statistics successfully loaded from hparams")
            except Exception as e:
                logging.warning(f"Failed to deserialize stats from hparams: {e}")
                self.stats_from_hparams = False
                # Fall back to existing stats or computation
                if self.input_stats is None or self.target_stats is None:
                    logging.info("Falling back to stat computation")
                    return  # Let prepare_data() handle computation

        # Apply target transforms to combined datasets to ensure remapping happens
        # regardless of dataset composition
        target_transforms = v2.Compose(
            [
                RemapFortypba(remap_dict=GNNForestAttr.remap_dict, on_key="mask"),
            ]
        )

        # Apply target transforms to all combined datasets that exist
        combined_datasets = [
            ("train_dataset", self.train_dataset),
            ("val_dataset", self.val_dataset),
            ("test_dataset", self.test_dataset),
            ("predict_dataset", self.predict_dataset),
        ]

        for name, dataset in combined_datasets:
            if dataset is not None:
                dataset.transforms = target_transforms
                logging.info(f"Applied target transforms to {name}")

        if self.input_stats is not None:
            input_transforms = v2.Compose(
                [
                    Normalize(
                        mean=self.input_stats["mean"],
                        std=self.input_stats["std"],
                        on_key="image",
                    ),
                ]
            )

            # Apply input transforms to individual input datasets that exist
            input_datasets = [
                ("input_dataset", self.input_dataset),
                ("val_input_dataset", self.val_input_dataset),
                ("test_input_dataset", self.test_input_dataset),
                ("predict_inputs_dataset", self.predict_inputs_dataset),
            ]

            for name, dataset in input_datasets:
                if dataset is not None:
                    dataset.transforms = input_transforms
                    logging.info(f"Applied input transforms to {name}")

            self.revert_inputs = Denormalize(
                mean=self.input_stats["mean"], std=self.input_stats["std"]
            )
            self.revert_target = Denormalize(
                mean=self.target_stats["mean"], std=self.target_stats["std"]
            )

    def prepare_data(self, overwrite: bool = False) -> None:
        """Prepare data by downloading datasets.

        Note: Statistics computation has been moved to a separate script.
        Use scripts/compute_stats.py to compute statistics and then pass
        the stats file path via the stats_path parameter.

        Args:
            overwrite: Whether to overwrite existing data
        """
        self.setup("fit")
        logging.info("Preparing training dataset...")

        # Statistics are now loaded from file in setup() method
        # or computed separately using scripts/compute_stats.py
        logging.info("Statistics should be precomputed using scripts/compute_stats.py")
        logging.info("and loaded via stats_path parameter")

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader using TileGeoSampler.

        Returns:
            DataLoader: Training data loader
        """
        # Lazy transform application: apply transforms if stats are available but not applied yet
        if self.input_stats is not None and not self._transforms_applied:
            logging.info("Lazy applying transforms in train_dataloader...")
            self.setup_transforms()
            self._transforms_applied = True

        if self.train_sampler is None:
            self.train_sampler = TileGeoSampler(
                self.train_dataset, self.train_tiles.data, shuffle=True
            )

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader using TileGeoSampler.

        Returns:
            DataLoader: Validation data loader
        """
        # Lazy transform application: apply transforms if stats are available but not applied yet
        if self.input_stats is not None and not self._transforms_applied:
            logging.info("Lazy applying transforms in val_dataloader...")
            self.setup_transforms()
            self._transforms_applied = True

        if self.val_sampler is None:
            self.val_sampler = TileGeoSampler(
                self.val_dataset, self.val_tiles.data, shuffle=True
            )

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            sampler=self.val_sampler,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader using TileGeoSampler.

        Returns:
            DataLoader: Test data loader
        """
        # Lazy transform application: apply transforms if stats are available but not applied yet
        if self.input_stats is not None and not self._transforms_applied:
            logging.info("Lazy applying transforms in test_dataloader...")
            self.setup_transforms()
            self._transforms_applied = True

        if not hasattr(self, "test_dataset") or self.test_dataset is None:
            self.setup("test")

        if self.test_sampler is None:
            self.test_sampler = TileGeoSampler(
                self.test_dataset, self.test_tiles.data, shuffle=False
            )

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            sampler=self.test_sampler,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )

    def predict_dataloader(self) -> DataLoader:
        """Return the prediction dataloader using TileGeoSampler.

        Returns:
            DataLoader: Prediction data loader
        """
        if not hasattr(self, "predict_dataset") or self.predict_dataset is None:
            raise RuntimeError(
                "Predict dataset not set up. Call setup('predict', year=...) first."
            )

        if self.predict_sampler is None:
            self.predict_sampler = TileGeoSampler(
                self.predict_dataset, self.predict_tiles.data, shuffle=False
            )

        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            sampler=self.predict_sampler,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )

    def setup(self, stage: str, year: int | None = None) -> None:
        """Setup datasets for the specified stage.

        Args:
            stage: The stage to setup ('fit', 'validate', 'test', or 'predict')
            year: Required for 'predict' stage, specifies the prediction year

        Raises:
            ValueError: If year is not provided for predict stage
        """
        if self.target_path:
            self.target_dataset = GNNForestAttr(paths=self.target_path, res=10)
        else:
            self.target_dataset = GNNForestAttr(res=10)

        if stage in ["fit", "validate"]:
            self._setup_fit_stage()

        elif stage == "test":
            self._setup_test_stage()

        elif stage == "predict":
            if year is None:
                raise ValueError("year parameter required for predict stage")
            self._setup_predict_stage(year)

        # Load statistics from file if available and not already loaded
        if self.input_stats is None and self.stats_path:
            self._load_stats_from_file()

        self.setup_transforms()

    def _setup_fit_stage(self) -> None:
        """Setup datasets for training and validation stages."""
        self.train_tiles = GPDFeatureCollection(
            os.path.join(self.root, self.train_tiles_path)
        )
        self.val_tiles = GPDFeatureCollection(
            os.path.join(self.root, self.val_tiles_path)
        )

        self.training_satimagery_path = os.path.join(
            self.root,
            "training",
            self.satimagery_class.__name__.lower(),
            str(self.year),
        )
        self.training_dem_path = os.path.join(
            self.root, "training", self.dem_class.__name__.lower(), str(self.year)
        )
        self.training_cimate_path = os.path.join(self.root, "training", "climatena")

        validation_satimagery_path = self.training_satimagery_path.replace(
            "training", "validation"
        )
        validation_dem_path = self.training_dem_path.replace("training", "validation")
        # We will use the same path for training and validation climate data
        validation_climate_path = self.training_cimate_path

        training_satimagery_dataset = self.satimagery_class(
            year=self.year,
            roi=self.train_tiles.bounds,
            path=self.training_satimagery_path,
            transforms=self.init_transforms,
            download=True,
        )
        training_dem_dataset = self.dem_class(
            roi=self.train_tiles.bounds,
            res=10,
            path=self.training_dem_path,
            transforms=self.init_transforms,
            download=True,
        )
        training_climate_dataset = self.climate_class(
            paths=self.training_cimate_path,
            glob="*.tif",
            crs=self.target_dataset.crs,
            res=10,
            is_image=True,
            nodata=-9999,
        )

        val_satimagery_dataset = self.satimagery_class(
            year=self.year,
            roi=self.val_tiles.bounds,
            path=validation_satimagery_path,
            transforms=self.init_transforms,
            download=True,
        )
        val_dem_dataset = self.dem_class(
            roi=self.val_tiles.bounds,
            res=10,
            path=validation_dem_path,
            transforms=self.init_transforms,
            download=True,
        )
        # val_climate_dataset = self.climate_class(
        #     paths=validation_climate_path,
        #     glob="*.vrt",
        #     crs=self.target_dataset.crs,
        #     res=10,
        #     is_image=True,
        # )

        # Combine datasets
        self.input_dataset = (
            training_satimagery_dataset
            & training_dem_dataset
            & training_climate_dataset
        )
        self.train_dataset = self.target_dataset & self.input_dataset

        self.val_input_dataset = (
            val_satimagery_dataset & val_dem_dataset & training_climate_dataset
        )
        self.val_dataset = self.target_dataset & self.val_input_dataset

    def _setup_test_stage(self) -> None:
        """Setup datasets for testing stage."""
        self.test_tiles = GPDFeatureCollection(
            os.path.join(self.root, self.test_tiles_path)
        )

        dataset_class_name = self.satimagery_class.__name__.lower()
        test_inputs_path = os.path.join(
            self.root, "test", dataset_class_name, str(self.year)
        )

        self.test_input_dataset = self.satimagery_class(
            year=self.year,
            roi=self.test_tiles.bounds,
            path=test_inputs_path,
            download=True,
        )

        # Combine mask & inputs
        self.test_dataset = self.target_dataset & self.test_input_dataset

    def _setup_predict_stage(self, year: int) -> None:
        """Setup datasets for prediction stage.

        Args:
            year (int): The year to predict for
        """
        self.predict_tiles = GPDFeatureCollection(
            os.path.join(self.root, self.predict_tiles_path)
        )

        dataset_class_name = self.satimagery_class.__name__.lower()
        predict_inputs_path = os.path.join(self.root, f"predict/{dataset_class_name}")

        self.predict_inputs_dataset = AnyRasterDataset(
            glob="*.tif",
            crs=self.target_dataset.crs,
            paths=predict_inputs_path,
            res=10,
            is_image=True,
        )

        self.predict_dataset = self.target_dataset & self.predict_inputs_dataset

    def cleanup(self) -> None:
        """Clean up loaded resources and statistics to free memory.

        Releases memory by clearing datasets, samplers, tiles, and statistics
        that were loaded during setup.
        """
        logging.info("Cleaning up data module resources")
        self.input_stats = None
        self.target_stats = None

        # Clear datasets if exist
        for attr in [
            "input_dataset",
            "val_input_dataset",
            "test_input_dataset",
            "predict_inputs_dataset",
            "train_dataset",
            "val_dataset",
            "test_dataset",
            "predict_dataset",
        ]:
            if hasattr(self, attr):
                setattr(self, attr, None)

        # Clear samplers if exist
        for attr in ["train_sampler", "val_sampler", "test_sampler", "predict_sampler"]:
            if hasattr(self, attr):
                setattr(self, attr, None)

        # Clear tiles if exist
        for attr in ["train_tiles", "val_tiles", "test_tiles", "predict_tiles"]:
            if hasattr(self, attr):
                setattr(self, attr, None)
