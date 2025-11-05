import os
import logging
from typing import Any, Dict, Optional, Union

import ee
from tqdm import tqdm
import torch
from kornia.enhance import Denormalize
from torch.utils.data import DataLoader
from torchgeo.datasets import stack_samples, GeoDataset
from torchvision.transforms import v2
from geopandas import GeoDataFrame
from dotenv import load_dotenv

from forestvision.datasets import (
    eMapRAGB,
    GEELandsatFTV,
    DatasetStats,
    GPDFeatureCollection,
)
from forestvision.datamodules.clouddatamodule import CloudDataModule
from forestvision.transforms import Normalize, ReplaceNodataVal
from forestvision.samplers import TileGeoSampler
from forestvision.deploy import AnyRasterDataset

torch.set_float32_matmul_precision("medium")


# Load from .env file
load_dotenv()
GEE_PROJECT_NAME = os.getenv("GEE_PROJECT_NAME")
TARGET_PATH = os.getenv("TARGET_PATH")


def download_dataset(
    dataset: GeoDataset,
    tiles: GeoDataFrame,
    batch_size: int = 5,
    workers: int | None = 20,
) -> None:
    """Download dataset using tile-based sampling.

    Args:
        dataset (GeoDataset): The GeoDataset to download
        tiles (GeoDataFrame): GeoDataFrame containing tile geometries
        batch_size (int): Size of download batches
        workers (int | None): Number of parallel workers for downloading
    """
    # TODO: check if dataset is already downloaded
    dataset._download = True
    sampler = TileGeoSampler(dataset, tiles)
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=workers,
        collate_fn=stack_samples,
    )

    os.makedirs(dataset.paths, exist_ok=True)

    for batch in tqdm(dataloader):
        pass  # Process batches to trigger download

    logging.info(f"{dataset.__class__.__name__} data saved to {dataset.paths}")


def get_stats(
    dataset: GeoDataset,
    tiles: GeoDataFrame,
    nodata: int | None = None,
    path: str | None = None,
    overwrite: bool = False,
) -> dict:
    """Compute dataset statistics including mean and standard deviation.

    Args:
        dataset (GeoDataset): The dataset to compute stats for
        tiles (GeoDataFrame): GeoDataFrame containing tile geometries for sampling
        nodata (int | None): No data value to exclude from statistics
        overwrite (bool): Whether to overwrite existing stats

    Returns:
        dict: Dictionary containing computed statistics (mean, std, etc.)
    """

    sampler = TileGeoSampler(dataset, tiles=tiles)

    channels = 1
    if hasattr(dataset, "_bands"):
        channels = len(dataset._bands)

    if nodata is None:
        nodata = dataset.nodata

    stats = DatasetStats(
        dataset,
        sampler,
        path=path,
        batch_size=5,
        num_workers=40,
        channels=channels,
        nodata=nodata,
        overwrite=overwrite,
    )
    return stats.compute()


class eMapREmulatorDataModule(CloudDataModule):
    """LightningDataModule implementation to emulate eMapR AGLB data.

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
        >>> datamodule = eMapREmulatorDataModule(
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
        >>> datamodule = eMapREmulatorDataModule(
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
        >>> datamodule = eMapREmulatorDataModule(..., hparams=hparams)
    """

    input_stats = None
    target_stats = None

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
            >>> serialized = eMapREmulatorDataModule._serialize_stats(stats)
            >>> print(serialized)
            {'mean': [1.0, 2.0], 'std': [0.1, 0.2]}
        """
        serialized = {}
        for key, value in stats_dict.items():
            if isinstance(value, torch.Tensor):
                # Convert tensor to list for YAML compatibility
                serialized[key] = value.tolist()
            elif isinstance(value, dict):
                # Recursively serialize nested dictionaries
                serialized[key] = eMapREmulatorDataModule._serialize_stats(value)
            else:
                # Keep primitive types as-is
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
            >>> deserialized = eMapREmulatorDataModule._deserialize_stats(serialized)
            >>> print(deserialized)
            {'mean': tensor([1., 2.]), 'std': tensor([0.1000, 0.2000])}
        """
        deserialized = {}
        for key, value in stats_dict.items():
            if isinstance(value, list):
                # Convert list back to tensor
                deserialized[key] = torch.tensor(value)
            elif isinstance(value, dict):
                # Recursively deserialize nested dictionaries
                deserialized[key] = eMapREmulatorDataModule._deserialize_stats(value)
            else:
                # Keep primitive types as-is
                deserialized[key] = value
        return deserialized

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
        # prep mask
        return {
            "mask": collated["mask"],
            "image": collated["image"],
            "crs": collated["crs"],
            "bounds": collated["bbox"],
        }

    def __init__(
        self,
        root: str,
        year: int,
        batch_size: int = 30,
        patch_size: int | tuple[int, int] = 64,
        epoch_length: int | None = None,
        num_workers: int = 10,
        target_path: str | None = None,
        test_tiles_path: str | None = None,
        val_tiles_path: str | None = None,
        predict_tiles_path: str | None = None,
        train_tiles_path: str | None = None,
        ee_project: str | None = None,
        hparams: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a new eMapREmulatorDataModule instance.

        Args:
            root (str): Root directory for data storage.
            year (int): Year of training, validation, and test data.
            batch_size (int): Size of each mini-batch.
            patch_size (int | tuple[int, int]): Size of each patch, either ``size`` or ``(height, width)``.
            epoch_length (int | None): Length of each training epoch.
            num_workers (int): Number of workers for parallel data loading.
            target_path (str | None): Path to target dataset. If None, uses default eMapRAGB path.
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
            >>> datamodule = eMapREmulatorDataModule(root="data", year=2020)
            >>>
            >>> # Resume/inference run - loads from hparams
            >>> datamodule = eMapREmulatorDataModule(
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
        self.train_tiles_path = train_tiles_path
        self.val_tiles_path = val_tiles_path
        self.test_tiles_path = test_tiles_path
        self.predict_tiles_path = predict_tiles_path

        # Track if stats were loaded from hparams
        self.stats_from_hparams = False

        self.inputs_class = GEELandsatFTV
        self.init_transforms = ReplaceNodataVal(-32768, 0, on_key="image")

        super().__init__(
            dataset_class=self.inputs_class,
            batch_size=batch_size,
            patch_size=patch_size,
            length=epoch_length,
            num_workers=num_workers,
            **kwargs,
        )

    def setup_transforms(self) -> None:
        """Setup transforms for datasets.

        Configures normalization transforms for both input and target datasets
        based on precomputed statistics. First checks if stats are available in hparams,
        otherwise falls back to computed stats.

        Workflow:
        1. Check if hparams contains precomputed stats
        2. If yes, deserialize and use them (skip computation)
        3. If no, use existing computed stats or compute new ones
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

        # Apply transforms using available stats
        if self.target_stats is not None:
            self.target_dataset.transforms = v2.Compose(
                [
                    Normalize(
                        mean=self.target_stats["mean"],
                        std=self.target_stats["std"],
                        on_key="mask",
                        nodata=self.target_dataset.nodata,
                    ),
                ]
            )

        if self.input_stats is not None:
            transforms = v2.Compose(
                [
                    ReplaceNodataVal(-32768, 0, on_key="image"),
                    v2.Normalize(
                        mean=self.input_stats["mean"], std=self.input_stats["std"]
                    ),
                ]
            )

            # Only set transforms for datasets that exist
            if hasattr(self, "input_dataset") and self.input_dataset is not None:
                self.input_dataset.transforms = transforms
            if (
                hasattr(self, "val_input_dataset")
                and self.val_input_dataset is not None
            ):
                self.val_input_dataset.transforms = transforms
            if (
                hasattr(self, "test_input_dataset")
                and self.test_input_dataset is not None
            ):
                self.test_input_dataset.transforms = transforms
            if (
                hasattr(self, "predict_inputs_dataset")
                and self.predict_inputs_dataset is not None
            ):
                self.predict_inputs_dataset.transforms = transforms

            self.revert_inputs = Denormalize(
                mean=self.input_stats["mean"], std=self.input_stats["std"]
            )
            self.revert_target = Denormalize(
                mean=self.target_stats["mean"], std=self.target_stats["std"]
            )

    def prepare_data(self, overwrite: bool = False) -> None:
        """Prepare data by downloading and computing statistics if needed.

        Args:
            overwrite: Whether to overwrite existing data and statistics
        """
        # Setup training inputs to ensure datasets are created
        self.setup("fit")

        logging.info("Preparing training dataset...")

        # Skip stat computation if stats were already loaded from hparams
        if self.stats_from_hparams:
            logging.info("Skipping stat computation - using stats from hparams")
            return

        if self.input_stats is None or self.target_stats is None or overwrite:
            try:
                self.input_stats = get_stats(
                    self.input_dataset,
                    self.train_tiles.data,
                    nodata=0,
                    path=self.input_stats,
                    overwrite=overwrite,
                )
                self.target_stats = get_stats(
                    self.target_dataset,
                    self.train_tiles.data,
                    path=self.target_stats,
                    overwrite=overwrite,
                )

                # Serialize stats for logging to hparams
                self.serialized_input_stats = self._serialize_stats(self.input_stats)
                self.serialized_target_stats = self._serialize_stats(self.target_stats)

                # Log stats to trainer if available
                if hasattr(self, "trainer") and self.trainer is not None:
                    hparams_to_log = {
                        "datamodule": {
                            "input_stats": self.serialized_input_stats,
                            "target_stats": self.serialized_target_stats,
                        }
                    }
                    if (
                        hasattr(self.trainer, "logger")
                        and self.trainer.logger is not None
                    ):
                        self.trainer.logger.log_hyperparams(hparams_to_log)
                        logging.info("Statistics logged to trainer hyperparameters")

            except Exception as e:
                logging.error(f"Failed to compute statistics: {e}")
                raise

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader using TileGeoSampler.

        Returns:
            DataLoader: Training data loader
        """
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
        if self.val_sampler is None:
            self.val_sampler = TileGeoSampler(
                self.val_dataset, self.val_tiles.data, shuffle=False
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
        # Create target dataset
        if self.target_path:
            self.target_dataset = eMapRAGB(year=self.year, paths=self.target_path)
        else:
            self.target_dataset = eMapRAGB(year=self.year)

        if stage in ["fit", "validate"]:
            # Setup training and validation datasets
            self._setup_fit_stage()

        elif stage == "test":
            # Setup test dataset
            self._setup_test_stage()

        elif stage == "predict":
            if year is None:
                raise ValueError("year parameter required for predict stage")
            # Setup prediction dataset
            self._setup_predict_stage(year)

        # Setup transforms after datasets are created
        self.setup_transforms()

    def _setup_fit_stage(self) -> None:
        """Setup datasets for training and validation stages."""
        # Load tile geometries
        self.train_tiles = GPDFeatureCollection(
            os.path.join(self.root, self.train_tiles_path)
        )
        self.val_tiles = GPDFeatureCollection(
            os.path.join(self.root, self.val_tiles_path)
        )

        # Create input datasets
        dataset_class_name = self.inputs_class.__name__.lower()
        training_inputs_path = os.path.join(
            self.root, f"training/{dataset_class_name}/{self.year}"
        )
        validation_inputs_path = os.path.join(
            self.root, f"validation/{dataset_class_name}/{self.year}"
        )

        self.input_dataset = self.inputs_class(
            year=self.year,
            roi=self.train_tiles.bounds,
            path=training_inputs_path,
            transforms=self.init_transforms,
            download=True,
        )
        self.val_input_dataset = self.inputs_class(
            year=self.year,
            roi=self.val_tiles.bounds,
            path=validation_inputs_path,
            download=True,
        )

        # Create combined datasets for training and validation
        self.train_dataset = self.target_dataset & self.input_dataset
        self.val_dataset = self.target_dataset & self.val_input_dataset

    def _setup_test_stage(self) -> None:
        """Setup datasets for testing stage."""
        # Load test tile geometries
        self.test_tiles = GPDFeatureCollection(
            os.path.join(self.root, self.test_tiles_path)
        )

        # Create test input dataset
        dataset_class_name = self.inputs_class.__name__.lower()
        test_inputs_path = os.path.join(
            self.root, f"test/{dataset_class_name}/{self.year}"
        )

        self.test_input_dataset = self.inputs_class(
            year=self.year,
            roi=self.test_tiles.bounds,
            path=test_inputs_path,
            download=True,
        )

        # Create combined test dataset
        self.test_dataset = self.target_dataset & self.test_input_dataset

    def _setup_predict_stage(self, year: int) -> None:
        """Setup datasets for prediction stage.

        Args:
            year (int): The year to predict for
        """
        # Load prediction tile geometries
        self.predict_tiles = GPDFeatureCollection(
            os.path.join(self.root, self.predict_tiles_path)
        )

        # Create prediction input dataset
        dataset_class_name = self.inputs_class.__name__.lower()
        predict_inputs_path = os.path.join(self.root, f"predict/{dataset_class_name}")

        self.predict_inputs_dataset = AnyRasterDataset(
            glob="*.tif",
            crs=self.target_dataset.crs,
            paths=predict_inputs_path,
            res=30,
            is_image=True,
        )

        # Create combined prediction dataset
        self.predict_dataset = self.target_dataset & self.predict_inputs_dataset

    def cleanup(self) -> None:
        """Clean up loaded resources and statistics to free memory.

        Releases memory by clearing datasets, samplers, tiles, and statistics
        that were loaded during setup.
        """
        logging.info("Cleaning up data module resources")
        self.input_stats = None
        self.target_stats = None

        # Clear datasets if they exist
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

        # Clear samplers if they exist
        for attr in ["train_sampler", "val_sampler", "test_sampler", "predict_sampler"]:
            if hasattr(self, attr):
                setattr(self, attr, None)

        # Clear tiles if they exist
        for attr in ["train_tiles", "val_tiles", "test_tiles", "predict_tiles"]:
            if hasattr(self, attr):
                setattr(self, attr, None)
