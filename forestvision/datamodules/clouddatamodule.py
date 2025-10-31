"""Abstract base class for ForestVision data modules."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from torchgeo.datamodules import GeoDataModule


class CloudDataModule(GeoDataModule, ABC):
    """Abstract base class for ForestVision data modules.

    Defines the interface contract required for integration with the AGB CLI.
    All data modules must inherit from this class and implement the abstract methods.
    """

    # Required attributes for stats management
    input_stats: Optional[Dict[str, Any]] = None
    target_stats: Optional[Dict[str, Any]] = None

    @abstractmethod
    def setup(self, stage: str, year: Optional[int] = None) -> None:
        """Setup datasets for the specified stage.

        Args:
            stage: The stage to setup ('fit', 'validate', 'test', or 'predict')
            year: Required for 'predict' stage, specifies the prediction year

        Raises:
            ValueError: If year is not provided for predict stage
        """
        pass

    @abstractmethod
    def prepare_data(self, overwrite: bool = False) -> None:
        """Prepare data by downloading and computing statistics if needed.

        Args:
            overwrite: Whether to overwrite existing data and statistics
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up loaded resources and statistics to free memory."""
        pass

    def validate_interface(self) -> None:
        """Validate that the datamodule implements the required interface.

        Raises:
            RuntimeError: If required attributes or methods are missing
        """
        # Check required attributes
        required_attrs = ["input_stats", "target_stats"]
        for attr in required_attrs:
            if not hasattr(self, attr):
                raise RuntimeError(f"Missing required attribute: {attr}")

        # Check that attributes are not None (if they should be initialized)
        if self.input_stats is None:
            raise RuntimeError("input_stats must be initialized (can be None)")
        if self.target_stats is None:
            raise RuntimeError("target_stats must be initialized (can be None)")
