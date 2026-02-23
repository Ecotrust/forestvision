"""
Optuna Hyperparameter Optimization for MultiTaskUNet.

This script performs Bayesian optimization of hyperparameters for the forestvision
MultiTaskUNet model, optimizing for validation loss across segmentation and regression tasks.

Usage:
    python scripts/optuna_hpo.py --config configs/optuna_base.yaml --n-trials 20 --timeout 14400

Features:
    - Bayesian optimization with TPE sampler
    - Early pruning of unpromising trials
    - Multi-GPU support via DDP (optional)
    - Automatic logging to SQLite database
    - TensorBoard integration for visualization
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

import torch
import yaml
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from forestvision.trainers.litunet import MultiTaskUNet
from forestvision.datamodules.fortypbadatamodule import ForTypesDataModule


def load_base_config(config_path: str) -> Dict[str, Any]:
    """Load the base configuration YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_search_space(trial: optuna.Trial, base_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Define the hyperparameter search space.
    
    Returns a dictionary of suggested hyperparameters based on the trial.
    """
    search_space = {}
    
    # Model hyperparameters
    search_space["lr"] = trial.suggest_float(
        "lr", 
        1e-5, 
        1e-2, 
        log=True
    )
    
    search_space["dropout"] = trial.suggest_float(
        "dropout", 
        0.1, 
        0.8
    )
   
    search_space["weight_decay"] = trial.suggest_float(
        "weight_decay", 
        1e-6, 
        1e-2, 
        log=True
    )
    
    search_space["scheduler_patience"] = trial.suggest_int(
        "scheduler_patience", 
        2, 
        10
    )
    
    search_space["scheduler_factor"] = trial.suggest_float(
        "scheduler_factor", 
        0.1, 
        0.8
    )

    # Focal Loss hyperparameters
    search_space["focal_alpha"] = trial.suggest_float(
        "focal_alpha",
        0.1,
        1.0
    )

    search_space["focal_gamma"] = trial.suggest_float(
        "focal_gamma",
        1.0,
        5.0
    )

    # Regression loss and uncertainty weighting hyperparameters
    search_space["sharploss_alpha"] = trial.suggest_float(
        "sharploss_alpha",
        0.0,
        1.0
    )

    search_space["loss_norm_momentum"] = trial.suggest_float(
        "loss_norm_momentum",
        0.5,
        0.99
    )

    # Data hyperparameters
    search_space["batch_size"] = trial.suggest_categorical(
        "batch_size", 
        [16, 32, 48]
    )

    # Constrained loss weighting: seg_loss_weight + sum(reg_loss_weights) = 1.0
    # Sample segmentation weight, then distribute remaining equally among regression channels
    seg_weight = trial.suggest_float("seg_loss_weight", 0.3, 0.9)
    
    # Get number of regression targets from num_classes_per_task
    # First task is classification, rest are regression
    num_classes_per_task = base_config.get("model", {}).get("init_args", {}).get("num_classes_per_task", [15, 1, 1, 1])
    num_reg_targets = sum(1 for nc in num_classes_per_task[1:] if nc == 1)  # Count regression tasks
    
    # Compute remaining weight for regression and distribute equally
    remaining_weight = 1.0 - seg_weight
    reg_weight_per_channel = remaining_weight / num_reg_targets if num_reg_targets > 0 else 0
    reg_weights = [reg_weight_per_channel] * num_reg_targets
    
    search_space["seg_loss_weight"] = seg_weight
    search_space["reg_loss_weights"] = reg_weights
    
    # Log the constraint for debugging
    total_weight = seg_weight + sum(reg_weights)
    trial.set_user_attr("total_loss_weight", total_weight)
    
    return search_space


def create_model_config(
    base_config: Dict[str, Any], 
    search_space: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge base config with suggested hyperparameters.
    
    Returns a complete configuration dictionary for model initialization.
    """
    model_config = base_config.get("model", {}).copy()
    init_args = model_config.get("init_args", {}).copy()
    
    # Update with search space values
    init_args.update({
        "lr": search_space["lr"],
        "dropout": search_space["dropout"],
        "weight_decay": search_space["weight_decay"],
        "scheduler_patience": search_space["scheduler_patience"],
        "scheduler_factor": search_space["scheduler_factor"],
        "focal_alpha": search_space["focal_alpha"],
        "focal_gamma": search_space["focal_gamma"],
        "sharploss_alpha": search_space["sharploss_alpha"],
        "loss_norm_momentum": search_space["loss_norm_momentum"],
        "seg_loss_weight": search_space["seg_loss_weight"],
        "reg_loss_weights": search_space["reg_loss_weights"],
    })
    
    model_config["init_args"] = init_args
    return model_config


def create_datamodule_config(
    base_config: Dict[str, Any], 
    search_space: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge base config with suggested data hyperparameters.
    Preserves all dataset and transform configurations.
    """
    data_config = base_config.get("data", {}).copy()
    init_args = data_config.get("init_args", {}).copy()
    
    # Update batch size
    init_args["batch_size"] = search_space["batch_size"]
    
    # Ensure all dataset and transform configs are preserved
    # These are required for the new ForTypesDataModule
    init_args["input_datasets"] = init_args.get("input_datasets", [])
    init_args["target_datasets"] = init_args.get("target_datasets", [])
    init_args["input_transforms"] = init_args.get("input_transforms")
    init_args["target_transforms"] = init_args.get("target_transforms")
    init_args["train_transforms"] = init_args.get("train_transforms")
    
    data_config["init_args"] = init_args
    return data_config


class OptunaObjective:
    """
    Callable objective class for Optuna optimization.
    
    This class encapsulates the training logic and configuration,
    making it pickleable for distributed training.
    """
    
    def __init__(
        self,
        base_config: Dict[str, Any],
        max_epochs: int = 30,
        accelerator: str = "gpu",
        devices: int = 1,
        patience: int = 3,
        monitor_metric: str = "val_loss",
    ):
        self.base_config = base_config
        self.max_epochs = max_epochs
        self.accelerator = accelerator
        self.devices = devices
        self.patience = patience
        self.monitor_metric = monitor_metric
    
    def __call__(self, trial: optuna.Trial) -> float:
        """
        Execute one training trial with suggested hyperparameters.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Validation loss for the trial
        """
        # Set seed for reproducibility within trial
        seed_everything(42 + trial.number)
        
        # Get suggested hyperparameters
        search_space = get_search_space(trial, self.base_config)
        
        # Log trial parameters
        trial.set_user_attr("config", search_space)
        
        # Create model and datamodule configurations
        model_config = create_model_config(self.base_config, search_space)
        data_config = create_datamodule_config(self.base_config, search_space)
        
        try:
            # Initialize model
            model = MultiTaskUNet(**model_config["init_args"])
            
            # Initialize datamodule
            datamodule_class = self._get_class(data_config["class_path"])
            datamodule = datamodule_class(**data_config["init_args"])
            
            # Setup logger
            logger = TensorBoardLogger(
                save_dir="optuna_logs",
                name=f"trial_{trial.number}",
                version="",
            )
            
            # Setup callbacks
            callbacks = [
                # Early stopping within trial
                EarlyStopping(
                    monitor=self.monitor_metric,
                    patience=self.patience,
                    mode="min",
                    verbose=False,
                ),
                # Model checkpoint
                ModelCheckpoint(
                    monitor=self.monitor_metric,
                    mode="min",
                    save_top_k=1,
                    filename="best",
                    enable_version_counter=False,
                ),
                # Optuna pruning callback
                PyTorchLightningPruningCallback(
                    trial, 
                    monitor=self.monitor_metric
                ),
            ]
            
            # Initialize trainer
            trainer = Trainer(
                max_epochs=self.max_epochs,
                accelerator=self.accelerator,
                devices=self.devices,
                logger=logger,
                callbacks=callbacks,
                enable_progress_bar=False,  # Reduce output noise
                enable_model_summary=False,
                log_every_n_steps=10,
            )
            
            # Train
            trainer.fit(model, datamodule=datamodule)
            
            # Get best validation loss
            best_metric = trainer.callback_metrics.get(
                self.monitor_metric, 
                float("inf")
            )
            
            # If tensor, convert to float
            if isinstance(best_metric, torch.Tensor):
                best_metric = best_metric.item()
            
            # Log additional metrics as user attributes
            for key, value in trainer.callback_metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                trial.set_user_attr(f"metric_{key}", value)
            
            return best_metric
            
        except Exception as e:
            # Log error and return inf (Optuna will prune this trial)
            trial.set_user_attr("error", str(e))
            print(f"Trial {trial.number} failed: {e}")
            return float("inf")
    
    def _get_class(self, class_path: str):
        """Dynamically import a class from module path."""
        module_path, class_name = class_path.rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)


def create_study(
    study_name: str,
    storage: Optional[str] = None,
    direction: str = "minimize",
) -> optuna.Study:
    """
    Create or load an Optuna study.
    
    Args:
        study_name: Name of the study
        storage: Database URL for persistence (None for in-memory)
        direction: Optimization direction ("minimize" or "maximize")
        
    Returns:
        Optuna study object
    """
    # Create sampler and pruner
    sampler = TPESampler(
        n_startup_trials=5,  # Random sampling for first 5 trials
        n_ei_candidates=24,
        seed=42,
    )
    
    pruner = MedianPruner(
        n_startup_trials=3,  # Don't prune first 3 trials
        n_warmup_steps=10,   # Don't prune before epoch 10
        interval_steps=1,
    )
    
    # Create study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction=direction,
        load_if_exists=True,
    )
    
    return study


def print_study_results(study: optuna.Study):
    """Print summary of study results."""
    print("\n" + "=" * 60)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION RESULTS")
    print("=" * 60)
    
    print(f"\nNumber of trials: {len(study.trials)}")
    print(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Number of completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    
    if study.best_trial is not None:
        print(f"\nBest trial: {study.best_trial.number}")
        print(f"Best value: {study.best_value:.6f}")
        print("\nBest hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        # Print additional metrics from best trial
        print("\nBest trial metrics:")
        for key, value in study.best_trial.user_attrs.items():
            if key.startswith("metric_"):
                metric_name = key.replace("metric_", "")
                print(f"  {metric_name}: {value:.6f}")
    
    print("=" * 60)


def save_best_config(study: optuna.Study, output_path: str, base_config: Dict[str, Any]):
    """Save the best configuration to a YAML file."""
    if study.best_trial is None:
        print("No best trial found. Skipping config save.")
        return
    
    # Get best hyperparameters
    best_params = study.best_params.copy()
    
    # Reconstruct reg_loss_weights from seg_loss_weight if needed
    if "seg_loss_weight" in best_params and "reg_loss_weights" not in best_params:
        seg_weight = best_params["seg_loss_weight"]
        # Use num_classes_per_task to determine number of regression targets
        num_classes_per_task = base_config.get("model", {}).get("init_args", {}).get("num_classes_per_task", [15, 1, 1, 1])
        num_reg_targets = sum(1 for nc in num_classes_per_task[1:] if nc == 1)
        remaining_weight = 1.0 - seg_weight
        reg_weight_per_channel = remaining_weight / num_reg_targets if num_reg_targets > 0 else 0
        best_params["reg_loss_weights"] = [reg_weight_per_channel] * num_reg_targets
    
    # Create complete config
    model_config = create_model_config(base_config, best_params)
    data_config = create_datamodule_config(base_config, best_params)
    
    best_config = {
        "trainer": base_config.get("trainer", {}),
        "model": model_config,
        "data": data_config,
    }
    
    # Update trainer config
    best_config["trainer"].update({
        "max_epochs": base_config.get("trainer", {}).get("max_epochs", 30),
    })
    
    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        yaml.dump(best_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nBest configuration saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Optuna Hyperparameter Optimization for MultiTaskUNet"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/optuna_base.yaml",
        help="Path to base configuration YAML file",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of optimization trials to run",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds for optimization",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="multitask_unet_hpo",
        help="Name of the Optuna study",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Database URL for study persistence (e.g., sqlite:///optuna.db)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=30,
        help="Maximum number of epochs per trial",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        choices=["gpu", "cpu"],
        help="Accelerator to use for training",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of devices (GPUs) to use",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience within each trial",
    )
    parser.add_argument(
        "--monitor",
        type=str,
        default="val_loss",
        help="Metric to monitor for optimization",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="configs/osugnn_best.yaml",
        help="Path to save best configuration",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    args = parser.parse_args()
    
    # Set global seed
    seed_everything(args.seed)
    
    # Load base configuration
    print(f"Loading base configuration from: {args.config}")
    base_config = load_base_config(args.config)
    
    # Create study
    print(f"Creating study: {args.study_name}")
    study = create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
    )
    
    # Create objective function
    objective = OptunaObjective(
        base_config=base_config,
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        patience=args.patience,
        monitor_metric=args.monitor,
    )
    
    # Run optimization
    print(f"\nStarting optimization: {args.n_trials} trials")
    print(f"Max epochs per trial: {args.max_epochs}")
    print(f"Timeout: {args.timeout}s" if args.timeout else "Timeout: None")
    print("-" * 60)
    
    start_time = time.time()
    
    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )
    
    elapsed = time.time() - start_time
    print(f"\nOptimization completed in {elapsed:.1f}s")
    
    # Print results
    print_study_results(study)
    
    # Save best configuration
    save_best_config(study, args.output, base_config)
    
    # Save study statistics
    stats_path = Path(args.output).parent / f"{args.study_name}_stats.json"
    study_stats = {
        "study_name": args.study_name,
        "n_trials": len(study.trials),
        "best_trial": study.best_trial.number if study.best_trial else None,
        "best_value": study.best_value if study.best_trial else None,
        "best_params": study.best_params if study.best_trial else None,
        "elapsed_time": elapsed,
    }
    
    with open(stats_path, "w") as f:
        json.dump(study_stats, f, indent=2)
    
    print(f"Study statistics saved to: {stats_path}")


if __name__ == "__main__":
    main()
