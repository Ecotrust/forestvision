"""
Optuna Hyperparameter Optimization for Focal Loss Parameters.

This script performs Bayesian optimization of focal loss hyperparameters
(focal_alpha, focal_gamma, focal_weight) for the MultiTaskUNet model,
optimizing for validation loss while freezing all other parameters.

Usage (HPO Mode):
    python scripts/optuna_hpo_focal.py --config data/dev/configs/gnn_v2/osugnn_v2.yaml --n-trials 50 --max-epochs 30

Usage (HPO with Pre-trained Checkpoint):
    python scripts/optuna_hpo_focal.py --config data/dev/configs/gnn_v2/osugnn_v2.yaml \\
        --ckpt-path /path/to/pretrained_model.ckpt \\
        --n-trials 50 --max-epochs 30

Usage (Resume Mode):
    python scripts/optuna_hpo_focal.py --config data/dev/configs/gnn_v2/osugnn_v2.yaml \\
        --resume-from-checkpoint optuna_logs/trial_5/checkpoints/best.ckpt \\
        --additional-epochs 45 --resume-output-dir continued_training/

Features:
    - Bayesian optimization with TPE sampler
    - Early pruning of unpromising trials
    - Multi-GPU support via DDP (optional)
    - Automatic logging to SQLite database
    - TensorBoard integration for visualization
    - Resume training from trial checkpoints
    - Fine-tune from pre-trained checkpoints with focal weight optimization
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

from forestvision.trainers.litunet import MultiTaskUNet, DeviceAwareFocalLoss
from forestvision.datamodules.fortypbadatamodule import ForTypesDataModule


def load_base_config(config_path: str) -> Dict[str, Any]:
    """Load the base configuration YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_search_space(
    trial: optuna.Trial, base_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Define the focal loss hyperparameter search space.

    Only optimizes focal loss parameters:
    - focal_alpha: class imbalance parameter
    - focal_gamma: focusing parameter
    - focal_weight: per-class weights (14 classes)

    All other parameters are frozen from base_config.

    Returns a dictionary of suggested hyperparameters.
    """
    search_space = {}

    # Focal Loss hyperparameters
    search_space["focal_alpha"] = trial.suggest_float("focal_alpha", 0.1, 0.1)

    search_space["focal_gamma"] = trial.suggest_float("focal_gamma", 2.9, 2.9)

    # focal_weight: per-class weights for 14 classes
    # Get number of classes from base config
    num_classes_per_task = (
        base_config.get("model", {})
        .get("init_args", {})
        .get("num_classes_per_task", [14, 1, 1, 1])
    )
    num_classes = num_classes_per_task[0]  # First task is classification

    focal_weights = []
    w0 = trial.suggest_float("focal_weight_0", 1.5, 1.5, log=True)
    w1 = trial.suggest_float("focal_weight_1", 1, 1.5, log=True)
    w2 = trial.suggest_float("focal_weight_2", 1, 3, log=True)
    w3 = trial.suggest_float("focal_weight_3", 1, 3, log=True)
    w4 = trial.suggest_float("focal_weight_4", 1, 3, log=True)
    w5 = trial.suggest_float("focal_weight_5", 0.5, 1, log=True)
    w6 = trial.suggest_float("focal_weight_6", 1, 2, log=True)
    w7 = trial.suggest_float("focal_weight_7", 3, 5, log=True)
    w8 = trial.suggest_float("focal_weight_8", 3, 5, log=True)
    w9 = trial.suggest_float("focal_weight_9", 2, 3, log=True)
    w10 = trial.suggest_float("focal_weight_10", 2, 3, log=True)
    w11 = trial.suggest_float("focal_weight_11", 0.5, 1, log=True)
    w12 = trial.suggest_float("focal_weight_12", 1, 2, log=True)
    w13 = trial.suggest_float("focal_weight_13", 1, 2, log=True)
    focal_weights.extend([w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13])

    search_space["focal_weight"] = focal_weights

    # Log the number of classes for debugging
    trial.set_user_attr("num_classes_optimized", num_classes)

    return search_space


def create_model_config(
    base_config: Dict[str, Any], search_space: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge base config with suggested focal loss hyperparameters.

    Returns a complete configuration dictionary for model initialization.
    All non-focal parameters are frozen from base_config.
    """
    model_config = base_config.get("model", {}).copy()
    init_args = model_config.get("init_args", {}).copy()

    # Update with search space values (only focal parameters)
    init_args.update(
        {
            "focal_alpha": search_space["focal_alpha"],
            "focal_gamma": search_space["focal_gamma"],
            "focal_weight": search_space["focal_weight"],
        }
    )

    model_config["init_args"] = init_args
    return model_config


def create_datamodule_config(
    base_config: Dict[str, Any], search_space: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Merge base config with suggested data hyperparameters.
    Preserves all dataset and transform configurations.
    """
    data_config = base_config.get("data", {}).copy()
    init_args = data_config.get("init_args", {}).copy()

    # Preserve all dataset and transform configs from base config
    # No data hyperparameters are optimized in this script
    init_args["input_datasets"] = init_args.get("input_datasets", [])
    init_args["target_datasets"] = init_args.get("target_datasets", [])
    init_args["input_transforms"] = init_args.get("input_transforms")
    init_args["target_transforms"] = init_args.get("target_transforms")
    init_args["train_transforms"] = init_args.get("train_transforms")
    init_args["post_aug_input_transforms"] = init_args.get("post_aug_input_transforms")
    init_args["post_aug_target_transforms"] = init_args.get(
        "post_aug_target_transforms"
    )

    data_config["init_args"] = init_args
    return data_config


def resume_from_checkpoint(
    checkpoint_path: str,
    base_config: Dict[str, Any],
    additional_epochs: int,
    output_dir: str,
    accelerator: str = "gpu",
    devices: int = 1,
    patience: int = 10,
    monitor_metric: str = "val_loss",
) -> None:
    """
    Resume training from a checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        base_config: Base configuration dictionary
        additional_epochs: Number of additional epochs to train
        output_dir: Directory to save continued training logs
        accelerator: Accelerator to use ("gpu" or "cpu")
        devices: Number of devices to use
        patience: Early stopping patience
        monitor_metric: Metric to monitor for early stopping
    """
    print("\n" + "=" * 60)
    print("RESUMING TRAINING FROM CHECKPOINT")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Additional epochs: {additional_epochs}")
    print(f"Output directory: {output_dir}")
    print("=" * 60 + "\n")

    # Verify checkpoint exists
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load model from checkpoint
    print("Loading model from checkpoint...")
    model = MultiTaskUNet.load_from_checkpoint(checkpoint_path)

    # Get current epoch
    current_epoch = model.current_epoch
    target_epochs = current_epoch + additional_epochs
    print(f"Current epoch: {current_epoch}")
    print(f"Target epochs: {target_epochs}")

    # Setup datamodule from base config
    print("Setting up datamodule...")
    data_config = create_datamodule_config(base_config)
    module_path, class_name = data_config["class_path"].rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    datamodule_class = getattr(module, class_name)
    datamodule = datamodule_class(**data_config["init_args"])

    # Setup logger
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger = TensorBoardLogger(
        save_dir=output_dir,
        name="continued",
        version="",
    )

    # Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor=monitor_metric,
            patience=patience,
            mode="min",
            verbose=True,
        ),
        ModelCheckpoint(
            monitor=monitor_metric,
            mode="min",
            save_top_k=1,
            filename="best",
            enable_version_counter=False,
        ),
    ]

    # Initialize trainer
    print("Initializing trainer...")
    trainer = Trainer(
        max_epochs=target_epochs,
        accelerator=accelerator,
        devices=devices,
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=10,
    )

    # Continue training
    print(f"\nStarting training from epoch {current_epoch} to {target_epochs}...")
    start_time = time.time()

    trainer.fit(model, datamodule=datamodule, ckpt_path=checkpoint_path)

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s")

    # Print final metrics
    print("\n" + "=" * 60)
    print("FINAL METRICS")
    print("=" * 60)
    for key, value in trainer.callback_metrics.items():
        if isinstance(value, torch.Tensor):
            value = value.item()
        print(f"  {key}: {value:.6f}")
    print("=" * 60)

    # Save final checkpoint info
    info_path = Path(output_dir) / "continued" / "training_info.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    training_info = {
        "original_checkpoint": checkpoint_path,
        "current_epoch": current_epoch,
        "additional_epochs": additional_epochs,
        "target_epochs": target_epochs,
        "final_epoch": model.current_epoch,
        "elapsed_time": elapsed,
        "final_metrics": {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in trainer.callback_metrics.items()
        },
    }
    with open(info_path, "w") as f:
        json.dump(training_info, f, indent=2)
    print(f"\nTraining info saved to: {info_path}")


class OptunaObjective:
    """
    Callable objective class for Optuna optimization.

    This class encapsulates the training logic and configuration,
    making it pickleable for distributed training.
    """

    def __init__(
        self,
        base_config: Dict[str, Any],
        max_epochs: int = 5,
        accelerator: str = "gpu",
        devices: int = 1,
        patience: int = 3,
        monitor_metric: str = "val_loss",
        ckpt_path: Optional[str] = None,
    ):
        self.base_config = base_config
        self.max_epochs = max_epochs
        self.accelerator = accelerator
        self.devices = devices
        self.patience = patience
        self.monitor_metric = monitor_metric
        self.ckpt_path = ckpt_path

    def _update_focal_loss_params(self, model: MultiTaskUNet, search_space: Dict[str, Any]) -> None:
        """Update focal loss parameters on a loaded or initialized model.
        
        This updates both the hyperparameters and the criterion with new
        focal loss weights suggested by Optuna.
        
        Args:
            model: The MultiTaskUNet model to update
            search_space: Dictionary containing focal_alpha, focal_gamma, focal_weight
        """
        # Update hyperparameters
        model.hparams.update({
            "focal_alpha": search_space["focal_alpha"],
            "focal_gamma": search_space["focal_gamma"],
            "focal_weight": search_space["focal_weight"],
        })
        
        # Update the focal loss criterion with new weights
        focal_weight = search_space["focal_weight"]
        if focal_weight is not None:
            if not isinstance(focal_weight, torch.Tensor):
                focal_weight = torch.tensor(focal_weight, dtype=torch.float32)
            # Update the registered buffer
            if hasattr(model, '_focal_weight'):
                model._focal_weight = focal_weight.to(model.device)
            else:
                model.register_buffer("_focal_weight", focal_weight)
        
        # Re-initialize the focal loss criterion with new parameters
        model.focal_loss = DeviceAwareFocalLoss(
            alpha=search_space["focal_alpha"],
            gamma=search_space["focal_gamma"],
            reduction="mean",
            weight=focal_weight if focal_weight is not None else None,
            ignore_index=model.hparams.get("ignore_index", -100),
        )

    def __call__(self, trial: optuna.Trial) -> float:
        """
        Execute one training trial with suggested focal loss hyperparameters.

        Args:
            trial: Optuna trial object

        Returns:
            Validation loss for the trial
        """
        # Set seed for reproducibility within trial
        seed_everything(42 + trial.number)

        # Get suggested hyperparameters (only focal loss params)
        search_space = get_search_space(trial, self.base_config)

        # Log trial parameters
        trial.set_user_attr(
            "focal_params",
            {
                "focal_alpha": search_space["focal_alpha"],
                "focal_gamma": search_space["focal_gamma"],
                "focal_weight": search_space["focal_weight"],
            },
        )
        
        # Log checkpoint path if using pre-trained model
        if self.ckpt_path:
            trial.set_user_attr("ckpt_path", self.ckpt_path)

        # Create model and datamodule configurations
        model_config = create_model_config(self.base_config, search_space)
        data_config = create_datamodule_config(self.base_config, search_space)

        try:
            # Initialize model (from checkpoint or from scratch)
            if self.ckpt_path and Path(self.ckpt_path).exists():
                print(f"Trial {trial.number}: Loading checkpoint from {self.ckpt_path}")
                model = MultiTaskUNet.load_from_checkpoint(self.ckpt_path)
                # Apply Optuna-suggested focal weights to the loaded model
                self._update_focal_loss_params(model, search_space)
                print(f"Trial {trial.number}: Applied focal weights from search space")
            else:
                # Initialize from scratch
                if self.ckpt_path:
                    print(f"Trial {trial.number}: Checkpoint not found at {self.ckpt_path}, initializing from scratch")
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
                PyTorchLightningPruningCallback(trial, monitor=self.monitor_metric),
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
                self.monitor_metric, float("inf")
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
        n_warmup_steps=5,  # Don't prune before epoch 5
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
    print("OPTUNA FOCAL LOSS OPTIMIZATION RESULTS")
    print("=" * 60)

    print(f"\nNumber of trials: {len(study.trials)}")
    print(
        f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}"
    )
    print(
        f"Number of completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}"
    )

    if study.best_trial is not None:
        print(f"\nBest trial: {study.best_trial.number}")
        print(f"Best value: {study.best_value:.6f}")
        print("\nBest focal loss parameters:")

        # Print focal_alpha and focal_gamma
        if "focal_alpha" in study.best_params:
            print(f"  focal_alpha: {study.best_params['focal_alpha']:.4f}")
        if "focal_gamma" in study.best_params:
            print(f"  focal_gamma: {study.best_params['focal_gamma']:.4f}")

        # Print focal_weight as a list
        focal_weights = []
        weight_keys = [
            k for k in study.best_params.keys() if k.startswith("focal_weight_")
        ]
        weight_keys_sorted = sorted(weight_keys, key=lambda x: int(x.split("_")[-1]))
        for key in weight_keys_sorted:
            focal_weights.append(study.best_params[key])
        print(f"  focal_weight: {focal_weights}")

        # Print additional metrics from best trial
        print("\nBest trial metrics:")
        for key, value in study.best_trial.user_attrs.items():
            if key.startswith("metric_"):
                metric_name = key.replace("metric_", "")
                if isinstance(value, float):
                    print(f"  {metric_name}: {value:.6f}")
                else:
                    print(f"  {metric_name}: {value}")

    print("=" * 60)


def save_best_config(
    study: optuna.Study, output_path: str, base_config: Dict[str, Any]
):
    """Save the best configuration to a YAML file."""
    if study.best_trial is None:
        print("No best trial found. Skipping config save.")
        return

    # Get best hyperparameters
    best_params = study.best_params.copy()

    # Reconstruct focal_weight from individual weight parameters
    focal_weights = []
    weight_keys = [k for k in best_params.keys() if k.startswith("focal_weight_")]
    weight_keys_sorted = sorted(weight_keys, key=lambda x: int(x.split("_")[-1]))
    for key in weight_keys_sorted:
        focal_weights.append(best_params[key])

    # Create search space with reconstructed focal_weight
    search_space = {
        "focal_alpha": best_params.get("focal_alpha", 0.25),
        "focal_gamma": best_params.get("focal_gamma", 2.0),
        "focal_weight": focal_weights,
    }

    # Create complete config
    model_config = create_model_config(base_config, search_space)
    data_config = create_datamodule_config(base_config, search_space)

    best_config = {
        "trainer": base_config.get("trainer", {}),
        "model": model_config,
        "data": data_config,
    }

    # Update trainer config
    best_config["trainer"].update(
        {
            "max_epochs": base_config.get("trainer", {}).get("max_epochs", 30),
        }
    )

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(best_config, f, default_flow_style=False, sort_keys=False)

    print(f"\nBest configuration saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Optuna Hyperparameter Optimization for Focal Loss Parameters"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="data/dev/configs/gnn_v2/osugnn_v2.yaml",
        help="Path to base configuration YAML file",
    )

    # HPO Mode Arguments
    hpo_group = parser.add_argument_group("HPO Mode")
    hpo_group.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of optimization trials to run",
    )
    hpo_group.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds for optimization",
    )
    hpo_group.add_argument(
        "--study-name",
        type=str,
        default="focal_loss_hpo",
        help="Name of the Optuna study",
    )
    hpo_group.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Database URL for study persistence (e.g., sqlite:///optuna_focal.db)",
    )
    hpo_group.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="Path to pre-trained checkpoint to load before optimizing focal weights",
    )

    # Resume Mode Arguments
    resume_group = parser.add_argument_group("Resume Mode")
    resume_group.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from (enables resume mode)",
    )
    resume_group.add_argument(
        "--additional-epochs",
        type=int,
        default=50,
        help="Number of additional epochs to train when resuming",
    )
    resume_group.add_argument(
        "--resume-output-dir",
        type=str,
        default="continued_training",
        help="Output directory for resumed training logs",
    )

    # Common Training Arguments
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=30,
        help="Maximum number of epochs per trial (HPO mode)",
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
        default="configs/focal_best.yaml",
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

    # RESUME MODE
    if args.resume_from_checkpoint:
        resume_from_checkpoint(
            checkpoint_path=args.resume_from_checkpoint,
            base_config=base_config,
            additional_epochs=args.additional_epochs,
            output_dir=args.resume_output_dir,
            accelerator=args.accelerator,
            devices=args.devices,
            patience=args.patience,
            monitor_metric=args.monitor,
        )
        return

    # HPO MODE
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
        ckpt_path=args.ckpt_path,
    )

    # Run optimization
    if args.ckpt_path:
        print(f"\nUsing pre-trained checkpoint: {args.ckpt_path}")
    print(f"\nStarting optimization: {args.n_trials} trials")
    print(f"Optimizing: focal_alpha, focal_gamma, focal_weight (14 classes)")
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
