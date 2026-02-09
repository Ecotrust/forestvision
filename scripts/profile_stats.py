"""
Profile input_stats and target_stats in MultiTaskUNet.

This script verifies that mean and std are applied to the correct input and target
channels across the entire training and validation pipelines.
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from forestvision.datamodules import ForTypesDataModule
from forestvision.trainers.litunet import MultiTaskUNet

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def profile_stats_from_config(config_path: str, root_dir: str = "data"):
    """
    Profile input_stats and target_stats from a configuration file.
    
    Args:
        config_path: Path to YAML config file
        root_dir: Root directory for data
    """
    import yaml
    
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Initialize DataModule
    data_config = config.get("data", {}).get("init_args", {})
    data_config["root"] = root_dir
    
    logger.info("\n" + "="*70)
    logger.info("STEP 1: Initializing ForTypesDataModule")
    logger.info("="*70)
    
    datamodule = ForTypesDataModule(**data_config)
    
    # Setup to trigger stats loading
    logger.info("\n--- Setting up datamodule (fit stage) ---")
    datamodule.setup("fit")
    
    # Profile loaded stats
    profile_datamodule_stats(datamodule)
    
    # Initialize MultiTaskUNet model
    logger.info("\n" + "="*70)
    logger.info("STEP 2: Initializing MultiTaskUNet Model")
    logger.info("="*70)
    
    model_config = config.get("model", {}).get("init_args", {})
    model = MultiTaskUNet(**model_config)
    
    # Profile model's understanding of stats
    profile_model_stats(model, datamodule)
    
    # Verify dataloader batches
    logger.info("\n" + "="*70)
    logger.info("STEP 3: Verifying Batches from DataLoaders")
    logger.info("="*70)
    
    verify_dataloader_batches(datamodule, model)
    
    return datamodule, model


def profile_datamodule_stats(datamodule):
    """Profile statistics loaded by the DataModule."""
    logger.info("\n--- DataModule Statistics Profile ---")
    
    # Input stats
    if hasattr(datamodule, "input_stats") and datamodule.input_stats is not None:
        input_mean = datamodule.input_stats.get("mean")
        input_std = datamodule.input_stats.get("std")

        logger.info(f"\nInput Stats:")
        logger.info(
            f"  Mean shape: {input_mean.shape if isinstance(input_mean, torch.Tensor) else 'N/A'}"
        )
        logger.info(
            f"  Std shape:  {input_std.shape if isinstance(input_std, torch.Tensor) else 'N/A'}"
        )
        logger.info(
            f"  Num input channels: {len(input_mean) if isinstance(input_mean, torch.Tensor) else 'N/A'}"
        )

        if isinstance(input_mean, torch.Tensor):
            logger.info(f"  Input mean values: {input_mean.tolist()}")
            logger.info(f"  Input std values:  {input_std.tolist()}")

        # Print per-config stats
        for i, cfg in enumerate(datamodule.input_configs):
            logger.info(f"  - Config {i} ({cfg.dataset_class.__name__}):")
            logger.info(f"    mean: {cfg.mean}")
            logger.info(f"    std:  {cfg.std}")
    else:
        logger.warning("  No input_stats found in datamodule!")
    
    # Target stats
    if hasattr(datamodule, 'target_stats') and datamodule.target_stats is not None:
        target_mean = datamodule.target_stats.get('mean')
        target_std = datamodule.target_stats.get('std')
        
        logger.info(f"\nTarget Stats:")
        logger.info(f"  Mean shape: {target_mean.shape if isinstance(target_mean, torch.Tensor) else 'N/A'}")
        logger.info(f"  Std shape:  {target_std.shape if isinstance(target_std, torch.Tensor) else 'N/A'}")
        logger.info(f"  Num target channels: {len(target_mean) if isinstance(target_mean, torch.Tensor) else 'N/A'}")
        
        if isinstance(target_mean, torch.Tensor):
            logger.info(f"  Target mean values: {target_mean.tolist()}")
            logger.info(f"  Target std values:  {target_std.tolist()}")
            
            # Identify target bands from config
            if hasattr(datamodule, 'target_configs'):
                for i, cfg in enumerate(datamodule.target_configs):
                    logger.info(f"\n  Target dataset {i}: {cfg.dataset_class.__name__}")
                    logger.info(f"    Bands: {cfg.bands}")
                    for j, band in enumerate(cfg.bands):
                        idx = j  # Simplified - assumes single dataset
                        if idx < len(target_mean):
                            logger.info(f"      Channel {idx} ({band}): mean={target_mean[idx]:.4f}, std={target_std[idx]:.4f}")
    else:
        logger.warning("  No target_stats found in datamodule!")
    
    # Input datasets breakdown
    if hasattr(datamodule, 'input_configs'):
        logger.info(f"\nInput Datasets Configuration:")
        total_expected_channels = 0
        for i, cfg in enumerate(datamodule.input_configs):
            num_bands = len(cfg.bands)
            total_expected_channels += num_bands
            logger.info(f"  Dataset {i}: {cfg.dataset_class.__name__}")
            logger.info(f"    Bands ({num_bands}): {cfg.bands}")
            if hasattr(cfg, 'transforms') and cfg.transforms:
                logger.info(f"    Transforms: {cfg.transforms}")
        logger.info(f"  Total expected input channels: {total_expected_channels}")


def profile_model_stats(model, datamodule):
    """Profile how the model sees and uses stats."""
    logger.info("\n--- Model Statistics Profile ---")
    
    # Check hparams
    input_stats = model.hparams.get('input_stats')
    target_stats = model.hparams.get('target_stats')
    
    logger.info(f"\nModel hparams:")
    logger.info(f"  input_stats in hparams: {input_stats is not None}")
    logger.info(f"  target_stats in hparams: {target_stats is not None}")
    
    if input_stats:
        mean = input_stats.get('mean')
        std = input_stats.get('std')
        logger.info(f"  Input mean (from hparams): {mean}")
        logger.info(f"  Input std (from hparams):  {std}")
        logger.info(f"  Num channels: {len(mean) if mean else 'N/A'}")
    
    if target_stats:
        mean = target_stats.get('mean')
        std = target_stats.get('std')
        logger.info(f"  Target mean (from hparams): {mean}")
        logger.info(f"  Target std (from hparams):  {std}")
        logger.info(f"  Num channels: {len(mean) if mean else 'N/A'}")
    
    # Check if model can access datamodule stats
    can_access = False
    try:
        can_access = hasattr(model, 'trainer') and model.trainer is not None
    except RuntimeError:
        pass
    logger.info(f"\nModel can access datamodule: {can_access}")


def verify_dataloader_batches(datamodule, model, num_batches=2):
    """Verify actual batches from dataloaders."""
    logger.info("\n--- Training DataLoader Verification ---")
    
    try:
        train_loader = datamodule.train_dataloader()
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= num_batches:
                break
            
            logger.info(f"\nBatch {batch_idx}:")
            image = batch['image']
            mask = batch['mask']
            
            logger.info(f"  Image shape: {image.shape} (B, C, H, W)")
            logger.info(f"  Mask shape:  {mask.shape} (B, C, H, W)")
            
            # Verify channel counts match stats
            num_input_channels = image.shape[1]
            num_target_channels = mask.shape[1]
            
            if hasattr(datamodule, 'input_stats') and datamodule.input_stats:
                expected_input = len(datamodule.input_stats['mean'])
                if num_input_channels != expected_input:
                    logger.error(f"  MISMATCH: Image has {num_input_channels} channels, but input_stats has {expected_input}")
                else:
                    logger.info(f"  Input channels match: {num_input_channels}")
            
            if hasattr(datamodule, 'target_stats') and datamodule.target_stats:
                expected_target = len(datamodule.target_stats['mean'])
                if num_target_channels != expected_target:
                    logger.error(f"  MISMATCH: Mask has {num_target_channels} channels, but target_stats has {expected_target}")
                else:
                    logger.info(f"  Target channels match: {num_target_channels}")
            
            # Check value ranges (after normalization, should be roughly -1 to 1)
            logger.info(f"  Image value range: [{image.min():.4f}, {image.max():.4f}]")
            logger.info(f"  Mask value range:  [{mask.min():.4f}, {mask.max():.4f}]")
            
            # Per-channel stats for image
            logger.info(f"  Image per-channel means: {[f'{m:.4f}' for m in image.mean(dim=[0,2,3]).tolist()]}")
            logger.info(f"  Image per-channel stds:  {[f'{s:.4f}' for s in image.std(dim=[0,2,3]).tolist()]}")
            
            # Per-channel stats for mask (excluding nodata)
            for c in range(mask.shape[1]):
                channel_data = mask[:, c]
                # Exclude -1 (nodata)
                valid = channel_data[channel_data != -1]
                if len(valid) > 0:
                    logger.info(f"  Mask channel {c} valid range: [{valid.min():.4f}, {valid.max():.4f}], mean: {valid.mean():.4f}")
    
    except Exception as e:
        logger.error(f"Error verifying training dataloader: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("\n--- Validation DataLoader Verification ---")
    
    try:
        val_loader = datamodule.val_dataloader()
        
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= num_batches:
                break
            
            logger.info(f"\nValidation Batch {batch_idx}:")
            image = batch['image']
            mask = batch['mask']
            
            logger.info(f"  Image shape: {image.shape}")
            logger.info(f"  Mask shape:  {mask.shape}")
            logger.info(f"  Image value range: [{image.min():.4f}, {image.max():.4f}]")
            logger.info(f"  Mask value range:  [{mask.min():.4f}, {mask.max():.4f}]")
    
    except Exception as e:
        logger.error(f"Error verifying validation dataloader: {e}")
        import traceback
        traceback.print_exc()


def test_denormalization(datamodule, model):
    """Test that denormalization works correctly in the model's plot_batch."""
    logger.info("\n" + "="*70)
    logger.info("STEP 4: Testing Denormalization in plot_batch")
    logger.info("="*70)
    
    try:
        # Get a sample batch
        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))
        
        # Add prediction key to simulate model output
        batch['prediction'] = batch['mask'].clone()
        
        # Test the revert function logic from plot_batch
        input_stats = model.hparams.get("input_stats")
        target_stats = model.hparams.get("target_stats")
        
        if input_stats is None:
            input_stats = getattr(datamodule, "input_stats", None)
        if target_stats is None:
            target_stats = getattr(datamodule, "target_stats", None)
        
        logger.info(f"\nUsing input_stats: {input_stats is not None}")
        logger.info(f"Using target_stats: {target_stats is not None}")
        
        if input_stats and target_stats:
            from kornia.enhance import Denormalize
            
            def revert(tensor, stats, is_target=False):
                if stats is not None:
                    m, s = stats["mean"], stats["std"]
                    
                    if isinstance(m, list):
                        m = torch.tensor(m).clone()
                    if isinstance(s, list):
                        s = torch.tensor(s).clone()
                    
                    if isinstance(m, torch.Tensor):
                        m = m.clone()
                    if isinstance(s, torch.Tensor):
                        s = s.clone()
                    
                    m = m.to(tensor.device)
                    s = s.to(tensor.device)
                    
                    # Check channel alignment
                    num_stats_channels = len(m)
                    num_tensor_channels = tensor.shape[-3]
                    
                    logger.info(f"  Revert: tensor channels={num_tensor_channels}, stats channels={num_stats_channels}")
                    
                    if num_tensor_channels != num_tensor_channels:
                         # Slicing handled by Denormalize or manually
                         m = m[:num_tensor_channels]
                         s = s[:num_tensor_channels]
                    
                    # Fix: Force identity for categorical channel in MultiTask targets
                    if is_target and len(m) > 0:
                        m[0] = 0.0
                        s[0] = 1.0
                        logger.info("  Applying forced identity for target channel 0")

                    return Denormalize(mean=m, std=s)(tensor.float())
                return tensor
            
            x = batch['image']
            y = batch['mask'].float()
            
            logger.info(f"\nBefore denormalization:")
            logger.info(f"  Image range: [{x.min():.4f}, {x.max():.4f}]")
            logger.info(f"  Mask range:  [{y.min():.4f}, {y.max():.4f}]")
            
            # Categorical channel inspection
            y0 = y[:, 0]
            logger.info(f"  Mask Channel 0 (Categorical) unique values: {torch.unique(y0).tolist()}")
            
            x_reverted = revert(x, input_stats)
            y_reverted = revert(y, target_stats, is_target=True)
            
            logger.info(f"\nAfter denormalization:")
            logger.info(f"  Image range: [{x_reverted.min():.4f}, {x_reverted.max():.4f}]")
            logger.info(f"  Mask range:  [{y_reverted.min():.4f}, {y_reverted.max():.4f}]")
            
            y0_reverted = y_reverted[:, 0]
            logger.info(f"  Mask Channel 0 unique values after revert: {torch.unique(y0_reverted).tolist()}")
            
            # Specifically check if target stats for channel 0 are identity
            m0 = target_stats['mean'][0]
            s0 = target_stats['std'][0]
            logger.info(f"  Target Stats Channel 0: mean={m0:.4f}, std={s0:.4f}")
            if abs(m0) > 1e-5 or abs(s0 - 1.0) > 1e-5:
                logger.error("  CRITICAL: Target stats for channel 0 are NOT identity! Denormalization WILL corrupt categorical labels.")
            
            # Check if values look like original (un-normalized) values
            # Sentinel-2 reflectance should be roughly 0-0.5 (or 0-5000 if not divided by 10000)
            # Elevation should be positive
            # Climate variables have their own ranges
            
    except Exception as e:
        logger.error(f"Error testing denormalization: {e}")
        import traceback
        traceback.print_exc()


def verify_channel_alignment(datamodule):
    """Verify that stats are aligned with the correct channels, accounting for reordering."""
    logger.info("\n" + "="*70)
    logger.info("STEP 5: Detailed Channel-to-Stats Alignment Verification")
    logger.info("="*70)
    
    if not hasattr(datamodule, 'input_stats') or datamodule.input_stats is None:
        logger.warning("No input_stats available for alignment verification")
        return
    
    input_mean = datamodule.input_stats['mean']
    input_std = datamodule.input_stats['std']
    
    # Construct the full list of intermediate bands (post-append, pre-select)
    intermediate_bands = []
    for cfg in datamodule.input_configs:
        ds_bands = list(cfg.bands)
        # Check for appends in transforms
        if hasattr(cfg, "transforms") and cfg.transforms:
            def find_appends(obj):
                if isinstance(obj, dict):
                    cp = obj.get("class_path", "")
                    if "AppendNDVI" in cp: ds_bands.append("NDVI")
                    if "AppendSAVI" in cp: ds_bands.append("SAVI")
                    if "AppendEVI" in cp: ds_bands.append("EVI")
                    if "AppendNIRv" in cp: ds_bands.append("NIRv")
                    if "AppendMSAVI" in cp: ds_bands.append("MSAVI")
                    for v in obj.values(): find_appends(v)
                elif isinstance(obj, list):
                    for i in obj: find_appends(i)
            find_appends(cfg.transforms)
        intermediate_bands.extend(ds_bands)

    # Now handle SelectBands if present in global transforms
    final_bands = list(intermediate_bands)
    select_indices = None
    
    def find_select(obj):
        nonlocal select_indices
        if isinstance(obj, dict):
            if "SelectBands" in obj.get("class_path", ""):
                select_indices = obj.get("init_args", {}).get("indices")
                return
            for v in obj.values(): find_select(v)
        elif isinstance(obj, list):
            for i in obj: find_select(i)
            
    if hasattr(datamodule, "input_transforms"):
        find_select(datamodule.input_transforms)
        
    if select_indices:
        final_bands = [intermediate_bands[i] for i in select_indices]
        logger.info(f"SelectBands applied. Indices: {select_indices}")
    
    logger.info("\n--- Final Input Channel Mapping ---")
    for i, band in enumerate(final_bands):
        if i < len(input_mean):
            logger.info(f"    Channel {i:2d} ({band:12s}): mean={input_mean[i]:10.4f}, std={input_std[i]:10.4f}")
        else:
            logger.warning(f"    Channel {i:2d} ({band:12s}): NO STATS AVAILABLE")
    
    if len(final_bands) != len(input_mean):
        logger.warning(f"\nMismatch: Final stack has {len(final_bands)} channels but stats have {len(input_mean)} entries")
    else:
        logger.info(f"\nAll {len(final_bands)} final channels matched with stats")
    
    # Target channel mapping
    if hasattr(datamodule, 'target_stats') and datamodule.target_stats:
        target_mean = datamodule.target_stats['mean']
        target_std = datamodule.target_stats['std']
        
        logger.info("\n--- Target Channel Mapping ---")
        channel_idx = 0
        for i, cfg in enumerate(datamodule.target_configs):
            ds_name = cfg.dataset_class.__name__
            bands = cfg.bands
            
            logger.info(f"\nDataset {i}: {ds_name}")
            logger.info(f"  Bands ({len(bands)}): {bands}")
            
            for j, band in enumerate(bands):
                if channel_idx < len(target_mean):
                    logger.info(f"    Channel {channel_idx} ({band}): mean={target_mean[channel_idx]:.4f}, std={target_std[channel_idx]:.4f}")
                    channel_idx += 1


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Profile stats in MultiTaskUNet")
    parser.add_argument("--config", type=str, default="data/dev/config/test_new_datamodule.yaml",
                       help="Path to config YAML file")
    parser.add_argument("--root", type=str, default="data",
                       help="Root data directory")
    
    args = parser.parse_args()
    
    # Run profiling
    datamodule, model = profile_stats_from_config(args.config, args.root)
    
    # Additional verification steps
    test_denormalization(datamodule, model)
    verify_channel_alignment(datamodule)
    
    logger.info("\n" + "="*70)
    logger.info("PROFILING COMPLETE")
    logger.info("="*70)
