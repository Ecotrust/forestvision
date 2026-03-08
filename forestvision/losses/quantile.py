"""Quantile regression losses for addressing regression toward the mean.

Quantile regression is particularly effective for canopy cover and other
regression tasks where the model tends to underpredict high values and
overpredict low values (regression toward the mean).

The key insight is that by predicting multiple quantiles (e.g., 0.1, 0.5, 0.9),
the model is forced to learn the full conditional distribution, not just
the conditional mean. This prevents the "shrinking" effect seen with MSE/MAE.
"""

from typing import List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantilePinballLoss(nn.Module):
    """Pinball (quantile) loss for quantile regression.

    The pinball loss penalizes under-prediction and over-prediction asymmetrically
    based on the quantile being estimated. For quantile q:
    - If error > 0 (under-prediction): loss = q * error
    - If error < 0 (over-prediction): loss = (q - 1) * error

    This forces the model to learn specific percentiles of the target distribution,
    rather than just the mean. When used with multiple quantiles, the model learns
    the full conditional distribution, preventing regression toward the mean.

    Args:
        quantiles: List of quantiles to predict, e.g., [0.1, 0.5, 0.9].
                   Default: [0.5] (median regression, equivalent to MAE).
        reduction: How to reduce the loss. Options: "mean", "sum", "none".
                   Default: "mean".

    Example:
        >>> # Single quantile (median regression)
        >>> loss_fn = QuantilePinballLoss(quantiles=[0.5])
        >>> pred = torch.randn(2, 1, 64, 64)  # [B, num_quantiles, H, W]
        >>> target = torch.randn(2, 1, 64, 64)
        >>> loss = loss_fn(pred, target)

        >>> # Multiple quantiles (learns full distribution)
        >>> loss_fn = QuantilePinballLoss(quantiles=[0.1, 0.5, 0.9])
        >>> pred = torch.randn(2, 3, 64, 64)  # 3 quantile outputs
        >>> target = torch.randn(2, 1, 64, 64)  # Single target
        >>> loss = loss_fn(pred, target)
    """

    def __init__(
        self,
        quantiles: List[float] = [0.5],
        reduction: str = "mean",
    ):
        super().__init__()
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        self.reduction = reduction

        # Register quantiles as buffer so they move with the model
        self.register_buffer("quantile_tensor", torch.tensor(quantiles))

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute pinball loss for quantile regression.

        Args:
            pred: Predicted tensor of shape [B, num_quantiles, H, W] or [B, num_quantiles].
                  Each channel corresponds to one quantile prediction.
            target: Target tensor of shape [B, 1, H, W] or [B, 1] or [B, H, W].
                    Will be broadcast to match pred's quantile dimension.
            mask: Optional boolean mask of shape [B, 1, H, W] where True
                  indicates pixels to ignore (e.g., nodata regions).
                  Default: None.

        Returns:
            torch.Tensor: Scalar loss value (if reduction != "none").
        """
        # Ensure target has the right shape for broadcasting
        if target.dim() == pred.dim() - 1:
            # target is [B, H, W], pred is [B, Q, H, W]
            target = target.unsqueeze(1)
        elif target.shape[1] != self.num_quantiles:
            # target is [B, 1, H, W], pred is [B, Q, H, W]
            # Broadcast target to match num_quantiles
            target = target.expand(-1, self.num_quantiles, -1, -1)

        # Compute errors: pred - target
        # Shape: [B, num_quantiles, H, W]
        errors = pred - target

        # Move quantiles to same device as predictions
        quantiles = self.quantile_tensor.to(pred.device).view(1, self.num_quantiles, 1, 1)

        # Pinball loss formula:
        # For error >= 0: loss = quantile * error
        # For error < 0: loss = (quantile - 1) * error
        # This can be written as: max(quantile * error, (quantile - 1) * error)
        # Or equivalently: quantile * errors * (errors >= 0) + (quantile - 1) * errors * (errors < 0)

        # More stable computation using torch.where
        loss = torch.where(
            errors >= 0,
            quantiles * errors,           # Under-prediction: q * error
            (quantiles - 1) * errors      # Over-prediction: (q-1) * error (negative)
        )

        # Take absolute value since (quantile - 1) is negative
        loss = torch.abs(loss)

        # Apply mask if provided
        if mask is not None:
            # Expand mask to match quantile dimension if needed
            if mask.dim() == pred.dim() - 1:
                mask = mask.unsqueeze(1)
            if mask.shape[1] == 1 and self.num_quantiles > 1:
                mask = mask.expand(-1, self.num_quantiles, -1, -1)

            loss = loss[~mask]
            if loss.numel() == 0:
                return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # Reduce loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # "none"
            return loss

    def get_quantile_predictions(self, pred: torch.Tensor) -> dict:
        """Convert model output to quantile predictions dictionary.

        Args:
            pred: Predicted tensor of shape [B, num_quantiles, H, W].

        Returns:
            dict: Mapping from quantile value to prediction tensor.
        """
        return {
            q: pred[:, i] for i, q in enumerate(self.quantiles)
        }


class BoundedQuantileLoss(nn.Module):
    """Quantile loss with physical bounds for regression tasks.

    Combines quantile regression with boundary penalties to ensure predictions
    stay within physically meaningful ranges (e.g., 0-100% canopy cover).

    This is particularly useful for canopy cover where values must be non-negative
    and typically have an upper bound (e.g., 10000 for 100% cover in certain units).

    Args:
        quantiles: List of quantiles to predict. Default: [0.5].
        min_val: Minimum physically valid value. Default: 0.0.
        max_val: Maximum physically valid value. Default: None (no upper bound).
        bound_penalty: Weight for boundary violation penalty. Default: 0.1.
        reduction: How to reduce the loss. Default: "mean".

    Example:
        >>> # Canopy cover regression with 0-10000 bounds
        >>> loss_fn = BoundedQuantileLoss(
        ...     quantiles=[0.1, 0.5, 0.9],
        ...     min_val=0.0,
        ...     max_val=10000.0,
        ...     bound_penalty=0.1
        ... )
    """

    def __init__(
        self,
        quantiles: List[float] = [0.5],
        min_val: float = 0.0,
        max_val: Optional[float] = None,
        bound_penalty: float = 0.1,
        reduction: str = "mean",
    ):
        super().__init__()
        self.quantile_loss = QuantilePinballLoss(quantiles, reduction="none")
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        self.min_val = min_val
        self.max_val = max_val
        self.bound_penalty = bound_penalty
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute bounded quantile loss.

        Args:
            pred: Predicted tensor of shape [B, num_quantiles, H, W].
            target: Target tensor of shape [B, 1, H, W].
            mask: Optional boolean mask where True indicates pixels to ignore.

        Returns:
            torch.Tensor: Combined quantile + boundary penalty loss.
        """
        # Compute base quantile loss
        loss = self.quantile_loss(pred, target, mask)

        # Compute boundary penalties
        bound_loss = torch.tensor(0.0, device=pred.device)

        if self.min_val is not None:
            # Penalize predictions below minimum
            min_violation = F.relu(self.min_val - pred)
            bound_loss = bound_loss + min_violation.sum()

        if self.max_val is not None:
            # Penalize predictions above maximum
            max_violation = F.relu(pred - self.max_val)
            bound_loss = bound_loss + max_violation.sum()

        # Combine losses
        total_loss = loss + self.bound_penalty * bound_loss

        if self.reduction == "mean":
            return total_loss.mean()
        elif self.reduction == "sum":
            return total_loss.sum()
        else:
            return total_loss


class AdaptiveQuantileLoss(nn.Module):
    """Adaptive quantile loss that adjusts quantiles based on prediction uncertainty.

    This advanced loss starts with wide-spaced quantiles (e.g., 0.1, 0.9) and
    progressively narrows them as training converges, forcing the model to
    learn more precise distributions over time.

    This can help address regression toward the mean by initially allowing
    the model to explore the full prediction space, then refining the
    quantile estimates.

    Args:
        initial_quantiles: Starting quantiles. Default: [0.1, 0.5, 0.9].
        final_quantiles: Target quantiles (narrower). Default: [0.25, 0.5, 0.75].
        warmup_epochs: Number of epochs to stay at initial_quantiles. Default: 10.
        anneal_epochs: Number of epochs to transition between quantile sets. Default: 20.
        reduction: How to reduce the loss. Default: "mean".

    Example:
        >>> loss_fn = AdaptiveQuantileLoss()
        >>> for epoch in range(num_epochs):
        ...     loss_fn.set_epoch(epoch)
        ...     loss = loss_fn(pred, target)
    """

    def __init__(
        self,
        initial_quantiles: List[float] = [0.1, 0.5, 0.9],
        final_quantiles: List[float] = [0.25, 0.5, 0.75],
        warmup_epochs: int = 10,
        anneal_epochs: int = 20,
        reduction: str = "mean",
    ):
        super().__init__()
        self.initial_quantiles = torch.tensor(initial_quantiles)
        self.final_quantiles = torch.tensor(final_quantiles)
        self.warmup_epochs = warmup_epochs
        self.anneal_epochs = anneal_epochs
        self.reduction = reduction
        self.current_epoch = 0

        # Initialize with starting quantiles
        self.quantile_loss = QuantilePinballLoss(initial_quantiles, reduction)
        self.num_quantiles = len(initial_quantiles)

    def set_epoch(self, epoch: int):
        """Update the current epoch to adjust quantile spacing."""
        self.current_epoch = epoch

        if epoch < self.warmup_epochs:
            # Warmup phase: use wide quantiles
            current_q = self.initial_quantiles
        elif epoch >= self.warmup_epochs + self.anneal_epochs:
            # Final phase: use narrow quantiles
            current_q = self.final_quantiles
        else:
            # Annealing phase: linear interpolation
            progress = (epoch - self.warmup_epochs) / self.anneal_epochs
            current_q = self.initial_quantiles + progress * (
                self.final_quantiles - self.initial_quantiles
            )

        # Update the underlying loss with new quantiles
        self.quantile_loss.quantiles = current_q.tolist()
        self.quantile_loss.quantile_tensor = current_q
        self.quantile_loss.num_quantiles = len(current_q)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute adaptive quantile loss."""
        return self.quantile_loss(pred, target, mask)


class MeanScaleQuantileLoss(nn.Module):
    """Combines mean prediction with scale (uncertainty) via quantile parameterization.

    Instead of predicting quantiles directly, this loss parameterizes predictions
    as mean + scale, where quantiles are computed as:
        q = mean + scale * Φ⁻¹(quantile)

    Where Φ⁻¹ is the inverse CDF of a standard normal. This enforces ordering
    of quantiles (q_0.1 < q_0.5 < q_0.9) and can be more stable.

    Args:
        quantiles: List of quantiles to predict. Default: [0.1, 0.5, 0.9].
        reduction: How to reduce the loss. Default: "mean".

    Example:
        >>> loss_fn = MeanScaleQuantileLoss(quantiles=[0.1, 0.5, 0.9])
        >>> # Model outputs [B, 2, H, W]: channel 0 = mean, channel 1 = scale
        >>> pred = torch.randn(2, 2, 64, 64)
        >>> target = torch.randn(2, 1, 64, 64)
        >>> loss = loss_fn(pred, target)
    """

    def __init__(
        self,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        reduction: str = "mean",
    ):
        super().__init__()
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        self.reduction = reduction

        # Precompute normal quantile values (z-scores)
        from scipy.stats import norm
        self.register_buffer(
            "z_scores",
            torch.tensor([norm.ppf(q) for q in quantiles])
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute mean-scale quantile loss.

        Args:
            pred: Predicted tensor of shape [B, 2, H, W] where:
                  - channel 0: mean prediction
                  - channel 1: scale (must be positive, enforced via softplus)
            target: Target tensor of shape [B, 1, H, W].
            mask: Optional boolean mask.

        Returns:
            torch.Tensor: Quantile loss computed from mean-scale parameterization.
        """
        # Extract mean and scale
        mean = pred[:, 0:1]  # [B, 1, H, W]
        scale = F.softplus(pred[:, 1:2])  # [B, 1, H, W], ensure positive

        # Compute quantile predictions: mean + scale * z_score
        # Shape: [B, num_quantiles, H, W]
        z = self.z_scores.to(pred.device).view(1, self.num_quantiles, 1, 1)
        quantile_pred = mean + scale * z

        # Compute pinball loss on quantile predictions
        # Expand target to match quantile dimension
        target_expanded = target.expand(-1, self.num_quantiles, -1, -1)
        errors = quantile_pred - target_expanded

        # Move quantiles to device and reshape
        quantiles = torch.tensor(self.quantiles, device=pred.device).view(
            1, self.num_quantiles, 1, 1
        )

        # Pinball loss
        loss = torch.where(
            errors >= 0,
            quantiles * errors,
            (quantiles - 1) * errors
        ).abs()

        # Apply mask
        if mask is not None:
            if mask.dim() == loss.dim() - 1:
                mask = mask.unsqueeze(1)
            if mask.shape[1] == 1 and self.num_quantiles > 1:
                mask = mask.expand(-1, self.num_quantiles, -1, -1)
            loss = loss[~mask]
            if loss.numel() == 0:
                return torch.tensor(0.0, device=pred.device, requires_grad=True)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def get_quantile_schedule(strategy: str = "standard") -> List[float]:
    """Get recommended quantile sets for different strategies.

    Args:
        strategy: One of "standard", "wide", "narrow", "asymmetric".
            - "standard": [0.1, 0.5, 0.9] - good balance
            - "wide": [0.05, 0.5, 0.95] - better for tail estimation
            - "narrow": [0.25, 0.5, 0.75] - better for central tendency
            - "asymmetric": [0.1, 0.5, 0.9] with more weight on low quantile

    Returns:
        List of quantile values.
    """
    schedules = {
        "standard": [0.1, 0.5, 0.9],
        "wide": [0.05, 0.5, 0.95],
        "narrow": [0.25, 0.5, 0.75],
        "asymmetric": [0.1, 0.5, 0.9],  # Same as standard but use with weighted loss
        "five_quantile": [0.1, 0.25, 0.5, 0.75, 0.9],
        "seven_quantile": [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
    }
    return schedules.get(strategy, schedules["standard"])


if __name__ == "__main__":
    # Test the quantile loss
    print("Testing QuantilePinballLoss...")

    # Single quantile (median regression)
    loss_fn_median = QuantilePinballLoss(quantiles=[0.5])
    pred_median = torch.randn(2, 1, 64, 64)
    target = torch.randn(2, 1, 64, 64)
    loss_median = loss_fn_median(pred_median, target)
    print(f"Median (0.5) loss: {loss_median.item():.4f}")

    # Multiple quantiles
    loss_fn_multi = QuantilePinballLoss(quantiles=[0.1, 0.5, 0.9])
    pred_multi = torch.randn(2, 3, 64, 64)
    loss_multi = loss_fn_multi(pred_multi, target)
    print(f"Multi-quantile loss: {loss_multi.item():.4f}")

    # Test with mask
    mask = torch.rand(2, 1, 64, 64) > 0.5
    loss_masked = loss_fn_multi(pred_multi, target, mask)
    print(f"Masked loss: {loss_masked.item():.4f}")

    # Test bounded loss
    print("\nTesting BoundedQuantileLoss...")
    loss_fn_bounded = BoundedQuantileLoss(
        quantiles=[0.1, 0.5, 0.9],
        min_val=0.0,
        max_val=10000.0,
    )
    # Create predictions that violate bounds
    pred_bounded = torch.randn(2, 3, 64, 64) * 20000  # Will have values outside [0, 10000]
    loss_bounded = loss_fn_bounded(pred_bounded, target)
    print(f"Bounded loss: {loss_bounded.item():.4f}")

    print("\nAll tests passed!")
