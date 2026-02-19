from typing import List, Literal, Union

from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torch

class HomoscedasticUncertaintyLoss(nn.Module):
    """
    Homoscedastic Uncertainty Loss Weighting for multi-task learning.

    Learns task-specific weights based on each task's homoscedastic uncertainty
    (task-dependent noise). The formulation follows:

        Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses
        for Scene Geometry and Semantics", CVPR 2018.
        Cipolla et al., "Multi-task Learning Using Uncertainty to Weigh Losses
        for Scene Geometry and Semantics", 2018.

    For a classification task (softmax likelihood):
        weighted_loss = (1 / σ²) * L + log σ

    For a regression task (Gaussian likelihood):
        weighted_loss = (1 / (2 * σ²)) * L + log σ

    To improve numerical stability, we parameterize using log(σ²):
        log_var = log(σ²)   =>   σ² = exp(log_var)
        log σ = 0.5 * log_var

    Then:
        - Classification weight = 1 / σ² = exp(-log_var)
        - Regression weight   = 1 / (2 * σ²) = 0.5 * exp(-log_var)
        - Regularization term = log σ = 0.5 * log_var

    Args:
        task_types: List of strings, each either 'classification' or 'regression',
                    indicating the type of each task.
        init_log_vars: Initial value for log(σ²) of all tasks (default: 0.0).
                       This gives initial weights ≈ 1 for classification and 0.5 for regression.
        clamp_log_var: Optional tuple (min, max) to clamp log_var for stability.
                       If None, no clamping is applied.

    Example:
        >>> loss_wrapper = HomoscedasticUncertaintyLoss(
        ...     task_types=['classification', 'regression', 'regression']
        ... )
        >>> class_loss = torch.tensor(1.2)   # cross-entropy loss
        >>> reg_loss1 = torch.tensor(0.8)    # MSE loss
        >>> reg_loss2 = torch.tensor(1.5)    # MSE loss
        >>> total_loss, logs = loss_wrapper([class_loss, reg_loss1, reg_loss2])
        >>> total_loss.backward()
    """

    def __init__(
        self,
        task_types: List[Union[str, Literal['classification', 'regression']]],
        init_log_vars: float = 0.0,
        clamp_log_var: tuple = None
    ):
        super().__init__()
        self.num_tasks = len(task_types)
        self.task_types = task_types
        self.clamp_log_var = clamp_log_var

        # Learnable log-variance parameters for each task: log(σ_i²)
        self.log_vars = nn.Parameter(torch.full((self.num_tasks,), init_log_vars))

        # Precompute multipliers for each task: 1 for classification, 0.5 for regression
        self.multipliers = torch.tensor([
            1.0 if t == 'classification' else 0.5 for t in task_types
        ])

    def forward(self, task_losses: List[torch.Tensor]) -> tuple[torch.Tensor, dict]:
        """
        Compute the weighted multi-task loss.

        Args:
            task_losses: List of scalar loss tensors, one per task.
                         Assumes they are already detached from computation graph
                         if not needed for gradients? Actually they must be part of
                         the graph for gradient flow. So pass the loss tensors directly.

        Returns:
            total_loss: Scalar tensor representing the combined weighted loss.
            log_dict: Dictionary containing individual weighted losses and learned parameters.
        """
        if len(task_losses) != self.num_tasks:
            raise ValueError(f"Expected {self.num_tasks} task losses, got {len(task_losses)}")

        # Clamp log variances if requested
        log_vars = self.log_vars
        if self.clamp_log_var is not None:
            log_vars = torch.clamp(log_vars, *self.clamp_log_var)

        # Compute precision (inverse variance) for each task
        # precision = exp(-log_var) = 1 / σ²
        precision = torch.exp(-log_vars)

        # Apply task-type specific multiplier (1 for classification, 0.5 for regression)
        weighted_precision = self.multipliers.to(precision.device) * precision

        # Regularization term: log σ = 0.5 * log_var
        log_sigma = 0.5 * log_vars

        weighted_losses = []
        log_dict = {}

        for i, loss in enumerate(task_losses):
            # Weighted loss component
            weighted_loss = weighted_precision[i] * loss + log_sigma[i]
            weighted_losses.append(weighted_loss)

            # Logging
            log_dict[f"task_{i}_raw_loss"] = loss.detach()
            log_dict[f"task_{i}_weight"] = weighted_precision[i].detach()  # effective weight on loss
            log_dict[f"task_{i}_log_var"] = log_vars[i].detach()
            log_dict[f"task_{i}_log_sigma"] = log_sigma[i].detach()

        total_loss = torch.stack(weighted_losses).sum()
        log_dict["total_loss"] = total_loss.detach()

        return total_loss, log_dict

    def get_task_weights(self) -> torch.Tensor:
        """
        Return the current effective weight multipliers for each task.
        These are the values that directly multiply the raw loss:
            classification: 1/σ²
            regression: 1/(2σ²)
        """
        with torch.no_grad():
            precision = torch.exp(-self.log_vars)
            weights = self.multipliers.to(precision.device) * precision
        return weights

class L1SSIMComboLoss(nn.Module):
    def __init__(self, w: list = [1, 1]):
        super(L1SSIMComboLoss, self).__init__()
        self.w = w
        self.ssim = StructuralSimilarityIndexMeasure()

    def forward(self, inputs: Tensor, targets: Tensor, mask: Tensor = None) -> Tensor:
        # L1 Loss
        l1_all = F.l1_loss(inputs, targets, reduction="none")
        if mask is not None:
            l1_valid = l1_all[~mask]
            l1_loss = (
                l1_valid.mean()
                if l1_valid.numel() > 0
                else torch.tensor(0.0, device=inputs.device)
            )
        else:
            l1_loss = l1_all.mean()

        # SSIM Loss
        # SSIM is sensitive to extreme values. If we have a mask, we fill masked regions
        # with target values to ensure they don't contribute to the loss.
        if mask is not None:
            inputs_masked = inputs.clone()
            inputs_masked[mask] = targets[mask].float()  
            ssim_val = self.ssim(inputs_masked, targets.float())
        else:
            ssim_val = self.ssim(inputs, targets.float())

        ssim_loss = 1 - ssim_val

        return l1_loss * self.w[0] + ssim_loss * self.w[1]


class SharpLoss(nn.Module):
    """
    Sharpness-aware regression loss combining MAE with gradient-based edge loss.

    This loss is particularly valuable for geospatial regression tasks where
    boundary sharpness and edge preservation matter. The gradient component
    encourages the model to match edge transitions in the target, while the
    MAE component ensures overall intensity accuracy.

    The gradient is computed using finite differences along spatial dimensions:
    - dy: vertical gradients (height-1 comparisons)
    - dx: horizontal gradients (width-1 comparisons)

    Args:
        alpha: Balance between MAE (1-alpha) and Gradient Loss (alpha).
               Default: 0.5 (equal weighting)

    Example:
        >>> loss_fn = SharpLoss(alpha=0.6)
        >>> pred = torch.randn(2, 1, 64, 64)
        >>> target = torch.randn(2, 1, 64, 64)
        >>> loss = loss_fn(pred, target)
        >>> loss.backward()
    """

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute sharpness-aware loss.

        Args:
            pred: Predicted tensor of shape [B, C, H, W]
            target: Target tensor of shape [B, C, H, W]
            mask: Optional boolean mask of shape [B, C, H, W] where True
                  indicates pixels to ignore (e.g., nodata regions).
                  Default: None

        Returns:
            torch.Tensor: Scalar loss value combining MAE and gradient loss
        """
        # 1. MAE Loss with optional masking
        if mask is not None:
            mae_all = F.l1_loss(pred, target, reduction="none")
            mae_valid = mae_all[~mask]
            mae_loss = (
                mae_valid.mean()
                if mae_valid.numel() > 0
                else torch.tensor(0.0, device=pred.device)
            )
        else:
            mae_loss = F.l1_loss(pred, target)

        # 2. Gradient Loss (Edge Loss)
        # Apply mask before gradient computation if provided by filling
        # masked regions with target values to neutralize gradients
        if mask is not None:
            pred_clean = pred.clone()
            # Ensure target is float for gradient computation
            target_clean = target.float().clone()
            pred_clean[mask] = target_clean[mask]
        else:
            # Ensure target is float for gradient computation
            pred_clean = pred
            target_clean = target.float()

        # Calculate horizontal and vertical differences
        # dy: vertical gradients [B, C, H-1, W]
        dy_pred = torch.abs(pred_clean[:, :, 1:, :] - pred_clean[:, :, :-1, :])
        dy_target = torch.abs(target_clean[:, :, 1:, :] - target_clean[:, :, :-1, :])

        # dx: horizontal gradients [B, C, H, W-1]
        dx_pred = torch.abs(pred_clean[:, :, :, 1:] - pred_clean[:, :, :, :-1])
        dx_target = torch.abs(target_clean[:, :, :, 1:] - target_clean[:, :, :, :-1])

        # Match the "sharpness" of the gradients
        grad_loss = F.l1_loss(dy_pred, dy_target) + F.l1_loss(dx_pred, dx_target)

        return (1 - self.alpha) * mae_loss + self.alpha * grad_loss


if __name__ == "__main__":

    # Create dummy input and target tensors
    input_tensor = torch.rand((1, 1, 256, 256), requires_grad=True)
    target_tensor = torch.rand((1, 1, 256, 256))

    # Define weights for the loss components
    weights = [0.5, 0.5]

    # Instantiate the loss function
    loss_fn = L1SSIMComboLoss()

    # Calculate the loss
    loss = loss_fn(input_tensor, target_tensor, weights)

    # Print the loss
    print(f"Calculated loss: {loss.item()}")
