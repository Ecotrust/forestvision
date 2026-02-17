from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torch


class MultiTaskLossWrapper(nn.Module):
    """
    Homoscedastic Uncertainty Weighting for multi-task learning.

    Learns task-specific weights based on each task's homoscedastic uncertainty
    (task-dependent noise). Tasks with higher uncertainty receive lower weights.

    The mathematical formulation follows Kendall et al., "Multi-Task Learning Using
    Uncertainty to Weigh Losses for Scene Geometry and Semantics", CVPR 2018.

    For each task i with loss L_i and homoscedastic uncertainty σ_i:
        Weighted Loss_i = (1 / (2 * σ_i²)) * L_i + log(σ_i)

    We parameterize using log(σ_i²) for numerical stability:
        log_var_i = log(σ_i²)
        precision_i = exp(-log_var_i) = 1 / σ_i²
        Weighted Loss_i = precision_i * L_i + log_var_i

    Args:
        num_tasks: Number of tasks to weight
        init_log_vars: Initial log-variance values (default: 0.0 for equal weights)

    Example:
        >>> loss_wrapper = MultiTaskLossWrapper(num_tasks=2)
        >>> seg_loss = torch.tensor(1.5)  # Segmentation loss
        >>> reg_loss = torch.tensor(0.5)  # Regression loss
        >>> total_loss, logs = loss_wrapper([seg_loss, reg_loss])
        >>> total_loss.backward()
    """

    def __init__(self, num_tasks: int, init_log_vars: float = 0.0):
        super().__init__()
        # Learnable log-variance parameters for each task
        # Shape: [num_tasks], where log_var_i = log(σ_i²)
        self.log_vars = nn.Parameter(torch.full((num_tasks,), init_log_vars))
        self.num_tasks = num_tasks

    def forward(self, task_losses: list[torch.Tensor]) -> tuple[torch.Tensor, dict]:
        """
        Compute weighted multi-task loss using homoscedastic uncertainty.

        Args:
            task_losses: List of scalar loss tensors [loss_1, loss_2, ...]

        Returns:
            tuple containing:
                - total_loss: Weighted sum of all task losses
                - log_dict: Dictionary with individual losses and learned weights

        Raises:
            ValueError: If number of task losses doesn't match num_tasks
        """
        if len(task_losses) != self.num_tasks:
            raise ValueError(
                f"Expected {self.num_tasks} task losses, got {len(task_losses)}"
            )

        weighted_losses = []
        log_dict = {}

        for i, loss in enumerate(task_losses):
            # Precision weight: 1 / σ² = exp(-log_var)
            precision = torch.exp(-self.log_vars[i])

            # Weighted loss + regularization: L_i * precision + log_var
            # The log_var term acts as regularization to prevent σ → ∞
            weighted_loss = precision * loss + self.log_vars[i]
            weighted_losses.append(weighted_loss)

            # Prepare logging info
            log_dict[f"task_{i}_raw_loss"] = loss.detach()
            log_dict[f"task_{i}_precision"] = precision.detach()
            log_dict[f"task_{i}_log_var"] = self.log_vars[i].detach()

        total_loss = torch.stack(weighted_losses).sum()
        log_dict["total_loss"] = total_loss.detach()

        return total_loss, log_dict

    def get_task_weights(self) -> torch.Tensor:
        """
        Get current precision weights for all tasks.

        Returns:
            Tensor of shape [num_tasks] with precision weights (1/σ²)
        """
        return torch.exp(-self.log_vars).detach()


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
            inputs_masked[mask] = targets[mask]
            ssim_val = self.ssim(inputs_masked, targets)
        else:
            ssim_val = self.ssim(inputs, targets)

        ssim_loss = 1 - ssim_val

        return l1_loss * self.w[0] + ssim_loss * self.w[1]


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
