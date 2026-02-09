from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torch


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
