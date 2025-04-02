from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torch


class L1SSIMComboLoss(nn.Module):
    def __init__(self, w: list = [1, 1]):
        self.w = w
        super(L1SSIMComboLoss, self).__init__()

    def forward(self, inputs: Tensor, targets: Tensor, mask: Tensor = None) -> Tensor:
        ssim = StructuralSimilarityIndexMeasure().to(inputs.device)
        l1_loss = nn.L1Loss(reduction="none").to(inputs.device)
        l1 = l1_loss(inputs, targets)
        if mask is not None:
            l1 = l1[~mask]
        ssim_loss = 1 - ssim(inputs, targets)  # * 0.5
        return l1.mean() * self.w[0] + ssim_loss.item() * self.w[1]


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
