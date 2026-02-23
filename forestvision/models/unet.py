"""
This U-Net implementation is based on https://github.com/milesial/Pytorch-UNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor


# Backbone configurations: (stem, layer1, layer2, layer3, layer4) channel counts
BACKBONE_CONFIGS = {
    "resnet18": {
        "channels": [64, 64, 128, 256, 512],
        "bottleneck": False,
        "weights": models.ResNet18_Weights.IMAGENET1K_V1,
    },
    "resnet34": {
        "channels": [64, 64, 128, 256, 512],
        "bottleneck": False,
        "weights": models.ResNet34_Weights.IMAGENET1K_V1,
    },
    "resnet50": {
        "channels": [64, 256, 512, 1024, 2048],
        "bottleneck": True,
        "weights": models.ResNet50_Weights.IMAGENET1K_V1,
    },
    "resnet101": {
        "channels": [64, 256, 512, 1024, 2048],
        "bottleneck": True,
        "weights": models.ResNet101_Weights.IMAGENET1K_V1,
    },
}


class ConvBlock(nn.Module):
    """Double convolution block: (conv => BN => ReLU) * 2

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        dropout: Dropout probability (0 to 1).

    Input Shape:
        [B, in_channels, H, W]

    Output Shape:
        [B, out_channels, H, W]
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float):
        super().__init__()
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            Output tensor of shape [B, out_channels, H, W].
        """
        return self.conv(x)


class EncoderBlock(nn.Module):
    """Encoder block with max pooling followed by double convolution.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        dropout: Dropout probability.

    Input Shape:
        [B, in_channels, H, W]

    Output Shape:
        [B, out_channels, H//2, W//2]
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2), ConvBlock(in_channels, out_channels, dropout=dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            Output tensor of shape [B, out_channels, H//2, W//2].
        """
        return self.mpconv(x)


class DecoderBlock(nn.Module):
    """Decoder block with upsampling and skip connection.

    Args:
        in_channels: Number of channels from deeper layer (before upsampling).
        skip_channels: Number of channels from skip connection.
        out_channels: Number of output channels.
        bilinear: If True, use bilinear upsampling; otherwise use transposed conv.

    Input Shapes:
        x1: [B, in_channels, H, W] - from deeper layer
        x2: [B, skip_channels, H*2, W*2] - skip connection

    Output Shape:
        [B, out_channels, H*2, W*2]
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        bilinear: bool = True,
    ):
        super().__init__()

        if bilinear:
            # Bilinear upsampling: preserves channels
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            conv_in_channels = in_channels + skip_channels
        else:
            # ConvTranspose2d: reduces channels from in_channels to out_channels
            self.up = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2
            )
            conv_in_channels = out_channels + skip_channels

        self.conv = ConvBlock(conv_in_channels, out_channels, dropout=0)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connection.

        Args:
            x1: Tensor from deeper layer.
            x2: Skip connection tensor.

        Returns:
            Output tensor [B, out_channels, H*2, W*2].
        """
        x1 = self.up(x1)

        # Ensure x1 and x2 have the same spatial size
        diff_h = x2.size(2) - x1.size(2)
        diff_w = x2.size(3) - x1.size(3)

        # Pad x1 to match x2 dimensions if needed
        x1 = F.pad(
            x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2]
        )

        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class OutConv(nn.Module):
    """Output convolution layer (1x1 conv for final prediction).

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels (number of classes).

    Input Shape:
        [B, in_channels, H, W]

    Output Shape:
        [B, out_channels, H, W]
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            Output tensor of shape [B, out_channels, H, W].
        """
        return self.conv(x)


class UNet(nn.Module):
    """U-Net architecture for semantic segmentation.

    Args:
        in_channels: Number of input channels (e.g., 3 for RGB).
        out_channels: Number of output channels (number of classes).
        bilinear: If True, use bilinear upsampling; otherwise use transposed conv.
        dropout: Dropout probability for encoder blocks (0 to 1).

    Input Shape:
        [B, in_channels, H, W] where H, W should be divisible by 16

    Output Shape:
        [B, out_channels, H, W]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()

        # Input validation
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")
        if not 0 <= dropout <= 1:
            raise ValueError(f"dropout must be in [0, 1], got {dropout}")

        factor = 2 if bilinear else 1

        # Encoder (contracting path)
        self.inc = ConvBlock(in_channels, 64, dropout=0)
        self.down1 = EncoderBlock(64, 128, dropout=dropout)
        self.down2 = EncoderBlock(128, 256, dropout=dropout)
        self.down3 = EncoderBlock(256, 512, dropout=dropout)
        self.down4 = EncoderBlock(512, 1024 // factor, dropout=dropout)

        # Decoder (expansive path)
        # DecoderBlock(in_channels, skip_channels, out_channels, bilinear)
        # With bilinear: in_channels = previous output, skip_channels = skip connection
        # up1: 1024 (x5) upsampled + 512 (x4) = 1536 (bilinear) or handled differently
        # Note: Original implementation assumed different structure
        # For UNet: x5 (1024//factor) -> up -> concat x4 (512) -> conv
        # With bilinear: upsample preserves 1024//factor, concat with 512 gives 1024//factor + 512
        # But DecoderBlock now expects explicit skip_channels
        
        if bilinear:
            # Bilinear: channels preserved during upsampling
            self.up1 = DecoderBlock(1024 // factor, 512, 512 // factor, bilinear)
            self.up2 = DecoderBlock(512 // factor, 256, 256 // factor, bilinear)
            self.up3 = DecoderBlock(256 // factor, 128, 128 // factor, bilinear)
            self.up4 = DecoderBlock(128 // factor, 64, 64, bilinear)
        else:
            # Transposed conv: reduces channels before concatenation
            self.up1 = DecoderBlock(1024 // factor, 512, 512 // factor, bilinear)
            self.up2 = DecoderBlock(512 // factor, 256, 256 // factor, bilinear)
            self.up3 = DecoderBlock(256 // factor, 128, 128 // factor, bilinear)
            self.up4 = DecoderBlock(128 // factor, 64, 64, bilinear)

        self.outc = OutConv(64, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through U-Net.

        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Output logits tensor of shape [B, out_channels, H, W].
        """
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)


class MTUNet(nn.Module):
    """Multi-Task U-Net for simultaneous segmentation and regression.

    Args:
        in_channels: Number of input channels.
        seg_channels: Number of segmentation classes.
        reg_channels: Number of regression targets.
        bilinear: If True, use bilinear upsampling; otherwise use transposed conv.
        dropout: Dropout probability for encoder blocks.
        use_tanh: If True, apply tanh activation to regression outputs.

    Input Shape:
        [B, in_channels, H, W]

    Output Shapes:
        seg_logits: [B, seg_channels, H, W]
        reg_out: [B, reg_channels, H, W] (with tanh if use_tanh=True)
    """

    def __init__(
        self,
        in_channels: int,
        seg_channels: int,
        reg_channels: int,
        bilinear: bool = True,
        dropout: float = 0.5,
        use_tanh: bool = True,
    ):
        super().__init__()

        # Input validation
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if seg_channels <= 0:
            raise ValueError(f"seg_channels must be positive, got {seg_channels}")
        if reg_channels < 0:
            raise ValueError(f"reg_channels must be non-negative, got {reg_channels}")
        if not 0 <= dropout <= 1:
            raise ValueError(f"dropout must be in [0, 1], got {dropout}")

        factor = 2 if bilinear else 1

        # Encoder (contracting path)
        self.inc = ConvBlock(in_channels, 64, dropout=0)
        self.down1 = EncoderBlock(64, 128, dropout=dropout)
        self.down2 = EncoderBlock(128, 256, dropout=dropout)
        self.down3 = EncoderBlock(256, 512, dropout=dropout)
        self.down4 = EncoderBlock(512, 1024 // factor, dropout=dropout)

        # Decoder (expansive path)
        # DecoderBlock(in_channels, skip_channels, out_channels, bilinear)
        self.up1 = DecoderBlock(1024 // factor, 512, 512 // factor, bilinear)
        self.up2 = DecoderBlock(512 // factor, 256, 256 // factor, bilinear)
        self.up3 = DecoderBlock(256 // factor, 128, 128 // factor, bilinear)
        self.up4 = DecoderBlock(128 // factor, 64, 64, bilinear)

        # Task-specific output heads
        self.seg_head = OutConv(64, seg_channels)
        self.reg_head = OutConv(64, reg_channels) if reg_channels > 0 else None

        self.use_tanh = use_tanh
        self.tanh = nn.Tanh() if use_tanh else nn.Identity()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through multi-task U-Net.

        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Tuple of (seg_logits, reg_out):
                - seg_logits: [B, seg_channels, H, W] - segmentation logits
                - reg_out: [B, reg_channels, H, W] - regression output
        """
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        seg_logits = self.seg_head(x)
        reg_logits = self.reg_head(x)
        reg_out = self.tanh(reg_logits)

        return seg_logits, reg_out


class OptimizedDecoderBlock(nn.Module):
    """Optimized decoder block with bottleneck in skip connection.

    This block addresses two problems in standard U-Net:
    1. Dimensionality mismatch - features from different levels have different channel counts
    2. Semantic gap - encoder features (spatial details) and decoder features (semantic context)
       may not align well

    The solution is a 1x1 convolution bottleneck before concatenation that:
    - Reduces channel depth of skip connections
    - Forces compression to retain most relevant information
    - Creates more balanced feature sets for subsequent convolutions
    - Reduces parameters and computational cost

    Args:
        in_channels: Number of channels from deeper layer (before upsampling).
        skip_channels: Number of channels from skip connection.
        out_channels: Number of output channels.
        bilinear: If True, use bilinear upsampling; otherwise use transposed conv.
        bottleneck_ratio: Ratio to reduce skip channels (e.g., 0.5 = halve channels).

    Input Shapes:
        x1: [B, in_channels, H, W] - from deeper layer
        x2: [B, skip_channels, H*2, W*2] - skip connection

    Output Shape:
        [B, out_channels, H*2, W*2]
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        bilinear: bool = True,
        bottleneck_ratio: float = 0.5,
    ):
        super().__init__()

        if not 0 < bottleneck_ratio <= 1.0:
            raise ValueError(
                f"bottleneck_ratio must be in (0, 1], got {bottleneck_ratio}"
            )

        # Calculate bottleneck channels (minimum 1)
        bottleneck_channels = max(1, int(skip_channels * bottleneck_ratio))

        # Bottleneck: 1x1 convolution to reduce dimensionality
        self.bottleneck = nn.Sequential(
            nn.Conv2d(skip_channels, bottleneck_channels, kernel_size=1),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
        )

        if bilinear:
            # Bilinear upsampling: preserves channels
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            conv_in_channels = in_channels + bottleneck_channels
        else:
            # ConvTranspose2d: reduces channels from in_channels to out_channels
            self.up = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2
            )
            conv_in_channels = out_channels + bottleneck_channels

        self.conv = ConvBlock(conv_in_channels, out_channels, dropout=0)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward pass with bottleneck skip connection.

        Args:
            x1: Tensor from deeper layer.
            x2: Skip connection tensor.

        Returns:
            Output tensor [B, out_channels, H*2, W*2].
        """
        # Apply bottleneck to skip connection
        x2 = self.bottleneck(x2)

        # Upsample deeper features
        x1 = self.up(x1)

        # Ensure x1 and x2 have the same spatial size
        diff_h = x2.size(2) - x1.size(2)
        diff_w = x2.size(3) - x1.size(3)

        # Pad x1 to match x2 dimensions if needed
        x1 = F.pad(
            x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2]
        )

        # Concatenate bottlenecked skip connection
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class OptimizedUNet(nn.Module):
    """Optimized U-Net with bottleneck skip connections for semantic segmentation.

    This architecture introduces 1x1 convolution bottlenecks in skip connections
    to reduce dimensionality and bridge the semantic gap between encoder and
    decoder features. This typically improves accuracy while reducing parameters.

    Reference:
        Clark et al. (2023) - Optimized U-Net for land-use/land-cover classification

    Args:
        in_channels: Number of input channels (e.g., 3 for RGB).
        out_channels: Number of output channels (number of classes).
        bilinear: If True, use bilinear upsampling; otherwise use transposed conv.
        dropout: Dropout probability for encoder blocks (0 to 1).
        bottleneck_ratio: Ratio to reduce skip channels (default: 0.5 = halve channels).

    Input Shape:
        [B, in_channels, H, W] where H, W should be divisible by 16

    Output Shape:
        [B, out_channels, H, W]

    Example:
        >>> model = OptimizedUNet(
        ...     in_channels=3,
        ...     out_channels=14,
        ...     bottleneck_ratio=0.5
        ... )
        >>> x = torch.randn(2, 3, 256, 256)
        >>> out = model(x)
        >>> out.shape
        torch.Size([2, 14, 256, 256])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = True,
        dropout: float = 0.5,
        bottleneck_ratio: float = 0.5,
    ):
        super().__init__()

        # Input validation
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")
        if not 0 <= dropout <= 1:
            raise ValueError(f"dropout must be in [0, 1], got {dropout}")
        if not 0 < bottleneck_ratio <= 1.0:
            raise ValueError(
                f"bottleneck_ratio must be in (0, 1], got {bottleneck_ratio}"
            )

        factor = 2 if bilinear else 1

        # Encoder (contracting path) - same as standard U-Net
        self.inc = ConvBlock(in_channels, 64, dropout=0)
        self.down1 = EncoderBlock(64, 128, dropout=dropout)
        self.down2 = EncoderBlock(128, 256, dropout=dropout)
        self.down3 = EncoderBlock(256, 512, dropout=dropout)
        self.down4 = EncoderBlock(512, 1024 // factor, dropout=dropout)

        # Decoder (expansive path) with optimized skip connections
        # OptimizedDecoderBlock(in_channels, skip_channels, out_channels, bilinear, bottleneck_ratio)
        if bilinear:
            self.up1 = OptimizedDecoderBlock(
                1024 // factor, 512, 512 // factor, bilinear, bottleneck_ratio
            )
            self.up2 = OptimizedDecoderBlock(
                512 // factor, 256, 256 // factor, bilinear, bottleneck_ratio
            )
            self.up3 = OptimizedDecoderBlock(
                256 // factor, 128, 128 // factor, bilinear, bottleneck_ratio
            )
            self.up4 = OptimizedDecoderBlock(
                128 // factor, 64, 64, bilinear, bottleneck_ratio
            )
        else:
            self.up1 = OptimizedDecoderBlock(
                1024 // factor, 512, 512 // factor, bilinear, bottleneck_ratio
            )
            self.up2 = OptimizedDecoderBlock(
                512 // factor, 256, 256 // factor, bilinear, bottleneck_ratio
            )
            self.up3 = OptimizedDecoderBlock(
                256 // factor, 128, 128 // factor, bilinear, bottleneck_ratio
            )
            self.up4 = OptimizedDecoderBlock(
                128 // factor, 64, 64, bilinear, bottleneck_ratio
            )

        self.outc = OutConv(64, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Optimized U-Net.

        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Output logits tensor of shape [B, out_channels, H, W].
        """
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with optimized skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)


class OptimizedMTUNet(nn.Module):
    """Optimized Multi-Task U-Net with bottleneck skip connections.

    Combines the multi-task capability of MTUNet with optimized skip connections
    that reduce dimensionality and bridge semantic gaps between encoder and decoder.

    Args:
        in_channels: Number of input channels.
        seg_channels: Number of segmentation classes.
        reg_channels: Number of regression targets.
        bilinear: If True, use bilinear upsampling; otherwise use transposed conv.
        dropout: Dropout probability for encoder blocks.
        use_tanh: If True, apply tanh activation to regression outputs.
        bottleneck_ratio: Ratio to reduce skip channels (default: 0.5 = halve channels).

    Input Shape:
        [B, in_channels, H, W]

    Output Shapes:
        seg_logits: [B, seg_channels, H, W]
        reg_out: [B, reg_channels, H, W] (with tanh if use_tanh=True)

    Example:
        >>> model = OptimizedMTUNet(
        ...     in_channels=3,
        ...     seg_channels=14,
        ...     reg_channels=2,
        ...     bottleneck_ratio=0.5
        ... )
        >>> x = torch.randn(2, 3, 256, 256)
        >>> seg, reg = model(x)
        >>> seg.shape
        torch.Size([2, 14, 256, 256])
        >>> reg.shape
        torch.Size([2, 2, 256, 256])
    """

    def __init__(
        self,
        in_channels: int,
        seg_channels: int,
        reg_channels: int,
        bilinear: bool = True,
        dropout: float = 0.5,
        use_tanh: bool = True,
        bottleneck_ratio: float = 0.5,
    ):
        super().__init__()

        # Input validation
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if seg_channels <= 0:
            raise ValueError(f"seg_channels must be positive, got {seg_channels}")
        if reg_channels < 0:
            raise ValueError(f"reg_channels must be non-negative, got {reg_channels}")
        if not 0 <= dropout <= 1:
            raise ValueError(f"dropout must be in [0, 1], got {dropout}")
        if not 0 < bottleneck_ratio <= 1.0:
            raise ValueError(
                f"bottleneck_ratio must be in (0, 1], got {bottleneck_ratio}"
            )

        factor = 2 if bilinear else 1

        # Encoder (contracting path) - same as standard MTUNet
        self.inc = ConvBlock(in_channels, 64, dropout=0)
        self.down1 = EncoderBlock(64, 128, dropout=dropout)
        self.down2 = EncoderBlock(128, 256, dropout=dropout)
        self.down3 = EncoderBlock(256, 512, dropout=dropout)
        self.down4 = EncoderBlock(512, 1024 // factor, dropout=dropout)

        # Decoder (expansive path) with optimized skip connections
        # OptimizedDecoderBlock(in_channels, skip_channels, out_channels, bilinear, bottleneck_ratio)
        self.up1 = OptimizedDecoderBlock(
            1024 // factor, 512, 512 // factor, bilinear, bottleneck_ratio
        )
        self.up2 = OptimizedDecoderBlock(
            512 // factor, 256, 256 // factor, bilinear, bottleneck_ratio
        )
        self.up3 = OptimizedDecoderBlock(
            256 // factor, 128, 128 // factor, bilinear, bottleneck_ratio
        )
        self.up4 = OptimizedDecoderBlock(
            128 // factor, 64, 64, bilinear, bottleneck_ratio
        )

        # Task-specific output heads
        self.seg_head = OutConv(64, seg_channels)
        self.reg_head = OutConv(64, reg_channels) if reg_channels > 0 else None

        self.use_tanh = use_tanh
        self.tanh = nn.Tanh() if use_tanh else nn.Identity()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through optimized multi-task U-Net.

        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Tuple of (seg_logits, reg_out):
                - seg_logits: [B, seg_channels, H, W] - segmentation logits
                - reg_out: [B, reg_channels, H, W] - regression output
        """
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with optimized skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        seg_logits = self.seg_head(x)
        
        if self.reg_head is not None:
            reg_logits = self.reg_head(x)
            reg_out = self.tanh(reg_logits)
        else:
            reg_out = torch.empty(x.size(0), 0, x.size(2), x.size(3), device=x.device)

        return seg_logits, reg_out


class ResNetUNet(nn.Module):
    """Single-task U-Net with ResNet backbone for semantic segmentation.

    This class replaces the standard U-Net encoder with a pretrained ResNet backbone
    while retaining the U-Net decoder structure.

    Args:
        in_channels: Number of input channels (e.g., 3 for RGB, 10 for multispectral).
        out_channels: Number of output channels (number of classes).
        backbone: ResNet variant to use ('resnet18', 'resnet34', 'resnet50', 'resnet101').
        pretrained: If True, use ImageNet pretrained weights.
        freeze_backbone: If True, freeze the backbone parameters.
        bilinear: If True, use bilinear upsampling; otherwise use transposed conv.
        dropout: Dropout probability for decoder blocks.

    Input Shape:
        [B, in_channels, H, W] where H, W should be divisible by 32

    Output Shape:
        [B, out_channels, H, W]

    Example:
        >>> model = ResNetUNet(
        ...     in_channels=10,
        ...     out_channels=14,
        ...     backbone='resnet50',
        ...     pretrained=True
        ... )
        >>> x = torch.randn(2, 10, 256, 256)
        >>> out = model(x)
        >>> out.shape
        torch.Size([2, 14, 256, 256])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        backbone: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        bilinear: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()

        # Input validation
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")
        if not 0 <= dropout <= 1:
            raise ValueError(f"dropout must be in [0, 1], got {dropout}")
        if backbone not in BACKBONE_CONFIGS:
            raise ValueError(
                f"backbone must be one of {list(BACKBONE_CONFIGS.keys())}, got {backbone}"
            )

        self.backbone_name = backbone
        self.bilinear = bilinear
        config = BACKBONE_CONFIGS[backbone]
        stem_ch, layer1_ch, layer2_ch, layer3_ch, layer4_ch = config["channels"]
        is_bottleneck = config["bottleneck"]

        # Create ResNet backbone
        weights = config["weights"] if pretrained else None
        if backbone == "resnet18":
            base_model = models.resnet18(weights=weights)
        elif backbone == "resnet34":
            base_model = models.resnet34(weights=weights)
        elif backbone == "resnet50":
            base_model = models.resnet50(weights=weights)
        elif backbone == "resnet101":
            base_model = models.resnet101(weights=weights)

        # Modify conv1 for multispectral input while preserving pretrained weights
        if in_channels != 3:
            self._init_conv1_for_multispectral(base_model, in_channels)

        # Extract features at multiple scales using feature_extraction API
        return_nodes = {
            "relu": "stem",
            "layer1": "layer1",
            "layer2": "layer2",
            "layer3": "layer3",
            "layer4": "layer4",
        }
        self.backbone = create_feature_extractor(base_model, return_nodes=return_nodes)

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Channel adapters for bottleneck architectures (ResNet50+)
        self.adapters = nn.ModuleDict()
        if is_bottleneck:
            self.adapters["layer4"] = nn.Conv2d(layer4_ch, 512, kernel_size=1)
            self.adapters["layer3"] = nn.Conv2d(layer3_ch, 256, kernel_size=1)
            self.adapters["layer2"] = nn.Conv2d(layer2_ch, 128, kernel_size=1)
            self.adapters["layer1"] = nn.Conv2d(layer1_ch, 64, kernel_size=1)
            for adapter in self.adapters.values():
                nn.init.kaiming_normal_(adapter.weight, mode="fan_out", nonlinearity="relu")
                if adapter.bias is not None:
                    nn.init.zeros_(adapter.bias)

        # Decoder path
        if is_bottleneck:
            self.up1 = DecoderBlock(512, 256, 256, bilinear)
            self.up2 = DecoderBlock(256, 128, 128, bilinear)
            self.up3 = DecoderBlock(128, 64, 64, bilinear)
        else:
            self.up1 = DecoderBlock(layer4_ch, layer3_ch, 256, bilinear)
            self.up2 = DecoderBlock(256, layer2_ch, 128, bilinear)
            self.up3 = DecoderBlock(128, layer1_ch, 64, bilinear)

        self.up4 = DecoderBlock(64, stem_ch, 64, bilinear)
        self.final_upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.outc = OutConv(64, out_channels)

    @staticmethod
    def _init_conv1_for_multispectral(base_model: nn.Module, in_channels: int) -> None:
        """Initialize conv1 for multispectral input while preserving pretrained weights.

        Uses mean-weight initialization: averages the original RGB weights and repeats
        across all input channels. This preserves edge/texture detection capabilities
        while supporting arbitrary channel counts.

        Args:
            base_model: ResNet base model to modify
            in_channels: Number of input channels for the new conv1 layer
        """
        # Save original 3-channel weights [64, 3, 7, 7]
        original_weights = base_model.conv1.weight.data.clone()

        # Create new layer [64, in_channels, 7, 7]
        new_conv = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        with torch.no_grad():
            # Mean initialization: average RGB and repeat across all channels
            mean_weight = original_weights.mean(dim=1, keepdim=True)  # [64, 1, 7, 7]
            new_conv.weight.copy_(mean_weight.repeat(1, in_channels, 1, 1))

        base_model.conv1 = new_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet-backed U-Net.

        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Output logits tensor of shape [B, out_channels, H, W].
        """
        # Extract features from ResNet backbone
        features = self.backbone(x)
        stem = features["stem"]
        layer1 = features["layer1"]
        layer2 = features["layer2"]
        layer3 = features["layer3"]
        layer4 = features["layer4"]

        # Apply channel adapters if using bottleneck architecture
        if self.adapters:
            layer4 = self.adapters["layer4"](layer4)
            layer3 = self.adapters["layer3"](layer3)
            layer2 = self.adapters["layer2"](layer2)
            layer1 = self.adapters["layer1"](layer1)

        # Decoder with skip connections
        x = self.up1(layer4, layer3)
        x = self.up2(x, layer2)
        x = self.up3(x, layer1)
        x = self.up4(x, stem)

        # Final upsampling to reach full resolution
        x = self.final_upsample(x)

        return self.outc(x)


class ResMTUNet(ResNetUNet):
    """Multi-Task U-Net with ResNet backbone for simultaneous segmentation and regression.

    This class extends ResNetUNet to provide dual outputs for multi-task learning.

    Args:
        in_channels: Number of input channels (e.g., 3 for RGB, 10 for multispectral).
        seg_channels: Number of segmentation classes.
        reg_channels: Number of regression targets.
        backbone: ResNet variant to use ('resnet18', 'resnet34', 'resnet50', 'resnet101').
        pretrained: If True, use ImageNet pretrained weights.
        freeze_backbone: If True, freeze the backbone parameters.
        bilinear: If True, use bilinear upsampling; otherwise use transposed conv.
        dropout: Dropout probability for decoder blocks.
        use_tanh: If True, apply tanh activation to regression outputs.

    Input Shape:
        [B, in_channels, H, W] where H, W should be divisible by 32

    Output Shapes:
        seg_logits: [B, seg_channels, H, W]
        reg_out: [B, reg_channels, H, W] (with tanh if use_tanh=True)

    Example:
        >>> model = ResMTUNet(
        ...     in_channels=10,
        ...     seg_channels=14,
        ...     reg_channels=2,
        ...     backbone='resnet50',
        ...     pretrained=True
        ... )
        >>> x = torch.randn(2, 10, 256, 256)
        >>> seg, reg = model(x)
        >>> seg.shape
        torch.Size([2, 14, 256, 256])
        >>> reg.shape
        torch.Size([2, 2, 256, 256])
    """

    def __init__(
        self,
        in_channels: int,
        seg_channels: int,
        reg_channels: int,
        backbone: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        bilinear: bool = True,
        dropout: float = 0.5,
        use_tanh: bool = True,
    ):
        # Input validation
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if seg_channels <= 0:
            raise ValueError(f"seg_channels must be positive, got {seg_channels}")
        if reg_channels < 0:
            raise ValueError(f"reg_channels must be non-negative, got {reg_channels}")
        if backbone not in BACKBONE_CONFIGS:
            raise ValueError(
                f"backbone must be one of {list(BACKBONE_CONFIGS.keys())}, got {backbone}"
            )

        # Initialize base class with seg_channels (for single output compatibility)
        super().__init__(
            in_channels=in_channels,
            out_channels=seg_channels,
            backbone=backbone,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            bilinear=bilinear,
            dropout=dropout,
        )

        self.use_tanh = use_tanh
        self.tanh = nn.Tanh() if use_tanh else nn.Identity()

        # Replace single output head with dual heads
        del self.outc
        self.seg_head = OutConv(64, seg_channels)
        self.reg_head = OutConv(64, reg_channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through ResNet-backed multi-task U-Net.

        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Tuple of (seg_logits, reg_out):
                - seg_logits: [B, seg_channels, H, W] - segmentation logits
                - reg_out: [B, reg_channels, H, W] - regression output
        """
        # Extract features from ResNet backbone
        features = self.backbone(x)
        stem = features["stem"]
        layer1 = features["layer1"]
        layer2 = features["layer2"]
        layer3 = features["layer3"]
        layer4 = features["layer4"]

        # Apply channel adapters if using bottleneck architecture
        if self.adapters:
            layer4 = self.adapters["layer4"](layer4)
            layer3 = self.adapters["layer3"](layer3)
            layer2 = self.adapters["layer2"](layer2)
            layer1 = self.adapters["layer1"](layer1)

        # Decoder with skip connections
        x = self.up1(layer4, layer3)
        x = self.up2(x, layer2)
        x = self.up3(x, layer1)
        x = self.up4(x, stem)

        # Final upsampling to reach full resolution
        x = self.final_upsample(x)

        # Task-specific outputs
        seg_logits = self.seg_head(x)
        reg_logits = self.reg_head(x)
        reg_out = self.tanh(reg_logits)

        return seg_logits, reg_out
