"""
This U-Net implementation is based on https://github.com/milesial/Pytorch-UNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        in_channels: Number of channels after concatenation (upsampled + skip).
        out_channels: Number of output channels.
        bilinear: If True, use bilinear upsampling; otherwise use transposed conv.

    Input Shapes:
        x1: [B, in_channels//2, H, W] - from deeper layer (or in_channels when bilinear=False)
        x2: [B, in_channels//2, H*2, W*2] - skip connection

    Output Shape:
        [B, out_channels, H*2, W*2]
    """

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()

        if bilinear:
            # Bilinear upsampling: preserves channels, then concat with skip gives in_channels
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            # ConvTranspose2d: reduces channels from in_channels to out_channels
            # After concat with skip (which has out_channels), total is in_channels
            self.up = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2
            )

        self.conv = ConvBlock(in_channels, out_channels, dropout=0)

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
        self.up1 = DecoderBlock(1024, 512 // factor, bilinear)
        self.up2 = DecoderBlock(512, 256 // factor, bilinear)
        self.up3 = DecoderBlock(256, 128 // factor, bilinear)
        self.up4 = DecoderBlock(128, 64, bilinear)

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
        self.up1 = DecoderBlock(1024, 512 // factor, bilinear)
        self.up2 = DecoderBlock(512, 256 // factor, bilinear)
        self.up3 = DecoderBlock(256, 128 // factor, bilinear)
        self.up4 = DecoderBlock(128, 64, bilinear)

        # Task-specific output heads
        self.seg_head = OutConv(64, seg_channels)
        self.reg_head = OutConv(64, reg_channels)

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
