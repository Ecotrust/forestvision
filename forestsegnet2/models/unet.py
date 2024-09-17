"""
This U-Net implementation is based on https://github.com/milesial/Pytorch-UNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from torchmetrics import R2Score
import lightning as pl
import numpy as np
import matplotlib.pyplot as plt


class ConvBlock(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, dropout=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        if dropout:
            self.conv.append(nn.Dropout())

    def forward(self, x):
        return self.conv(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(EncoderBlock, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2), ConvBlock(in_channels, out_channels, dropout=dropout)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(DecoderBlock, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)

        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Make sure x1 and x2 have the same size
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2)))

        # Concat x2 and x1; this is the skip connection
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        # self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 1),
            nn.BatchNorm2d(in_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 2, out_ch, 1),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, dropout=False):
        super(UNet, self).__init__()
        factor = 2 if bilinear else 1

        # Encoder
        self.inc = ConvBlock(in_channels, 64)
        self.down1 = EncoderBlock(64, 128)
        self.down2 = EncoderBlock(128, 256)
        self.down3 = EncoderBlock(256, 512)
        self.down4 = EncoderBlock(512, 1024 // factor, dropout=dropout)

        # Decoder
        self.up1 = DecoderBlock(1024, 512 // factor, bilinear)
        self.up2 = DecoderBlock(512, 256 // factor, bilinear)
        self.up3 = DecoderBlock(256, 128 // factor, bilinear)
        self.up4 = DecoderBlock(128, 64, bilinear)

        # if out_channels == 1:
        #     self.sem_out = nn.Conv2d(64, out_channels, kernel_size=1)
        # else:
        self.sem_out = OutConv(64, out_channels)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.sem_out(x)


def test():
    from torchinfo import summary

    x = torch.randn((3, 1, 160, 160))
    model = UNet(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape, x.shape)
    summary(model, x.shape)
    assert preds.shape == x.shape, "Shape mismatch"


if __name__ == "__main__":
    test()
