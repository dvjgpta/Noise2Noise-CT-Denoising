""" Full assembly of the parts to form the complete UNet network for CT denoising """

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=False, use_checkpoint=False):
        """
        U-Net architecture for grayscale CT image denoising.

        Args:
            n_channels (int): Input channels (1 for grayscale CT).
            n_classes (int): Output channels (1 for regression/denoising).
            bilinear (bool): Use bilinear upsampling instead of transposed conv.
            use_checkpoint (bool): Enable gradient checkpointing for memory efficiency.
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_checkpoint = use_checkpoint

        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Helper to apply checkpointing only when requested
        def run(layer, *inputs):
            if self.use_checkpoint:
                return checkpoint.checkpoint(layer, *inputs)
            else:
                return layer(*inputs)

        # Encoder path
        x1 = run(self.inc, x)
        x2 = run(self.down1, x1)
        x3 = run(self.down2, x2)
        x4 = run(self.down3, x3)
        x5 = run(self.down4, x4)

        # Decoder path
        x = run(self.up1, x5, x4)
        x = run(self.up2, x, x3)
        x = run(self.up3, x, x2)
        x = run(self.up4, x, x1)
        logits = run(self.outc, x)

        return logits
