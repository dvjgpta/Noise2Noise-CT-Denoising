import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from unet_noise_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=False, use_checkpoint=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.use_checkpoint = use_checkpoint

        # CHANGED: reduced width (closer to MRI architecture)
        self.inc = DoubleConv(n_channels, 48)              # CHANGED
        self.down1 = Down(48, 48)                          # CHANGED
        self.down2 = Down(48, 48)                          # CHANGED
        self.down3 = Down(48, 48)                          # CHANGED
        self.down4 = Down(48, 48)                          # CHANGED

        # Decoder channels mimic NVIDIA (48→96)
        self.up1 = Up(96, 96)                              # CHANGED
        self.up2 = Up(144, 96)                             # CHANGED
        self.up3 = Up(144, 96)                             # CHANGED
        self.up4 = Up(144, 48)                             # CHANGED

        self.outc = OutConv(48, n_classes)

    def forward(self, x):
        def run(layer, *inputs):
            return checkpoint.checkpoint(layer, *inputs) if self.use_checkpoint else layer(*inputs)

        x1 = run(self.inc, x)
        x2 = run(self.down1, x1)
        x3 = run(self.down2, x2)
        x4 = run(self.down3, x3)
        x5 = run(self.down4, x4)

        x = run(self.up1, x5, x4)
        x = run(self.up2, x, x3)
        x = run(self.up3, x, x2)
        x = run(self.up4, x, x1)

        logits = run(self.outc, x)
        return logits