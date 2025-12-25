# unet_model.py (fixed)
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class DoubleConv(nn.Module):
    """(Conv => InstanceNorm => Swish) * 2 with Residual connection"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            Swish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            Swish()
        )
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.conv_block(x) + self.res_conv(x)


class Down(nn.Module):
    """Downscale with MaxPool then DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """
    Upscale then DoubleConv.

    in_channels: number of channels coming from the deeper layer (before any reduction).
                 e.g. 1024 at bottleneck.
    out_channels: desired channels after the block (e.g. 512).
    """
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            # Upsample, then reduce channels so that concat works as expected
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            # reduce channels from in_channels -> in_channels//2 (to match skip connection channels)
            self.reduce = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False)
            # After concat (in_channels//2 + in_channels//2 = in_channels), feed to DoubleConv(in_channels, out_channels)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            # ConvTranspose: take in_channels -> in_channels//2 (upsampled)
            # i.e. input to convtranspose will have in_channels channels (x1), and it outputs in_channels//2
            # We therefore define convtranspose to accept in_channels and output in_channels//2
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # after upsample x1 has in_channels//2, skip x2 has in_channels//2, concat -> in_channels // -> feed conv
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        x1: deeper feature map (channels = in_channels)
        x2: skip connection from encoder (channels = in_channels // 2)
        """
        x1 = self.up(x1)

        # for bilinear branch we applied reduce earlier in __init__
        if self.bilinear:
            x1 = self.reduce(x1)

        # pad if sizes mismatch (rare for odd sizes)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        if diffY != 0 or diffX != 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

        # concatenate along channel dim: [B, C_skip + C_up, H, W]
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=False, use_checkpoint=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_checkpoint = use_checkpoint

        factor = 2 if bilinear else 1

        # Encoder (channel scheme: 64,128,256,512,1024)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)  # if bilinear True, bottom channels are 512

        # Decoder (must match concatenation logic)
        # up1: input channels at bottleneck == 1024//factor
        # For non-bilinear factor=1 => in_channels=1024 -> convtranspose -> out 512; skip x4 has 512 -> concat -> 1024 -> DoubleConv(1024, 512)
        self.up1 = Up(1024 // factor, 512 // factor, bilinear)
        self.up2 = Up(512 // factor, 256 // factor, bilinear)
        self.up3 = Up(256 // factor, 128 // factor, bilinear)
        self.up4 = Up(128 // factor, 64, bilinear)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        def run(layer, *inputs):
            return checkpoint.checkpoint(layer, *inputs) if self.use_checkpoint else layer(*inputs)

        x1 = run(self.inc, x)     # 64
        x2 = run(self.down1, x1)  # 128
        x3 = run(self.down2, x2)  # 256
        x4 = run(self.down3, x3)  # 512
        x5 = run(self.down4, x4)  # 1024//factor

        x = run(self.up1, x5, x4)
        x = run(self.up2, x, x3)
        x = run(self.up3, x, x2)
        x = run(self.up4, x, x1)
        logits = run(self.outc, x)

        return logits
