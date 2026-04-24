import torch
import torch.nn as nn


def enable_dropout(model: nn.Module) -> None:
    """
    Re-enable all Dropout2d layers after model.eval().

    Call this immediately after model.eval() to activate MC Dropout:
        model.eval()
        enable_dropout(model)

    This keeps BatchNorm in eval mode (uses running stats, not batch stats)
    while keeping Dropout active (randomly zeroing feature maps each forward pass),
    which is the correct setup for Monte Carlo Dropout inference.
    """
    for m in model.modules():
        if isinstance(m, nn.Dropout2d):
            m.train()


class Model(nn.Module):
    """
    Vanilla U-Net with MC Dropout.

    Dropout2d layers are added after every DoubleConv block (encoder,
    bottleneck, and decoder).  During training they behave as standard
    regularisation.  At inference time, calling enable_dropout(model)
    after model.eval() keeps them active for Monte Carlo Dropout.
    """
    def __init__(self, in_channels=3, n_classes=19, dropout_p=0.5):
        super().__init__()
        self.in_channels = in_channels

        # Encoding path
        self.inc   = DoubleConv(in_channels, 64,  dropout_p=dropout_p)
        self.down1 = Down(64,  128, dropout_p=dropout_p)
        self.down2 = Down(128, 256, dropout_p=dropout_p)
        self.down3 = Down(256, 512, dropout_p=dropout_p)
        self.down4 = Down(512, 512, dropout_p=dropout_p)

        # Decoding path
        self.up1  = Up(1024, 256, dropout_p=dropout_p)
        self.up2  = Up(512,  128, dropout_p=dropout_p)
        self.up3  = Up(256,  64,  dropout_p=dropout_p)
        self.up4  = Up(128,  64,  dropout_p=dropout_p)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {x.shape[1]}")

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)
        return self.outc(x)


class DoubleConv(nn.Module):
    """(Conv → BN → ReLU) × 2, followed by Dropout2d."""

    def __init__(self, in_channels, out_channels, mid_channels=None, dropout_p=0.5):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """MaxPool → DoubleConv."""

    def __init__(self, in_channels, out_channels, dropout_p=0.5):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout_p=dropout_p),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Bilinear upsample → DoubleConv."""

    def __init__(self, in_channels, out_channels, bilinear=True, dropout_p=0.5):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, dropout_p=dropout_p)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x  = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
