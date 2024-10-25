import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        if dropout:
            self.double_conv.append(nn.Dropout2d(p=0.1))

    def forward(self, x):
        return self.double_conv(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.max_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):

        return self.max_conv(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)  # After concatenation

    def forward(self, x, skip_features):
        x = self.up(x)
        if x.shape != skip_features.shape:
            x = torch.nn.functional.interpolate(
                x, size=skip_features.shape[-2:], mode="bilinear", align_corners=True
            )
        x = torch.cat([x, skip_features], dim=1)  # Concatenate skip features
        x = self.conv(x)
        return x
