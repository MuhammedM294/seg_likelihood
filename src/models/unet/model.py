import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import DoubleConv, EncoderBlock, DecoderBlock


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512]):
        super().__init__()

        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        # Create encoder path
        for feature in features:
            self.encoder_blocks.append(EncoderBlock(in_channels, feature))
            in_channels = feature

        # Center part (bottleneck)
        self.center = DoubleConv(features[-1], features[-1] * 2)

        # Create decoder path
        for feature in reversed(features):
            self.decoder_blocks.append(DecoderBlock(feature * 2, feature))

        # Final segmentation layer
        self.segmentation_head = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder path
        for encoder_block in self.encoder_blocks:
            x, skip = encoder_block(x)
            skip_connections.append(skip)

        # Bottleneck
        x = self.center(x)

        # Decoder path
        skip_connections = skip_connections[::-1]
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip = skip_connections[i]
            x = decoder_block(x, skip)

        # Final segmentation map
        return self.segmentation_head(x)
