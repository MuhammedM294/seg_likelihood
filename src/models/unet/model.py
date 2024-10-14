import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import DoubleConv, EncoderBlock, DecoderBlock


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512]):
        super().__init__()

        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        self.init_conv = DoubleConv(in_channels, features[0])
        # Create encoder path

        for i in range(1, len(features)):
            self.encoder_blocks.append(EncoderBlock(features[i - 1], features[i]))

        # Center part (bottleneck)
        self.center = DoubleConv(features[-1], features[-1] * 2)

        # Create decoder path
        for feature in reversed(features):
            self.decoder_blocks.append(DecoderBlock(feature * 2, feature))

        # Final segmentation layer
        self.segmentation_head = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        x = self.init_conv(x)
        skip_connections.append(x)
        # Encoder path
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            skip_connections.append(x)

        # Bottleneck
        x = self.center(x)

        # Decoder path
        skip_connections = skip_connections[::-1]
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip = skip_connections[i]
            x = decoder_block(x, skip)

        # Final segmentation map
        return self.segmentation_head(x)
