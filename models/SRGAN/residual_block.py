import torch
from torch import nn

from .conv_block import ConvBlock

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv1 = ConvBlock(
            in_channels, 
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.conv2 = ConvBlock(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_act=False
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        return x + out