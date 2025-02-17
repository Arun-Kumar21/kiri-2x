import torch.nn as nn

from .residual_block import ResidualBlock
from .upsampler import UpSampler

class EDSR(nn.Module):
    def __init__(self, hidden_channels = 64, num_residual_block = 20, scale_factor=2):
        super().__init__()

        self.conv1 = nn.Conv2d(3, hidden_channels, kernel_size=3, padding=1)

        self.res_blocks = nn.Sequential(*[ResidualBlock(hidden_channels) for _ in range(num_residual_block)])

        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)

        # Upsampling Layer
        self.upsample = UpSampler(hidden_channels, scale_factor)

        # Output Convolution
        self.conv3 = nn.Conv2d(hidden_channels, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        residual = self.res_blocks(x)
        x = x + residual
        x = self.conv2(x)
        x = self.upsample(x)
        x = self.conv3(x)
        return x