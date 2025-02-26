import torch 
from torch import nn

class Upsample(nn.Module):
    def __init__(self, in_channels, scale=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * scale ** 2, 3, 1, 1)
        self.ps = nn.PixelShuffle(scale)
        self.act = nn.PReLU(num_parameters=in_channels)

    def forward(self, x):
        return self.act(self.ps(self.conv(x)))