import torch 
import torch.nn as nn

from .RCAB import RCAB
from .RG import ResidualGroup
from .upsampler import Upsampler

class RCANModel(nn.Module):
    def __init__(self, scale, num_channels = 3, num_features=64, num_blocks=20, num_groups=10, reduction=16):
        super().__init__()

        self.head = nn.Conv2d(num_channels, num_features, 3, padding=1, bias=True)

        self.body = nn.Sequential(*[ResidualGroup(num_blocks, num_features, reduction) for _ in range(num_groups)])
        self.conv_body = nn.Conv2d(num_features, num_features, 3, padding=1, bias=True)

        self.upsampler = Upsampler(scale, num_features)

        self.tail = nn.Conv2d(num_features, num_channels, 3, padding=1, bias=True)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res = self.conv_body(res)
        res += x
        x = self.upsampler(res)
        x = self.tail(x)

        return x