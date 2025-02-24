import torch
import torch.nn as nn

from .RCAB import RCAB

class ResidualGroup(nn.Module):
    def __init__(self, num_blocks, channels, reduction):
        super().__init__()
        self.block = nn.Sequential(*[RCAB(channels, reduction) for _ in range(num_blocks)])
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=True)

    def forward(self, x):
        res = self.block(x)
        res = self.conv(res)

        return x + res