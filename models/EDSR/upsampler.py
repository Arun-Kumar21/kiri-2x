import torch.nn as nn


class UpSampler(nn.Module):
    def __init__(self, num_channels, scale_factor=2):
        super().__init__()

        self.conv = nn.Conv2d(num_channels, num_channels * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)

        return self.relu(x)
    