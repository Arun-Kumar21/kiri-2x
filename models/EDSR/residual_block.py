import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.res_scale = 0.1

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        return x + self.res_scale * residual
    