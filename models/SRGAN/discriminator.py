import torch
from torch import nn

from .conv_block import ConvBlock

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 64, 128, 128, 256, 256, 512, 512]):
        super().__init__()

        blocks=[]
        for idx, feature in enumerate(features):
            blocks.append(
                ConvBlock(
                    in_channels,
                    feature,
                    kernel_size=3,
                    padding=1,
                    stride= 1+ idx % 2,
                    discriminator=True,
                    use_act=True,
                    use_bn=False if idx==0 else True
                )
            )
            in_channels = feature

        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512*6*6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )

    def foward(self, x):
        x = self.blocks(x)
        return self.classifier(x)