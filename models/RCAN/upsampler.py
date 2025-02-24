import torch 
import torch.nn as nn

class Upsampler(nn.Sequential):
    def __init__(self, scale, channels):
        layers = []
        for _ in range(int(scale / 2)):
            layers.append(nn.Conv2d(channels, 4 * channels, 3, padding=1, bias=True))
            layers.append(nn.PixelShuffle(2))
        super(Upsampler, self).__init__(*layers)