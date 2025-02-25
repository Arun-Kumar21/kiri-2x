import torch
import torch.nn as nn

class DnCNN(nn.Module):
    def __init__(self, num_layers=20, num_features=64):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, num_features, kernel_size=3, padding=1, stride=1, bias=False),
            nn.ReLU(inplace=True),

            *[nn.Sequential(
                nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(num_features),
                nn.ReLU(inplace=True)
            ) for _ in range(num_layers-2)],
            
            nn.Conv2d(num_features, 3, kernel_size=3, padding=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        noise = self.layers(x)

        return x - noise