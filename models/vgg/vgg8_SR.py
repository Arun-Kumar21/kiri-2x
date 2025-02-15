import torch
from torch import nn

class VGG8_SR(nn.Module):
    def __init__(self):
        super(VGG8_SR, self).__init__()
        
        self.convBlock1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
        )
        
        self.convBlock2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
        )
        
        self.convBlock3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
        )
        
        self.convBlock4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
        )
        
        self.finalConv = nn.Conv2d(256, 3, kernel_size=3, padding=1, stride=1)
        
    def forward(self, x): 
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = self.convBlock3(x)
        # x = self.convBlock4(x)
        
        x = self.finalConv(x)
         
        return x
