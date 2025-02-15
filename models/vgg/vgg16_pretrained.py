import torch 
import torch.nn as nn
import torchvision.models as model

class VGG16Upscaler(nn.Module):
    def __init__(self):
        super(VGG16Upscaler, self).__init__()

        vgg16 = model.vgg16(weights=model.VGG16_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(vgg16.features.children())[:-1])

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1) 
        )

    def forward(self, x):
        x = self.feature_extractor(x) 
        x = self.upsample(x) 
        return x
