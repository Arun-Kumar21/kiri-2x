import torch
import torch.nn as nn

class EDSRPerceptualLoss(nn.Module):
    def __init__(self):
        super(EDSRPerceptualLoss, self).__init__()
        self.l1_loss = nn.L1Loss() 

    def forward(self, sr, hr, sr_features, hr_features):
        pixel_loss = self.l1_loss(sr, hr) 
        feature_loss = self.l1_loss(sr_features, hr_features) 
        return pixel_loss + 0.01 * feature_loss 
