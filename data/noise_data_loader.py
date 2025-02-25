from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import torch
from PIL import Image
import os

from config.config import CONFIG

transform = transforms.Compose([
   transforms.ToTensor() 
])

class NoisyDataSet(Dataset):
    def __init__(self, root, noise_std = 50):
        self.images_dir = os.path.join(root, 'images')
        self.images_path = sorted([os.path.join(self.images_dir, f) 
                                   for f in os.listdir(self.images_dir) 
                                   if f.endswith(('.jpg', '.png'))])
        
        self.noise_std = noise_std / 255
        self.transform = transform

    def __len__(self):
        return len(self.images_path)
    
    def __getitem__(self, index):
        clean_img = Image.open(self.images_path[index]).convert("RGB")
        clean_img = self.transform(clean_img)

        noise = torch.randn_like(clean_img) * self.noise_std
        # y = x + w
        noise_img = clean_img + noise

        return noise_img.clamp(0, 1), clean_img


noise_train_dataset = NoisyDataSet('data/dataset/train')
noise_val_dataset = NoisyDataSet('data/dataset/valid')

noise_train_loader = DataLoader(noise_train_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True)
noise_train_loader = DataLoader(noise_val_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False)
