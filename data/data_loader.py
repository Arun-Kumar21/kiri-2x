from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import os


transform = transforms.Compose([
    transforms.ToTensor()
])

class SRDataset(Dataset):
    def __init__(self, root, upscale_factor=2, image_size=(256, 256)):
        self.image_dir = os.path.join(root, "images") 
        self.image_paths = sorted([os.path.join(self.image_dir, f) 
                                   for f in os.listdir(self.image_dir) 
                                   if f.endswith(('.jpg', '.png'))])

        self.upscale_factor = upscale_factor
        self.image_size = image_size 

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index]).convert("RGB") 
        img = img.resize(self.image_size, Image.BICUBIC)

        hr = TF.to_tensor(img)

        # Create Low-Resolution image
        lr = TF.resize(img, [self.image_size[0] // self.upscale_factor, 
                             self.image_size[1] // self.upscale_factor], 
                       interpolation=Image.BICUBIC)
        lr = TF.resize(lr, self.image_size, interpolation=Image.BICUBIC) 

        return TF.to_tensor(lr), hr

# Load datasets
train_dataset = SRDataset(root="data/dataset/train", image_size=(256, 256))
valid_dataset = SRDataset(root="data/dataset/valid", image_size=(256, 256))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)