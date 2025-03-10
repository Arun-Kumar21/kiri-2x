import torch 
import torchvision.transforms.functional as TF
from PIL import Image

from models.srcnn import SRCNN
from models.vgg.vgg8_SR import VGG8_SR
from models.vgg.vgg16_pretrained import VGG16Upscaler
from models.DnCNN.dncnn import DnCNN
from models.EDSR.edsr import EDSR
from models.RCAN.model import RCANModel


from config.config import CONFIG

from lib.preprocessing import preprocessing_img
from lib.compare_plt import ImageComparator

from evaluation.psnr import PSNRComparator


class SuperResolution:
    def __init__(self, model_path, model, device=CONFIG.DEVICE):
        self.device = device
        self.model = model.to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()
    
    def upscale(self, img_path, save_path=None, show=False):
        # lr_image, lr_tensor = preprocessing_img(img_path, device=self.device)
        img = Image.open(img_path).convert("RGB")
        img = TF.to_tensor(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            sr_tensor = self.model(img).clamp(0, 1)
        
        sr_image = TF.to_pil_image(sr_tensor.squeeze(0))
        
        if save_path:
            sr_image.save(save_path)
        
        if show:
            sr_image.show()
        
        return sr_image

# if __name__ == "__main__":
#     # noise_model = DnCNN().to(CONFIG.DEVICE)
#     # cleanSr = SuperResolution("weights/dncnn.pth", noise_model)
#     # cleanSr.upscale("images/other/mizuki.png", save_path="images/other/mizuki-DNCNN.png", show=True)
    

#     # model = EDSR().to(CONFIG.DEVICE)
#     # sr = SuperResolution('weights/edsr.pth', model, CONFIG.DEVICE)
#     # sr.upscale('images/other/mizuki.png', save_path='images/other/mizuki_EDSR.png', show=True)

#     # cmp = ImageComparator(['images/other/mizuki.png', 'images/other/mizuki-DNCNN_EDSR.png', 'images/other/mizuki_2x.png'], [ "EDSR-DnCNN", "Internet"], [50, 50, 150, 150])
#     cmp = ImageComparator(
#     image_paths=[
#         "images/other/mizuki.png",             
#         "images/other/mizuki_EDSR.png",
#         "images/other/mizuki_waifu2x.png"        
#     ],
#     image_labels=["EDSR", "Waifu2x"], 
#     crop_coords=(1600, 700, 1650, 750)
#     )

#     cmp.compare()  # Run the comparison
