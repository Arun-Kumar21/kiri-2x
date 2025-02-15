import torch 
import torchvision.transforms.functional as TF

from models.srcnn import SRCNN
from models.vgg.vgg8_SR import VGG8_SR
from models.vgg.vgg16_pretrained import VGG16Upscaler

from config.config import CONFIG

from lib.preprocessing import preprocessing_img
from lib.compare_plt import ImageComparator

from evaluation.psnr import PSNRComparator


class SuperResolution:
    def __init__(self, model_path, model, device=CONFIG.DEVICE):
        """
        Initialize the SuperResolution model.
        :param model_path: Path to the pre-trained model.
        :param device: Device to run the model on (CPU/GPU).
        """
        self.device = device
        self.model = model.to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
    
    def upscale(self, img_path, save_path=None, show=False):
        """
        Perform super-resolution on an image.
        :param img_path: Path to the low-resolution image.
        :param save_path: Path to save the output image (optional).
        :param show: Whether to display the image after processing.
        """
        lr_image, lr_tensor = preprocessing_img(img_path, device=self.device)
        
        with torch.no_grad():
            sr_tensor = self.model(lr_tensor).clamp(0, 1)
        
        sr_image = TF.to_pil_image(sr_tensor.squeeze(0))
        
        if save_path:
            sr_image.save(save_path)
        
        if show:
            sr_image.show()
        
        return sr_image

if __name__ == "__main__":
    model = VGG8_SR().to(CONFIG.DEVICE)
    sr = SuperResolution("weights/vgg8_rgb.pth", model)
    sr.upscale("images/waifu/waifu_low.jpg", save_path="images/waifu/waifu_VGG8_SR.jpg", show=True)
    
    # cmp = ImageComparator(['images/waifu/waifu_low.jpg', 'images/waifu/waifu_SR.jpg', 'images/waifu/waifu_VGG8_SR.jpg'], [50, 50, 150, 150])
    # cmp.compare()

    # cmp = PSNRComparator('images/waifu/waifu_low.jpg', 'images/waifu/waifu_VGG8_SR.jpg')
    # dif = cmp.compare()
    # print(dif)

    