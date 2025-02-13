import cv2
import numpy as np
from PIL import Image

class PSNRComparator:
    def __init__(self, img1_path, img2_path):
        """
        Initialize the PSNR comparator with two image paths.
        :param img1_path: Path to the original low-resolution image.
        :param img2_path: Path to the super-resolution image.
        """
        self.img1_path = img1_path
        self.img2_path = img2_path
        self.img1 = None
        self.img2 = None
    
    def load_images(self):
        """Load and process images."""
        original_img = Image.open(self.img1_path)
        
        # Upscale using bicubic interpolation
        self.img1 = original_img.resize((original_img.width * 2, original_img.height * 2), Image.BICUBIC)
        self.img1 = cv2.cvtColor(np.array(self.img1), cv2.COLOR_RGB2GRAY)
        
        # Load super-resolution image in grayscale
        self.img2 = cv2.imread(self.img2_path, cv2.IMREAD_GRAYSCALE)
    
    @staticmethod
    def calculate_psnr(img1, img2):
        """Calculate the PSNR between two images."""
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr
    
    def compare(self):
        """Perform PSNR comparison and return the result."""
        if self.img1 is None or self.img2 is None:
            self.load_images()
        return self.calculate_psnr(self.img1, self.img2)