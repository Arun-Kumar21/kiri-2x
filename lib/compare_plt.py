import matplotlib.pyplot as plt
from PIL import Image

class ImageComparator:
    def __init__(self, image_paths, crop_coords):
        """
        Initialize the ImageComparator with image paths and cropping coordinates.
        :param image_paths: List of image file paths.
        :param crop_coords: Tuple containing cropping coordinates (x1, y1, x2, y2).
        """
        self.image_paths = image_paths
        self.crop_coords = crop_coords
        self.images = []
    
    def load_images(self):
        """Load and process images."""
        for path in self.image_paths:
            img = Image.open(path).convert("RGB")
            self.images.append(img)
    
    def crop_images(self):
        """Crop images based on the provided coordinates."""
        return [
            img.crop((
                self.crop_coords[0] * 2, self.crop_coords[1] * 2,
                self.crop_coords[2] * 2, self.crop_coords[3] * 2
            )) if i > 0 else img.crop(self.crop_coords)
            for i, img in enumerate(self.images)
        ]
    
    def compare(self):
        """Display the images side by side for comparison."""
        if not self.images:
            self.load_images()
        cropped_images = self.crop_images()
        
        fig, axes = plt.subplots(1, len(self.image_paths), figsize=(10, 5))
        titles = ["Original (Zoomed In)"] + [f"Image {i+1} (Zoomed In)" for i in range(1, len(self.image_paths))]
        
        for ax, img, title in zip(axes, cropped_images, titles):
            ax.imshow(img)
            ax.set_title(title)
            ax.axis("off")
        
        plt.show()

