import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import math

class ImageComparator:
    def __init__(self, image_paths, image_labels, crop_coords):
        self.image_paths = image_paths
        self.image_labels = image_labels
        self.crop_coords = crop_coords
        self.images = []
    
    def load_images(self):
        """Load and process images."""
        for path in self.image_paths:
            img = Image.open(path).convert("RGB")
            self.images.append(img)
    
    def crop_images(self):
        """Crop images based on provided coordinates."""
        cropped = []
        for i, img in enumerate(self.images):
            if i == 0:  # Original image
                cropped.append(img.crop(self.crop_coords))
            else:  # Processed images (assuming 2x resolution)
                scaled_coords = (
                    self.crop_coords[0] * 2,
                    self.crop_coords[1] * 2,
                    self.crop_coords[2] * 2,
                    self.crop_coords[3] * 2
                )
                cropped.append(img.crop(scaled_coords))
        return cropped
    
    def compare(self):
        """Display comparison with original image and zoomed regions."""
        if not self.images:
            self.load_images()
        
        cropped_images = self.crop_images()
        
        # Calculate grid layout for cropped images
        num_cropped = len(cropped_images)
        grid_cols = 2  # Number of columns in the grid
        grid_rows = math.ceil(num_cropped / grid_cols)
        
        # Create figure with gridspec for custom layout
        fig = plt.figure(figsize=(10, 6))
        
        # Create a larger subplot for the original image
        ax_original = plt.subplot2grid((grid_rows, grid_cols+1), (0, 0), rowspan=grid_rows, colspan=1)
        
        # Display original image with red rectangle
        ax_original.imshow(self.images[0])
        ax_original.add_patch(Rectangle(
            (self.crop_coords[0], self.crop_coords[1]),
            self.crop_coords[2] - self.crop_coords[0],
            self.crop_coords[3] - self.crop_coords[1],
            linewidth=2, edgecolor='r', facecolor='none'
        ))
        ax_original.set_title("Original Image", fontsize=12)
        ax_original.axis('off')
        
        # Display cropped images in a grid layout
        titles = ["Original (Zoomed)"] + [f"{lbl} (Zoomed)" for lbl in self.image_labels]
        
        for i, (img, title) in enumerate(zip(cropped_images, titles)):
            row = i // grid_cols
            col = i % grid_cols + 1  # +1 to place after the original image
            
            ax = plt.subplot2grid((grid_rows, grid_cols+1), (row, col))
            ax.imshow(img)
            ax.set_title(title)
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()