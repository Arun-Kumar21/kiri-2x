import torchvision.transforms.functional as TF

from PIL import Image

def preprocessing_img(image_path, device,upscale_factor=2):
    img = Image.open(image_path).convert("RGB")

    # if image is not low resolution
    # img = img.resize((img.width // upscale_factor, img.height // upscale_factor), Image.BICUBIC) 

    img = img.resize((img.width * upscale_factor, img.height * upscale_factor), Image.BICUBIC)

    lr_tensor = TF.to_tensor(img).unsqueeze(0).to(device)
    return img, lr_tensor