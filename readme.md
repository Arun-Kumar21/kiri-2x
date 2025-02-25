# kiri-2x

Image super-resolution for anime-style artwork using a deep convolutional neural network, primarily based on SRCNN [Research Paper](https://arxiv.org/abs/1501.00092).

## Results:

![image](https://myjournalbucket-arun.s3.eu-north-1.amazonaws.com/waifu-upscale-comparsion.png)

Compared to Waifu2x, which utilizes a more advanced deep convolutional network with integrated noise reduction, SRCNN produces inferior results due to its simpler architecture.

> Work is in progress to improve the results to match Waifu2x quality.

## Enchanced version:

Implementing Enhanced Deep Residual Networks (EDSR) for further improvement in image super-resolution quality [Research Paper](https://arxiv.org/abs/1707.02921)

## Results:

![EDSR](https://myjournalbucket-arun.s3.eu-north-1.amazonaws.com/mizuki-edsr-waifu2x.png)

Compared to Waifu2x, the results are comparable, with EDSR performing at the same level as Waifu2x.

## Usage

Perform super-resolution on an image:

1. Place your input image in the `images/` folder.
2. Update the file path in `super_resolution.py` to point to your image.
3. Choose an output file name for the upscaled image.
4. Run the following command:

```bash
python super_resolution.py
```

> The default super_resolution.py uses a pre-trained model stored in the `weights/` directory.
