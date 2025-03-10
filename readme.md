# kiri-2x

Image super-resolution for anime-style artwork using a deep convolutional neural network, primarily based on SRCNN [Research Paper](https://arxiv.org/abs/1501.00092).

## Results:

![image](https://myjournalbucket-arun.s3.eu-north-1.amazonaws.com/waifu-upscale-comparsion.png)

Compared to Waifu2x, which utilizes a more advanced deep convolutional network with integrated noise reduction, SRCNN produces inferior results due to its simpler architecture.

> Work is in progress to improve the results to match Waifu2x quality.

## Enhanced version:

Implementing Enhanced Deep Residual Networks (EDSR) for further improvement in image super-resolution quality [Research Paper](https://arxiv.org/abs/1707.02921)

## Results:

![EDSR](https://myjournalbucket-arun.s3.eu-north-1.amazonaws.com/mizuki-edsr-waifu2x.png)

Compared to Waifu2x, the results are comparable, with EDSR performing at the same level as Waifu2x.

## How to use

Perform super-resolution on an image:

1. Clone the repository:

```bash
git clone https://github.com/Arun-Kumar21/kiri-2x.git
cd kiri-2x
```

2. Install required dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Run the GUI with the following script:

```bash
python gui.py
```

4. You will see an interface like this:

![GUI](https://myjournalbucket-arun.s3.eu-north-1.amazonaws.com/gui.png)

6. Select the image for upscaling by clicking the `Browse` button.
7. Choose the save path by clicking the `Browse Save Path` button.
8. Click the `Start Upscaling` button.
9. Once upscaling is complete, a message dialog will appear, and your file will be saved to the specified path with the name `filename_2x.png`.

> The default super_resolution.py uses a pre-trained model stored in the `weights/` directory.
