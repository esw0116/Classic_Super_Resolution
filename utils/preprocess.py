import numpy as np
from PIL import Image

image_name = 'house'
img_path = "sample/{}.jpg".format(image_name)
upscale_factor = 3


def image_ycc(path):
    imagefile = Image.open(path)
    imagefile = imagefile.convert('YCbCr')
    image_ycc = np.array(imagefile)

    return image_ycc


def ycbcr_2_rgb(image):
    image_rgb = image.convert('RGB')

    return image_rgb


def crop_input(input, label):
    # crop input(low_resolution image) size to label(high resolution image) size
    h = label.shape[0]
    w = label.shape[1]
    input = input[0:h, 0:w, :]
    return input
