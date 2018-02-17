import numpy as np
from PIL import Image


def convert_btw_rgb_ycbcr(image, dir):
    if dir:  # rgb 2 ycbcr
        image_ycbcr = image.convert('YCbCr')
        return image_ycbcr
    else:  # ycbcr to rgb
        image_rgb = image.convert('RGB')
        return image_rgb


def crop_input(input, label):
    # crop input(low_resolution image) size to label(high resolution image) size
    h = label.shape[0]
    w = label.shape[1]
    input = input[0:h, 0:w, :]
    return input


def image_label_gen(image_path, label_path):
    image = np.array(convert_btw_rgb_ycbcr(Image.open(image_path), dir=True))
    label = np.array(convert_btw_rgb_ycbcr(Image.open(label_path), dir=True))
    image = crop_input(input=image, label=label)

    return image, label


if __name__ == '__main__':
    image_name = 'house'
    img_path = "sample/{}.jpg".format(image_name)
    upscale_factor = 3