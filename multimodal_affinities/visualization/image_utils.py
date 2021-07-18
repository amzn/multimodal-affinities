# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-4.0

import cv2
import numpy as np
from PIL import Image
import random
import string
import os

class ImageUtils(object):
    @staticmethod
    def read_image_for_bokeh(image_path, resize_height=None):
        # Open image, and make sure it's RGB*A*
        image = Image.open(image_path).convert('RGBA')
        print("image: {}".format(image))
        if resize_height:
            image = ImageUtils.resize_image_by_height(image, resize_height)

        image_width, image_height = image.size
        # Create an array representation for the image `img`, and an 8-bit "4
        # layer/RGBA" version of it `view`.
        img = np.empty((image_height, image_width), dtype=np.uint32)
        view = img.view(dtype=np.uint8).reshape((image_height, image_width, 4))
        # Copy the RGBA image into view, flipping it so it comes right-side up
        # with a lower-left origin
        view[:, :, :] = np.flipud(np.asarray(image))
        print("input image width x height {}x{}".format(image_width, image_height))
        return view, (image_width, image_height)

    @staticmethod
    def resize_image_by_height(pil_image, dst_height):
        src_width, src_height = pil_image.size
        factor = float(src_height) / dst_height
        dst_width = int(src_width / factor)
        pil_image.thumbnail((dst_width, dst_height), Image.ANTIALIAS)
        return pil_image


def resize_image(img, output_dimensions):
    '''
    resizes an img to output dimensions in x and y while preserving aspect ratio.
    pads (or cuts) along vertical direction if needed
    :param img:
    :param output_dimensions:
    :return:
    '''

    image_width = output_dimensions[0]
    image_height = output_dimensions[1]
    img_shape = img.shape
    num_pad_x = image_width - img.shape[1]
    pad_both_x_and_y = True
    if pad_both_x_and_y and num_pad_x > 0:
        num_pad_l = int(float(num_pad_x) / 2)
        num_pad_r = int(num_pad_x) - num_pad_l
        img = cv2.copyMakeBorder(img, 0, 0, num_pad_l, num_pad_r, cv2.BORDER_WRAP)
    elif not pad_both_x_and_y or num_pad_x < 0:
        resize_factor = float(img_shape[1]) / image_width
        img = cv2.resize(img, (int(img_shape[1] / resize_factor),
                               int(img_shape[0] / resize_factor)))

    num_pad_y = image_height - img.shape[0]
    if num_pad_y > 0:
        num_pad_t = int(float(num_pad_y) / 2)
        num_pad_b = int(num_pad_y) - num_pad_t
        img = cv2.copyMakeBorder(img, num_pad_t, num_pad_b, 0, 0, cv2.BORDER_WRAP)
    elif num_pad_y < 0:
        num_pad_t = int(float(-num_pad_y) / 2)
        num_pad_b = int(-num_pad_y) - num_pad_t
        img = img[num_pad_t:-num_pad_b,:,:]

    # # debugging crops
    # random_filename = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    # cv2.imwrite(os.path.join(output_directory, random_filename + '.jpg'), img)
    return img