import random
import numpy as np
from random import randint
from others.other_components.daman.daman_util.daman_helps_docker import StdIP as IP


def unet_augment(sample, use_weight, use_edgeweight):
    image, masks = sample['image'], sample['masks']
    if use_weight:
        wt = sample['weight']

    if use_edgeweight:
        edgewt = sample['edgeweight']

    vertical_prob = 0.5
    horizontal_prob = 0.5
    gaussian_prob = 0.4
    blur_prob = 0.4
    gamma_prob = 0.3
    bright_prob = 0.3

    if (random.random() < vertical_prob):
        image = np.flipud(image)
        masks = np.flipud(masks)
        if use_weight:
            wt = np.flipud(wt)
        if use_edgeweight:
            edgewt = np.flipud(edgewt)

    if (random.random() < horizontal_prob):
        image = np.fliplr(image)
        masks = np.fliplr(masks)
        if use_weight:
            wt = np.fliplr(wt)
        if use_edgeweight:
            edgewt = np.fliplr(edgewt)

    if (random.random() < bright_prob):  # brightness change
        value = randint(2, 15)
        image = image + value
        image[image > 255] = 255.
        image[image < 0] = 0.
        masks = masks
        if use_weight:
            wt = wt
        if use_edgeweight:
            edgewt = edgewt

    if (random.random() < gamma_prob):  # Gaussian noise
        im = image
        row, col = im.shape
        const = 2
        b = 2
        a = 0.1
        gamma = (1 / ((b - a) * random.random() + a))
        gamma_im = const * (im ** gamma)
        image = gamma_im
        masks = masks
        if use_weight:
            wt = wt
        if use_edgeweight:
            edgewt = edgewt

    image = IP.linstretch(image)
    sample_return = {'image': image, 'masks': masks}

    if use_weight:
        sample_return['weight'] = wt

    if use_edgeweight:
        sample_return['edgeweight'] = edgewt

    return sample_return


def man_augment(image, masks, edgewt):
    """
    Assumes that the sample is {'image', 'masks', 'edgeweight'}
    :param sample:
    :return:
    """
    vertical_prob = 0.5
    horizontal_prob = 0.5
    gaussian_prob = 0.4
    blur_prob = 0.4
    gamma_prob = 0.3
    bright_prob = 0.3
    if (random.random() < vertical_prob):
        image = np.flipud(image)
        masks = np.flipud(masks)
        edgewt = np.flipud(edgewt)

    if (random.random() < horizontal_prob):
        image = np.fliplr(image)
        masks = np.fliplr(masks)
        edgewt = np.fliplr(edgewt)

    if (random.random() < bright_prob):  # brightness change
        value = randint(2, 15)
        image = image + value
        image[image > 255] = 255.
        image[image < 0] = 0.
        masks = masks
        edgewt = edgewt

    if (random.random() < gamma_prob):  # Gaussian noise
        im = image
        row, col = im.shape
        const = 2
        b = 2
        a = 0.1
        gamma = (1 / ((b - a) * random.random() + a))
        gamma_im = const * (im ** gamma)
        image = gamma_im
        masks = masks
        edgewt = edgewt

    image = IP.linstretch(image)

    return image, masks, edgewt


