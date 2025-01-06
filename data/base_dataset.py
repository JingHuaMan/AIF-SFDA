""" This module implements an abstract base class (ABC) 'BaseDataset' for datasets.
    It also includes common transformation functions, which can be later used in subclasses.
"""
import random
from collections.abc import Iterable

import numpy as np
import torch.utils.data as data
from PIL import Image, ImageFilter
import torchvision.transforms as transforms
from abc import ABC, abstractmethod

from utils import logger


class BaseDataset(data.Dataset, ABC):
    """ This class is an abstract base class (ABC) for datasets.

        To create a subclass, you need to implement the following four functions:
        -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
        -- <__len__>:                       return the size of dataset.
        -- <__getitem__>:                   get a data point.
        -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """ Initialize the class; save the options in the class

        :param opt: stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.data_dirname

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """ Add new dataset-specific options, and rewrite default values for existing options.

        :param parser: the option parser
        :param is_train: whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        :return parser: the modified parser
        """
        return parser

    @abstractmethod
    def __len__(self):
        """ Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """ Return a data point and its metadata information.

        :param index: a random integer for data indexing
        :return a dictionary of data with their names. It usually contains the data itself and its metadata information.
        """
        pass


def get_size_list(raw_size_list):
    """ Get the size list from the raw size list. Used in resize and crop transformations.
        Support both fix size and random size.
    """
    if len(raw_size_list) == 1:
        # operation_256, means do this operation to size (256, 256)
        return [int(raw_size_list[0])] * 2
    elif len(raw_size_list) == 2:
        # operation_256_512, means do this operation to size (256, 512)
        return [int(raw_size_list[0]), int(raw_size_list[1])]
    elif len(raw_size_list) == 3:
        # operation_*_256_512, means the x size and y size are randomly chosen from [256, 512]
        lower_bound, upper_bound = int(raw_size_list[1]), int(raw_size_list[2])
        return [random.randint(lower_bound, upper_bound), random.randint(lower_bound, upper_bound)]
    elif len(raw_size_list) == 4:
        if raw_size_list[0] == '*':
            # operation_*_256_512_256, means the x size is randomly chosen from [256, 512] and y size is 256
            return [random.randint(int(raw_size_list[1]), int(raw_size_list[2])), int(raw_size_list[3])]
        else:
            # operation_256_*_256_512, means the y size is randomly chosen from [256, 512] and x size is 256
            return [int(raw_size_list[0]), random.randint(int(raw_size_list[2]), int(raw_size_list[3]))]
    elif len(raw_size_list) == 6:
        # operation_*_128_256_*_256_512, means the x size is randomly chosen from [128, 256] and y size is randomly
        # chosen from [256, 512]
        return [random.randint(int(raw_size_list[1]), int(raw_size_list[2])),
                random.randint(int(raw_size_list[4]), int(raw_size_list[5]))]
    else:
        raise ValueError(f'the resize size {raw_size_list} is not valid')


def get_transform(opt):
    """ Get two different but **parallel** sequences of transformations for image and label.
        image_transform is applied to the images that need color-dependent transformations (and do not require accuracy)
        label_transform is applied to the images that do not need color-dependent transformations (and require accuracy)

        :param opt: stores all the experiment flags; needs to be a subclass of BaseOptions
    """
    if not isinstance(opt.load_size, Iterable):
        opt.load_size = (opt.load_size, opt.load_size)

    image_transform_list = []
    label_transform_list = []

    # the current_size is the size of the image before any transformation, and may be updated after each transformation
    if len(opt.load_size) == 1:
        current_size = opt.load_size * 2
    elif len(opt.load_size) == 2:
        current_size = opt.load_size
    else:
        raise ValueError(f'the load size {opt.load_size} is not valid')

    for preprocess_item in opt.preprocess:
        if preprocess_item[:6] == 'resize':
            _, *raw_resize_size_list = preprocess_item.split('_')
            resize_size_list = get_size_list(raw_resize_size_list)
            if resize_size_list[0] <= 0 or resize_size_list[1] <= 0:
                raise ValueError(f'the resize size {resize_size_list} is not valid')
            current_size = resize_size_list
            image_transform_list.append(transforms.Resize(resize_size_list, Image.Resampling.BICUBIC))
            label_transform_list.append(transforms.Resize(resize_size_list, Image.Resampling.NEAREST))
        elif preprocess_item[:4] == 'crop':
            _, *raw_crop_size_list = preprocess_item.split('_')
            crop_size_list = get_size_list(raw_crop_size_list)
            if crop_size_list[0] <= 0 or crop_size_list[1] <= 0 or crop_size_list[0] > current_size[0] or \
                    crop_size_list[1] > current_size[1]:
                raise ValueError(f'the crop size {crop_size_list} is not valid')
            random_crop_x = random.randint(0, np.maximum(0, current_size[0] - crop_size_list[0]))
            random_crop_y = random.randint(0, np.maximum(0, current_size[1] - crop_size_list[1]))
            image_transform_list.append(
                transforms.Lambda(lambda img: __crop(img, (random_crop_x, random_crop_y), crop_size_list)))
            label_transform_list.append(
                transforms.Lambda(lambda img: __crop(img, (random_crop_x, random_crop_y), crop_size_list)))
        elif preprocess_item == 'horizontal_flip':
            do_flip_horizontal = bool(random.getrandbits(1))
            image_transform_list.append(transforms.Lambda(lambda img: __horizontal_flip(img, do_flip_horizontal)))
            label_transform_list.append(transforms.Lambda(lambda img: __horizontal_flip(img, do_flip_horizontal)))
        elif preprocess_item == 'vertical_flip':
            do_flip_vertical = bool(random.getrandbits(1))
            image_transform_list.append(transforms.Lambda(lambda img: __vertical_flip(img, do_flip_vertical)))
            label_transform_list.append(transforms.Lambda(lambda img: __vertical_flip(img, do_flip_vertical)))
        elif preprocess_item == 'rotate':
            random_angle = random.randrange(-180, 180)
            image_transform_list.append(
                transforms.Lambda(lambda img: __rotate(img, random_angle, Image.Resampling.BICUBIC)))
            label_transform_list.append(
                transforms.Lambda(lambda img: __rotate(img, random_angle, Image.Resampling.NEAREST)))
        elif preprocess_item == 'grayscale':
            image_transform_list.append(transforms.Grayscale())
            label_transform_list.append(transforms.Grayscale())
        elif preprocess_item[:13] == 'gaussian_blur':
            if len(preprocess_item.split('_')) == 2:
                preprocess_item += '_2'
            blur_radius = preprocess_item.split('_')[-1]
            blur_radius = int(blur_radius)
            image_transform_list.append(transforms.Lambda(lambda img: __gaussian_blur(img, blur_radius)))
        elif preprocess_item[:7] == 'unsharp':
            _, *unsharp_config = preprocess_item.split('_')
            unsharp_radius, unsharp_percent, unsharp_threshold = 2, 150, 3
            if len(unsharp_config) >= 1: unsharp_radius = int(unsharp_config[0])
            if len(unsharp_config) >= 2: unsharp_percent = int(unsharp_config[1])
            if len(unsharp_config) == 3: unsharp_threshold = int(unsharp_config[2])
            image_transform_list.append(
                transforms.Lambda(lambda img: __unsharp(img, unsharp_radius, unsharp_percent, unsharp_threshold)))
        elif preprocess_item[:9] == 'normalize':
            _, *normalize_config = preprocess_item.split('_')
            normalize_alpha, normalize_beta = 1.0, 0.0
            if len(normalize_config) >= 1: normalize_alpha = float(normalize_config[0])
            if len(normalize_config) == 2: normalize_beta = float(normalize_config[1])
            image_transform_list.append(transforms.Lambda(lambda img: __normalize(img, normalize_alpha, normalize_beta)))
        elif preprocess_item[:14] == 'gaussian_noise':
            _, _, *gaussian_noise_config = preprocess_item.split('_')
            gaussian_noise_mean, gaussian_noise_variance, do_grayscale = 0, 0.01, False
            if len(gaussian_noise_config) >= 1: gaussian_noise_mean = float(gaussian_noise_config[0])
            if len(gaussian_noise_config) >= 2: gaussian_noise_variance = float(gaussian_noise_config[1])
            if len(gaussian_noise_config) == 3: do_grayscale = bool(gaussian_noise_config[2])
            image_transform_list.append(
                transforms.Lambda(lambda img: __gaussian_noise(img, gaussian_noise_mean, gaussian_noise_variance, do_grayscale)))
        elif preprocess_item[:21] == 'salt_and_pepper_noise':
            _, _, _, _, *salt_and_pepper_noise_config = preprocess_item.split('_')
            salt_prob, pepper_prob, do_grayscale = 0.01, 0.01, False
            if len(salt_and_pepper_noise_config) >= 1: salt_prob = float(salt_and_pepper_noise_config[0])
            if len(salt_and_pepper_noise_config) >= 2: pepper_prob = float(salt_and_pepper_noise_config[1])
            if len(salt_and_pepper_noise_config) == 3: do_grayscale = bool(salt_and_pepper_noise_config[2])
            image_transform_list.append(
                transforms.Lambda(lambda img: __salt_and_pepper_noise(img, salt_prob, pepper_prob, do_grayscale)))
        elif preprocess_item[:13] == 'speckle_noise':
            _, _, *speckle_noise_config = preprocess_item.split('_')
            speckle_variance, do_grayscale = 0.04, False
            if len(speckle_noise_config) >= 1: speckle_variance = float(speckle_noise_config[0])
            if len(speckle_noise_config) == 2: do_grayscale = bool(speckle_noise_config[1])
            image_transform_list.append(
                transforms.Lambda(lambda img: __speckle_noise(img, speckle_variance, do_grayscale)))
        else:
            raise ValueError(f'preprocess {preprocess_item} is not implement')

        if preprocess_item.startswith('make_multiple_'):
            # get the final number
            multiplier = int(preprocess_item.split('_')[-1])
            image_transform_list.append(
                transforms.Lambda(lambda img: __make_multiple(img, multiplier, Image.Resampling.BICUBIC)))
            label_transform_list.append(
                transforms.Lambda(lambda img: __make_multiple(img, multiplier, Image.Resampling.NEAREST)))

    image_transform_list.append(transforms.ToTensor())
    label_transform_list.append(transforms.ToTensor())

    return transforms.Compose(image_transform_list), transforms.Compose(label_transform_list)


def __make_multiple(img, base, interpolation):
    ow, oh = img.size
    w = int(round(ow / base) * base)
    h = int(round(oh / base) * base)
    if w == ow and h == oh:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), interpolation)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    if isinstance(size, int):
        tw = th = size
    else:
        tw, th = size
    if ow > tw or oh > th:
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __horizontal_flip(img, flip):
    if flip:
        if isinstance(img, np.ndarray):
            return np.flip(img, axis=1).copy()
        return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return img


def __vertical_flip(img, flip):
    if flip:
        if isinstance(img, np.ndarray):
            return np.flip(img, axis=0).copy()
        return img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    return img


def __rotate(img, angle, interpolation):
    return img.rotate(angle, resample=interpolation)


def __gaussian_blur(img, radius):
    return img.filter(ImageFilter.GaussianBlur(radius))


def __unsharp(img, radius, percent, threshold):
    return img.filter(ImageFilter.UnsharpMask(radius, percent, threshold))


def __normalize(img, alpha, beta):
    return Image.fromarray(np.clip(np.array(img) * alpha + beta, 0, 255).astype(np.uint8))


def __gaussian_noise(image, mean=0, variance=0.01, grayscale=False):
    np_image = np.array(image).astype(np.float32) / 255.0
    if grayscale:
        np_image = np_image[:, :, 0]
        gauss = np.random.normal(mean, variance ** 0.5, np_image.shape)
        noisy_image = np.clip(np_image + gauss, 0, 1)
        return Image.fromarray((noisy_image * 255).astype(np.uint8)[:, :, np.newaxis].repeat(3, axis=2))
    else:
        gauss = np.random.normal(mean, variance ** 0.5, np_image.shape)
        noisy_image = np.clip(np_image + gauss, 0, 1)
        return Image.fromarray((noisy_image * 255).astype(np.uint8))


def __salt_and_pepper_noise(image,  salt_prob=0.01, pepper_prob=0.01, grayscale=False):
    np_image = np.array(image)
    if grayscale:
        np_image = np_image[:, :, 0]
        salt_mask = np.random.choice([0, 1], size=np_image.shape, p=[1 - salt_prob, salt_prob])
        pepper_mask = np.random.choice([0, 1], size=np_image.shape, p=[1 - pepper_prob, pepper_prob])
        np_image[salt_mask == 1] = 255
        np_image[pepper_mask == 1] = 0
        return Image.fromarray(np_image[:, :, np.newaxis].repeat(3, axis=2))
    else:
        salt_mask = np.random.choice([0, 1], size=np_image.shape, p=[1 - salt_prob, salt_prob])
        pepper_mask = np.random.choice([0, 1], size=np_image.shape, p=[1 - pepper_prob, pepper_prob])
        np_image[salt_mask == 1] = 255
        np_image[pepper_mask == 1] = 0
        return Image.fromarray(np_image)


def __speckle_noise(image, variance=0.04, grayscale=False):
    np_image = np.array(image).astype(np.float32) / 255.0
    if grayscale:
        np_image = np_image[:, :, 0]
        noise = np.random.normal(0, variance ** 0.5, np_image.shape)
        noisy_image = np.clip(np_image + np_image * noise, 0, 1)
        return Image.fromarray((noisy_image * 255).astype(np.uint8)[:, :, np.newaxis].repeat(3, axis=2))
    else:
        noise = np.random.normal(0, variance ** 0.5, np_image.shape)
        noisy_image = np.clip(np_image + np_image * noise, 0, 1)
        return Image.fromarray((noisy_image * 255).astype(np.uint8))


def __print_size_warning(ow, oh, w, h):
    """ Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        logger.info("The image size needs to be a multiple of 4. "
                    "The loaded image size was (%d, %d), so it was adjusted to "
                    "(%d, %d). This adjustment will be done to all images "
                    "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
