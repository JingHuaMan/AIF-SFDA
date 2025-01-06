"""This package includes a miscellaneous collection of useful helper functions."""

from __future__ import print_function

import importlib
import random
import subprocess

import cv2
import torch
import numpy as np
from PIL import Image

from .logger import init_logger


def tensor2im(input_image, image_type=np.uint8):
    """ Converts a Tensor array into a numpy image array.
        The value range should be [0, 1].

    :param input_image: the input image tensor array
    :param image_type: the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0  # post-processing: transpose
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(image_type)


def multi_class_segmentation_reduce_dim(segmentation_prediction):
    """ Prepare for the visualization of the segmentation prediction of multiple classes, or used as pseudo label.
        Transform a (B, C, H, W) tensor into a (B, 1, H, W) tensor with a value range of [0, C-1] and dtype of long.

    :param segmentation_prediction: the segmentation prediction tensor, shape (B, C, H, W), no softmax or sigmoid is needed.
    """
    channel_size = segmentation_prediction.shape[1]
    if channel_size == 1:
        segmentation_prediction = torch.sigmoid(segmentation_prediction)
        segmentation_prediction = torch.concat([1 - segmentation_prediction, segmentation_prediction], dim=1)
    else:
        segmentation_prediction = torch.softmax(segmentation_prediction, dim=1)
    return torch.max(segmentation_prediction, dim=1, keepdim=True)


def color_multi_class_label(label, class_num, color_map=cv2.COLORMAP_JET):
    """ Colorize the multi-class label tensor.

    :param label: the label tensor, shape (B, 1, H, W).
    :param class_num: the number of classes.
    :param color_map: the color map for the visualization (cv2 is used. Example: cv2.COLORMAP_JET).
    """
    if class_num == 1:
        return label
    if isinstance(label, torch.Tensor):
        label = label.detach().cpu().numpy()
    visualization = (label.astype(np.float32) * 255 / (class_num - 1)).astype(np.uint8)
    visualization = np.transpose(visualization, (0, 2, 3, 1))
    visualization = cv2.applyColorMap(visualization, color_map)
    visualization = np.transpose(visualization, (0, 3, 1, 2))
    return torch.from_numpy(visualization)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """ Save a numpy image to the disk

    :param image_numpy: input numpy array
    :param image_path: the path of the image
    :param aspect_ratio: aspect ratio of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_stat(var_to_print, var_name='', print_to_log=True, val=True, shp=False):
    """ Print the mean, min, max, median, std, and size of a torch array

    :param var_to_print: the variable to print the statistics
    :param var_name: the name of the variable
    :param print_to_log: if print the statistics to the logger
    :param val: if print the values of the torch array
    :param shp: if print the shape of the torch array
    """
    stat_info = '' if var_name == '' else '%s: ' % var_name
    if shp:
        stat_info += 'shape = %s, ' % str(var_to_print.shape)
    if val:
        stat_info += 'min = %.3f, max = %.3f' % (var_to_print.min(), var_to_print.max())
    if print_to_log:
        logger.info(stat_info)
    else:
        print(stat_info)


def get_class_from_subclasses(cls, subclass_name, allow_case=True, allow_underline=False):
    """ Get a class from its name.

    :param cls: the base class.
    :param subclass_name: the name of the subclass.
    :param allow_case: if True, the queried subclass name can contain case difference.
    :param allow_underline: if True, the queried subclass name can contain underline.
    :return: the subclass.
    """
    if allow_case:
        subclass_name = subclass_name.lower()
    if allow_underline:
        subclass_name = subclass_name.replace('_', '')

    for subclass in cls.__subclasses__():
        temp_subclass_name = subclass.__name__
        if allow_case:
            temp_subclass_name = temp_subclass_name.lower()
        if allow_underline:
            temp_subclass_name = temp_subclass_name.replace('_', '')
        if temp_subclass_name == subclass_name:
            return subclass
    raise NotImplementedError('Subclass not found: ' + subclass_name)


def import_class_from_module(module_name, cls_name_to_find, allow_case=True, allow_underline=False):
    """ Import a class from a module.

        :param module_name: the name of the module. e.g. 'models.base_model'
        :param cls_name_to_find: the name of the class to find.
        :param allow_case: if True, the queried class name can contain case difference.
        :param allow_underline: if True, the queried class name can contain underline.
    """
    if allow_case:
        cls_name_to_find = cls_name_to_find.lower()
    if allow_underline:
        cls_name_to_find = cls_name_to_find.replace('_', '')

    for cls_name, cls in importlib.import_module(module_name).__dict__.items():
        if allow_case:
            cls_name = cls_name.lower()
        if allow_underline:
            cls_name = cls_name.replace('_', '')
        if cls_name == cls_name_to_find:
            return cls
    else:
        raise NotImplementedError('Class not found: ' + cls_name_to_find)


def find_freest_gpus(gpu_num=1):
    """ Find the freest gpu(s) in the system.

    :param gpu_num: The number of gpus to find.
    :return: A list of gpu ids.
    """
    try:
        # query memory usage for all GPUs
        memory_free_info = subprocess.check_output(
            'nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader',
            shell=True, encoding='utf-8')
        memory_free_values = [(gpu_index, int(free_mem)) for gpu_index, free_mem in
                              enumerate(memory_free_info.strip().split('\n'))]
        # sort from lowest to highest memory free
        memory_free_values = sorted(memory_free_values, key=lambda x: -x[1])
        # select the indices of the lowest n
        return [gpu_index for gpu_index, _ in memory_free_values[:gpu_num]]
    except Exception as e:
        logger.error('Failed to find free GPUs:\r\n' + str(e))
        exit(0)


def set_all_random_seed(seed):
    """ Set the random seed for all the random number generators.

    :param seed: the random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Remove randomness (maybe slower on Tesla GPUs)
    # https://pytorch.org/docs/stable/notes/randomness.html
    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
