import os
import pickle
import random

import numpy as np
import torch
from PIL import Image

from data import BaseDataset
from data.base_dataset import get_transform


class FixPointPromptDataset(BaseDataset):
    """ A dataset class for labeled image dataset, including fixed SAM point prompt
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_mask', action='store_true', help='whether the dataset has mask')
        return parser

    def __init__(self, opt):
        """ Initialize this dataset class.

        :param opt: stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        self.opt = opt
        self.len = len(os.listdir(self.opt.data_dirname))

    def __getitem__(self, index):
        """ Return a data dict and its metadata information.

        :param index: an integer for data indexing
        :return a dictionary of data with their names. It usually contains the data itself and its metadata information.
        """

        original_path = os.path.join(self.opt.data_dirname, str(index), 'image.png')
        label_path = os.path.join(self.opt.data_dirname, str(index), 'label.png')
        mask_path = os.path.join(self.opt.data_dirname, str(index), 'mask.png')
        prompts_path = os.path.join(self.opt.data_dirname, str(index), 'prompts.pickle')

        original = Image.open(original_path).convert('RGB')
        label = Image.open(label_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        raw_transform, label_transform = get_transform(self.opt)

        original = raw_transform(original)
        label = label_transform(label)
        mask = label_transform(mask)

        with open(prompts_path, 'rb') as f:
            original_size, prompt_coordinates, prompt_labels = pickle.load(f)
        current_size = label.shape[-1]
        prompt_coordinates = (prompt_coordinates * current_size / original_size).to(dtype=prompt_coordinates.dtype)

        return {'image_original': original, 'mask': mask, 'label': label,
                'prompt': (prompt_coordinates, prompt_labels), 'source_path': original_path}

    def __len__(self):
        """ Return the total number of images in the dataset."""
        return self.len
