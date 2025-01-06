import os
import random
from datetime import datetime

import numpy as np
import torch
from PIL import Image

from data import BaseDataset
from data.base_dataset import get_transform
from utils import set_all_random_seed


class RandomPointPromptDataset(BaseDataset):
    """ A dataset class for labeled image dataset, including SAM point prompt
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_mask', action='store_true', help='whether the dataset has mask')
        parser.add_argument('--positive_num', type=int, default=1000,
                            help='the ratio of prompt points within foreground pixels')
        parser.add_argument('--negative_num', type=int, default=1000,
                            help='the num of prompt points within background pixels')
        parser.add_argument('--dataset_random_seed', default=None, type=int, help='random seed start')
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

        original = Image.open(original_path).convert('RGB')
        label = Image.open(label_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        raw_transform, label_transform = get_transform(self.opt)

        original = raw_transform(original)
        label = label_transform(label)
        mask = label_transform(mask)

        if self.opt.dataset_random_seed is not None:
            set_all_random_seed(self.opt.dataset_random_seed + index)

        positive_coordinates = torch.flip(torch.nonzero(label[0]), dims=[1])
        # print(positive_coordinates.shape)
        positive_coordinates = positive_coordinates[
            torch.randperm(positive_coordinates.size(0))[:self.opt.positive_num]]
        negative_coordinates = torch.flip(torch.nonzero(label[0] == 0), dims=[1])
        negative_coordinates = negative_coordinates[
            torch.randperm(negative_coordinates.size(0))[:self.opt.negative_num]]
        prompt_coordinates = torch.cat([positive_coordinates, negative_coordinates], dim=0)
        prompt_labels = torch.cat([torch.ones(positive_coordinates.size(0)), torch.zeros(negative_coordinates.size(0))])

        if self.opt.dataset_random_seed is not None:
            set_all_random_seed(
                self.opt.random_seed if self.opt.random_seed is not None else int(datetime.now().timestamp()))

        return {'image_original': original, 'mask': mask, 'label': label,
                'prompt': (prompt_coordinates, prompt_labels), 'source_path': original_path}

    def __len__(self):
        """ Return the total number of images in the dataset."""
        return self.len
