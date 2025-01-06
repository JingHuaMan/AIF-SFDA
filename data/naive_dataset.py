import os

from PIL import Image

from data import BaseDataset
from data.base_dataset import get_transform


class NaiveDataset(BaseDataset):
    """ A dataset class for labeled image dataset.

        The file structure should be:
        - data_root
            - 0
                - image.png (original image)
                - label.png (ground truth)
                - mask.png (used to ignore unwanted pixels)
            - 1
            - 2
            ...
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

        original = Image.open(original_path).convert('RGB')
        label = Image.open(label_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        raw_transform, label_transform = get_transform(self.opt)

        original = raw_transform(original)
        label = label_transform(label)
        mask = label_transform(mask)

        return {'image_original': original, 'mask': mask, 'label': label, 'source_path': original_path}

    def __len__(self):
        """ Return the total number of images in the dataset."""
        return self.len
