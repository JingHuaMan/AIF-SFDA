import os

from PIL import Image

from data import BaseDataset
from data.base_dataset import get_transform


class ONLYIMAGEDataset(BaseDataset):
    """ A dataset class for image-only dataset.

        The file structure should be:
        - data_root
            - 0.png
            - 1.png
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

        original_path = os.path.join(self.opt.data_dirname, str(index) + '.png')
        original = Image.open(original_path).convert('RGB')

        raw_transform, _ = get_transform(self.opt)
        original = raw_transform(original)

        return {'image_original': original, 'source_path': original_path}

    def __len__(self):
        """ Return the total number of images in the dataset."""
        return self.len
