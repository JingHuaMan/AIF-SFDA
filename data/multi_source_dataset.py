import os.path
import random

from data.base_dataset import BaseDataset, get_transform
from PIL import Image


class MULTISOURCEDataset(BaseDataset):
    """
    A main dataset and auxiliary datasets
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--additional_source_list', type=str)
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.data_root = opt.data_dirname
        self.additional_root_list = opt.additional_source_list.split(',')

        for root in self.additional_root_list:
            assert (os.path.exists(root))
        assert (os.path.exists(self.data_root))

        self.opt = opt
        self.len = len(os.listdir(self.data_root))
        self.additional_len_list = [len(os.listdir(root)) for root in self.additional_root_list]

    def get_one_dataset(self, dataset_id=-1, index=-1, postfix=None):
        current_dataset_root = self.data_root if dataset_id == -1 else self.additional_root_list[dataset_id]
        current_dataset_len = self.len if dataset_id == -1 else self.additional_len_list[dataset_id]

        # index = -1 means random index
        if index == -1:
            index = random.randint(0, current_dataset_len - 1)

        image_path = os.path.join(current_dataset_root, str(index), 'image.png')
        label_path = os.path.join(current_dataset_root, str(index), 'label.png')
        mask_path = os.path.join(current_dataset_root, str(index), 'mask.png')

        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        raw_transform, label_transform = get_transform(self.opt)

        image = raw_transform(image)
        mask = label_transform(mask)
        label = label_transform(label)

        return_dict = {'image_original' + ('' if postfix is None else '_' + postfix): image,
                       'mask' + ('' if postfix is None else '_' + postfix): mask,
                       'label' + ('' if postfix is None else '_' + postfix): label,
                       'source_path' + ('' if postfix is None else '_' + postfix): image_path}

        return return_dict

    def __getitem__(self, index):
        return_dict = self.get_one_dataset(dataset_id=-1, index=index, postfix=None)
        for i in range(len(self.additional_root_list)):
            return_dict.update(self.get_one_dataset(dataset_id=i, index=-1, postfix='aux_' + str(i)))

        return return_dict

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.len
