""" This package includes all the modules related to data loading and preprocessing

To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_name dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import torch.utils.data

from data.base_dataset import BaseDataset
from utils import logger, get_class_from_subclasses, import_class_from_module, lightning_fabric_utils


def get_option_setter(dataset_name):
    """ Return the static method <modify_commandline_options> of the dataset class."""
    dataset_cls = import_class_from_module('data.' + dataset_name + '_dataset', dataset_name + 'dataset',
                                           allow_case=True, allow_underline=True)
    assert issubclass(dataset_cls, BaseDataset)
    return dataset_cls.modify_commandline_options


def create_dataset(opt):
    """ Create a dataset given the option.
        This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'validate.py'
    """
    return CustomDatasetDataLoader(opt)


class CustomDatasetDataLoader:
    """ Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt):
        """ Initialize this class
            Step 1: create a dataset instance given the name [dataset_mode]
            Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        dataset_cls = import_class_from_module('data.' + opt.dataset_name + '_dataset', opt.dataset_name + 'dataset',
                                               allow_case=True, allow_underline=True)
        assert issubclass(dataset_cls, BaseDataset)
        self.dataset = dataset_cls(opt)
        logger.info("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=opt.num_workers,
            drop_last=opt.drop_last)
        if opt.use_lightning_fabric:
            self.dataloader = lightning_fabric_utils.get_fabric().setup_dataloaders(self.dataloader)

    def __len__(self):
        """ Return the number of data in the dataset"""
        return len(self.dataset)

    def __iter__(self):
        """ Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            yield data
