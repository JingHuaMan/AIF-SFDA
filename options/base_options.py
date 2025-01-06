import argparse

import yaml

import models
import data
import time

import utils
from utils import logger, schedulers, optimizers


class BaseOptions:
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        self.opt = self.gather_options()

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        #
        parser.add_argument('--config_file', type=str, default=None, help='path to a yaml configuration file, will overwrite all the default values (but not the command line arguments)')

        # basic parameters
        parser.add_argument('--data_dirname', type=str, nargs='+', help='path to images, should be organized like:\n-<data_dirname>\n\t- 0\n\t\t- image.png\n\t\t- label.png\n\t- 1\n\t\t- image.png\n\t\t- label.png\n\t- ...\n\t- n\n\t\t- image.png\n\t\t- label.png\n')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--secondary_dirname', type=str, default=time.strftime("%Y-%m-%d_%H:%M", time.localtime()), help='if not specified, the secondary directory name will be set to timestamp.')
        parser.add_argument('--gpu_ids', type=int, nargs='+', default=utils.find_freest_gpus(gpu_num=1), help='gpu ids, use -1 for CPU. if not specified, the freest gpu will be used.')
        parser.add_argument('--results_dirname', type=str, default='./results/', help='saves results here. Including model checkpoints and images')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--random_seed', type=int, default=None, help='random seed for training/testing code')
        parser.add_argument('--set_detect_anomaly', action='store_true', help='turn on set_detect_anomaly in pytorch')

        # lightning fabric for multi-gpu training
        parser.add_argument('--use_lightning_fabric', action='store_true', help='use lightning fabric to manage the experiment')
        parser.add_argument('--lightning_fabric_strategy', type=str, default='ddp', help='lightning fabric strategy [dp, ddp, ddp_spawn, xla, deepspeed, fsdp]')

        # model parameters
        parser.add_argument('--model_name', type=str, default='naive_baseline', help='chooses which model to use')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        parser.add_argument('--norm_type', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')

        # dataset parameters
        parser.add_argument('--dataset_name', type=str, default='naive', help='chooses which dataset are loaded.')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_workers', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--load_size', type=int, nargs='+', default=[512], help='the input size of the original image')
        parser.add_argument('--preprocess', type=str, nargs='*', default=[], help='basic augmentations. currently available preprocess methods: resize, crop, flip, rotate. to_tensor is always used.')
        parser.add_argument('--drop_last', action='store_true', help='if specified, drop the last incomplete batch, if the dataset size is not divisible by the batch size.')

        # load model parameters
        model_load_group = parser.add_mutually_exclusive_group()
        model_load_group.add_argument('--load_epoch', type=str, default=None, help='which epoch to load. set to last to use last cached model')
        model_load_group.add_argument('--load_path', type=str, nargs='*', default=None, help='the full path of the model to load. If specified, it will overwrite load_epoch.')

        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser)

        # modify model-related parser options
        opt, _ = parser.parse_known_args()
        model_option_setter = models.get_option_setter(opt.model_name)
        parser = model_option_setter(parser, opt.phase == 'train')

        # modify dataset-related parser options
        opt, _ = parser.parse_known_args()
        dataset_option_setter = data.get_option_setter(opt.dataset_name)
        parser = dataset_option_setter(parser, opt.phase == 'train')

        # TODO this part should not be here, but currently we have no better solution
        if opt.phase == 'train':
            # modify scheduler-related parser options
            opt, _ = parser.parse_known_args()
            scheduler_option_setter = schedulers.get_option_setter(opt.lr_scheduler)
            parser = scheduler_option_setter(parser)

            # modify optimizer-related parser options
            opt, _ = parser.parse_known_args()
            parser = optimizers.CommonOptimizer.modify_commandline_options(parser, opt.optimizer)

        # modify all the default values according to the config file
        if opt.config_file is not None:
            # parse the config file and overwrite the default values, while no utils is used
            with open(opt.config_file, 'r') as f:
                config = yaml.safe_load(f)
            for k, v in config.items():
                if k in vars(opt):
                    parser.set_defaults(**{k: v})
                else:
                    logger.error(f'config file has unknown option: {k}')

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """ Output options to logger
            It will output both current options and default values(if different).
        """
        logger.debug('----------------- Options ---------------')
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = f'\t[default: {default}]'
            logger.debug('{:>25}: {:<30}{}'.format(str(k), str(v), comment))
        logger.debug('----------------- End -------------------')
