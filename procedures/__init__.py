import os

import torch
from lightning.fabric import Fabric
from future.moves import sys

from utils import logger, set_all_random_seed, lightning_fabric_utils


def preprocedure(opt):
    opt.is_train = opt.phase == 'train'  # train or val/test/others

    opt.display_env = opt.name

    logger.init_logger(os.path.join(opt.results_dirname, opt.name, opt.secondary_dirname, opt.phase, 'log.txt'))
    logger.debug('Command: python ' + ' '.join(sys.argv))

    if opt.use_lightning_fabric:
        lightning_fabric_utils.init(opt)

    if opt.phase == 'train' or opt.phase == 'val':
        assert len(opt.data_dirname) == 1
        opt.data_dirname = opt.data_dirname[0]

    if opt.phase == 'val':
        opt.load_epoch = None
        opt.load_path = None
        opt.metrics_as_sort_index = opt.metrics_as_sort_index if opt.metrics_as_sort_index is not None else \
            opt.metrics_list[0]
    elif opt.phase == 'test':
        opt.load_path = None
        opt.save_dataset_name = [dirname.split('/')[-2] for dirname in opt.data_dirname] if opt.save_dataset_name is None \
            else opt.save_dataset_name
        assert len(opt.save_dataset_name) == len(opt.data_dirname)

    if opt.random_seed is not None:
        set_all_random_seed(opt.random_seed)
        logger.info('Set random seed to:' + str(opt.random_seed))

    # set default gpu ids
    if all([i > 0 for i in opt.gpu_ids]):
        torch.cuda.set_device(opt.gpu_ids[0])  # not working actually
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in opt.gpu_ids])

    torch.autograd.set_detect_anomaly(opt.set_detect_anomaly)
