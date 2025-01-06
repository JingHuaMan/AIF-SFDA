import csv
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import utils.visualizer
from procedures import preprocedure
from utils import logger

if __name__ == '__main__':
    # get test options
    test_options = TestOptions()
    opt = test_options.opt
    preprocedure(opt)
    test_options.print_options(opt)

    logger.info('Current testing name: ' + opt.name)
    logger.info('Current testing secondary name: ' + opt.secondary_dirname)

    # create a model
    model = create_model(opt)
    model.setup()

    data_dirname_list = opt.data_dirname
    save_dataset_name_list = opt.save_dataset_name

    all_results_list = []

    for i in range(len(data_dirname_list)):
        opt.data_dirname = data_dirname_list[i]
        opt.save_dataset_name = save_dataset_name_list[i]
        logger.info('processing dataset: ' + opt.save_dataset_name)

        # create a dataset
        dataset = create_dataset(opt)

        # test with eval mode. This affects layers like bn and dropout
        if opt.eval:
            model.eval()

        start_time = time.time()

        for j, data in enumerate(dataset):
            # only apply our model to opt.num_test images.
            if j >= opt.num_test:
                break

            # unpack data from data loader
            model.set_input(data)

            # run inference
            model.test()

        end_time = time.time()

        logger.info('Time consumption: %.2f s' % (end_time - start_time))
        logger.info('Time consumption per image: %.2f s' % ((end_time - start_time) / len(dataset)))
