import shutil
import sys
import os
import csv

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from data import create_dataset
from procedures import preprocedure
from utils import logger
from options.test_options import TestOptions
from models import create_model
from tqdm import tqdm

if __name__ == '__main__':
    # get val options
    test_options = TestOptions()
    opt = test_options.opt
    preprocedure(opt)
    test_options.print_options(opt)

    logger.info('Current training name: ' + opt.name)
    logger.info('Current training secondary name: ' + opt.secondary_dirname)

    # create a dataset
    dataset = create_dataset(opt)

    # create a model (and do not setup model here)
    model = create_model(opt)

    metrics_result_list = []

    epoch_list = []
    for pth_file_name in os.listdir(model.save_dir):
        epoch_name = pth_file_name.split('_')[0]
        if pth_file_name.endswith('.pth') and epoch_name not in epoch_list:
            epoch_list.append(epoch_name)

    for epoch in tqdm(epoch_list):
        # setup model with specified pth file
        # opt.load_path = [os.path.join(checkpoints_root, pth_file_name)]
        opt.load_epoch = epoch
        model.setup()

        # validation with eval mode. This affects layers like bn and dropout
        if opt.eval:
            model.eval()

        for i, data in enumerate(dataset):
            # only apply our model to opt.num_test images.
            if i >= opt.num_test:
                break

            # unpack data from data loader
            model.set_input(data)

            # run inference
            model.test()

            # get image results, and prepare for sorting
        metrics_result_list.append((epoch, model.get_metric_results()[opt.metrics_as_sort_index]))

    # get the best pth file
    metrics_result_list = sorted(metrics_result_list, key=lambda x: -x[1])
    for model_name in model.model_names:
        shutil.copy(os.path.join(model.save_dir, '%s_%s_%s_%s.pth' % (
            str(metrics_result_list[0][0]), model_name, opt.name, opt.secondary_dirname)),
                    os.path.join(model.save_dir, 'best_%s_%s_%s.pth' % (model_name, opt.name, opt.secondary_dirname)))

    if opt.val_remove_all_but_the_best:
        # remove all the checkpoints except the best one
        for pth_file_name, _ in metrics_result_list[1:]:
            if not pth_file_name.startswith('best'):
                os.remove(os.path.join(model.save_dir, pth_file_name))

        logger.info('Validation finished. Removed all the checkpoints except the best one')
    else:
        os.makedirs(os.path.join(opt.results_dirname, opt.name, opt.secondary_dirname, 'val'), exist_ok=True)
        with open(os.path.join(opt.results_dirname, opt.name, opt.secondary_dirname, 'val',
                               'validation_sorting_result.csv'), 'w') as result_file:
            csv_writer = csv.writer(result_file)
            csv_writer.writerow(['epoch', opt.metrics_as_sort_index])
            csv_writer.writerows(metrics_result_list)

        logger.info('Validation finished. The best epoch is {}'.format(metrics_result_list[0][0]))
