import csv
import os
import sys

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

        # create a website
        web_dir = os.path.join(opt.results_dirname, opt.name, opt.secondary_dirname, opt.phase, 'web', opt.save_dataset_name)
        logger.info('creating web directory')
        webpage = utils.visualizer.WebPageGenerator(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.load_epoch))

        # test with eval mode. This affects layers like bn and dropout
        if opt.eval:
            model.eval()

        for j, data in enumerate(dataset):
            # only apply our model to opt.num_test images.
            if j >= opt.num_test:
                break

            # unpack data from data loader
            model.set_input(data)

            # run inference
            model.test()

            # get image results
            visuals = model.get_current_visuals()

            # get image paths
            img_path = model.get_image_paths()
            if j % 5 == 0:
                logger.info('processing (%04d)-th image... %s' % (j, img_path))

            # save images to an HTML file
            for keys, image in visuals.items():
                image_numpy = utils.tensor2im(image)
                img_path = os.path.join(webpage.get_image_dir(), 'sample_%.3d_%s.png' % (j, keys))
                utils.save_image(image_numpy, img_path)
            webpage.add_images_in_a_line_with_header('sample_%.3d' % j, visuals.keys())

        # save the HTML
        webpage.save()

        # calculate metrics
        results = model.get_metric_results()
        logger.info('metrics results of dataset: ' + opt.save_dataset_name)
        logger.info('\t'.join(results.keys()))
        logger.info('\t'.join([str(i) for i in results.values()]))
        all_results_list += list(results.values())

        # save the metrics to files
        test_results_file_path = os.path.join(opt.results_dirname, opt.name, opt.secondary_dirname, 'test', 'metrics.csv')
        test_results_file_exists = os.path.exists(test_results_file_path)
        with open(test_results_file_path, 'a') as f:
            writer = csv.writer(f)
            if not test_results_file_exists:
                writer.writerow(['dataset'] + list(results.keys()))
            writer.writerow([opt.save_dataset_name] + list(results.values()))

    logger.info('metrics results of all datasets:')
    logger.info('\t'.join(save_dataset_name_list))
    logger.info('\t'.join([str(i) for i in all_results_list]))
