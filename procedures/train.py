import os
import time
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from procedures import preprocedure
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from utils import logger
from utils.visualizer import Visualizer

if __name__ == '__main__':
    # get and modify the options
    train_options = TrainOptions()
    opt = train_options.opt
    preprocedure(opt)
    train_options.print_options(opt)

    logger.info('Current training name: ' + opt.name)
    logger.info('Current training secondary name: ' + opt.secondary_dirname)

    # create dataset
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    logger.info('The number of training images = %d' % dataset_size)

    # create model
    model = create_model(opt)
    model.setup()

    # create visualizer for display loss and results
    visualizer = Visualizer(opt)
    total_iters = 0

    for epoch in range(opt.epochs_num):
        # timer for entire epoch
        epoch_start_time = time.time()
        # timer for data loading per iteration
        iter_data_time = time.time()
        # the number of training iterations in current epoch, reset to 0 every epoch
        epoch_iter = 0
        # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        visualizer.reset()

        for _ in range(opt.sample_repeat):
            # inner loop within one epoch
            for i, data in enumerate(dataset):
                # timer for computation per iteration
                iter_start_time = time.time()
                t_data = iter_start_time - iter_data_time if total_iters % opt.print_freq == 0 else 0

                total_iters += opt.batch_size
                epoch_iter += opt.batch_size

                # unpack data from dataset
                model.set_input(data)
                # calculate loss functions, get gradients, update network weights
                model.optimize_parameters()

                # display images on visdom and save images to an HTML file
                if total_iters % opt.display_freq == 0:
                    save_result = total_iters % opt.update_html_freq == 0
                    # in train mode, compute_visuals() will not be called automatically
                    model.compute_visuals()
                    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                # print training losses and save logging information to the disk
                if total_iters % opt.print_freq == 0:
                    losses = model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                    if opt.display_id > 0:
                        visualizer.plot_current_losses(epoch, float(epoch_iter) / (dataset_size * opt.sample_repeat),
                                                       losses)

                iter_data_time = time.time()

        # update learning rates at the end of every epoch
        model.update_learning_rate()

        # cache our model every <save_epoch_freq> epochs
        if epoch % opt.save_epoch_freq == 0:
            logger.info('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('last')
            model.save_networks(epoch)

        logger.info('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.epochs_num, time.time() - epoch_start_time))

    logger.info('Training process finishes')
    model.save_networks('last')
