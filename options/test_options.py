from future.moves import sys

from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        # define shared options
        parser = BaseOptions.initialize(self, parser)

        # set default parameters for test phase
        parser.set_defaults(phase='test', load_size=parser.get_default('crop_size'), num_threads=0, batch_size=1,
                            serial_batches=True, display_id=-1, load_epoch='best')

        parser.add_argument('--num_test', type=int, default=sys.maxsize, help='only test the first #num_test images, default: all')
        # Dropout and Batch norm has different behavior during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--save_dataset_name', type=str, nargs='*', help='save the results with this name, used in test phase')

        # metrics
        parser.add_argument('--metrics_as_sort_index', type=str, default=None, help='sort the images by this metric, for example: dice (should be in metrics list)')
        parser.add_argument('--metrics_list', type=str, nargs='+', default=['dice', 'iou'], help='metrics to be calculated')
        parser.add_argument('--metrics_threshold', type=float, default=0.5)
        parser.add_argument('--metrics_calculate_std_var', action='store_true', help='calculate the standard variance of the metrics')

        # used when saving results for visualization
        parser.add_argument('--save_variable_names', type=str, default='', help='save the images with this variable')
        parser.add_argument('--save_file_names', type=str, default='', help='save the images with this variable')
        parser.add_argument('--save_dir', type=str, default='', help='save the images with this variable')

        # used in validation mode
        parser.add_argument('--val_remove_all_but_the_best', action='store_true', help='when phase=val, remove all the checkpoints except the best one')

        # self.isTrain = False
        return parser
