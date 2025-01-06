from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.set_defaults(phase='train')

        # training parameters
        parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type')
        parser.add_argument('--lr_scheduler', type=str, default='linear', help='learning rate update policy. [linear | step | plateau | cosine]')
        parser.add_argument('--epochs_num', type=int, default=100, help='number of total epochs to run')

        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--sample_repeat', type=int, default=1, help='repeat the same sample for n times')

        # visdom and HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=40, help='frequency of showing training results on screen')
        parser.add_argument('--display_column_num', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default=None, help='visdom display environment name (default is opt.name)')
        parser.add_argument('--display_port', type=int, default=19191, help='visdom port of the web display')
        parser.add_argument('--display_window_size', type=int, default=256, help='display window size for both visdom and HTML')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=10, help='frequency of showing training results on console')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')

        return parser
