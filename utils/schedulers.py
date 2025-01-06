import abc
import inspect
from functools import partial

import torch


def get_scheduler_cls(lr_scheduler):
    """ Return the scheduler class

    :param lr_scheduler: the name of the learning rate scheduler
    """
    if lr_scheduler == 'constant':
        return ConstantScheduler
    elif lr_scheduler == 'linear':
        return LinearScheduler
    else:
        return CommonScheduler


def get_scheduler(optimizer, opt):
    """ Return a learning rate scheduler, which is defined by <opt.lr_scheduler>.
        For 'constant', we keep the same learning rate for the entire training process.
        For 'linear', we keep the same learning rate for the first <opt.decay_epochs_num> epochs,
        and linearly decay the rate to zero over the next <opt.epochs_num - opt.decay_epochs_num> epochs.
        And other schedulers defined in torch.optim.lr_scheduler are also supported.

    :param optimizer: the optimizer of the network
    :param opt: stores all the experiment flags; needs to be a subclass of BaseOptions．　
    """
    scheduler_cls = get_scheduler_cls(opt.lr_scheduler)
    scheduler = scheduler_cls(opt, optimizer)
    return scheduler


def get_option_setter(scheduler_name):
    """ Return the static method <modify_commandline_options> of the scheduler class."""
    dataset_class = get_scheduler_cls(scheduler_name)
    return partial(dataset_class.modify_commandline_options, lr_scheduler=scheduler_name)


class BaseScheduler(abc.ABC):
    """ This class is an abstract base class (ABC) for schedulers.
        To create a subclass, you need to implement the following five functions:
            -- <__init__>:                      initialize the class; first call BaseScheduler.__init__(self, opt).
            -- <step>:                          update learning rate and update network weights.
            -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    @staticmethod
    def modify_commandline_options(parser, *args, **kwargs):
        """ Add model-specific options and set default options."""
        return parser

    def __init__(self, opt, optimizer):
        """ Initialize the class; save options in the class"""
        self.optimizer = optimizer
        self.opt = opt

    @abc.abstractmethod
    def step(self, *args, **kwargs):
        """ Update learning rate and update network weights."""
        pass


class ConstantScheduler(BaseScheduler):
    """ Actually this scheduler is not a scheduler, it just keeps the learning rate constant."""

    def step(self):
        pass


class LinearScheduler(BaseScheduler):
    """ In the first (epochs_num - decay_epochs_num) epochs, the lr is constant.
        In the last decay_epochs_num epochs, the lr is linearly decayed to 0.
    """

    @staticmethod
    def modify_commandline_options(parser, *args, **kwargs):
        parser.add_argument('--decay_epochs_num', type=int, required=True,
                            help='in the last decay_epochs_num epochs, the lr is linearly decayed to 0')
        return parser

    def __init__(self, opt, optimizer):
        super().__init__(opt, optimizer)
        # the call_times is assume as the current epoch number (just for convenience)
        self.called_times = 0

    def step(self):
        self.called_times += 1
        # if the current epoch number is larger than the decay_epochs_num, then decay the learning rate
        if self.called_times > self.opt.decay_epochs_num:
            # linearly decay the learning rate to 0
            lr = self.opt.optimizer_lr * (
                        self.opt.epochs_num - self.called_times + 1) / self.opt.decay_epochs_num
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # keep the learning rate
            pass


def get_scheduler_cls_by_name(name):
    for scheduler_name, scheduler_cls in inspect.getmembers(torch.optim.lr_scheduler, inspect.isclass):
        if scheduler_name.lower() == name.lower():
            return scheduler_cls
    raise NotImplementedError('Scheduler [%s] not recognized.' % name)


class CommonScheduler(BaseScheduler):
    """ This scheduler is a wrapper of common schedulers in torch.optim.lr_scheduler"""

    @staticmethod
    def modify_commandline_options(parser, *args, **kwargs):
        scheduler_cls = get_scheduler_cls_by_name(kwargs['lr_scheduler'])
        for arg_name, arg_parameter in inspect.signature(scheduler_cls.__init__).parameters.items():
            if arg_name in ['self', 'optimizer']:
                continue
            if arg_parameter.default == inspect.Parameter.empty:
                parser.add_argument('--lr_scheduler_' + arg_name, required=True)
            else:
                parser.add_argument('--lr_scheduler_' + arg_name, default=arg_parameter.default)
        return parser

    def __init__(self, opt, optimizer):
        super(CommonScheduler, self).__init__(opt, optimizer)
        self.scheduler = get_scheduler_cls_by_name(opt.lr_scheduler)(optimizer,
                                                                     **{k.replace('lr_scheduler_', ''): v for k, v in
                                                                        vars(opt).items() if
                                                                        k.startswith('lr_scheduler_')})

    def step(self, *args, **kwargs):
        self.scheduler.step(*args, **kwargs)
