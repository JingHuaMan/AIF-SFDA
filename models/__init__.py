from collections import OrderedDict

import torch
import torch.nn as nn
import functools
from torch.nn import init

from models.base_model import BaseModel
from utils import logger, import_class_from_module


def get_option_setter(model_name):
    """ Return the static method <modify_commandline_options> of the model class."""
    model_cls = import_class_from_module('models.' + model_name + '_model', model_name + 'model', allow_case=True,
                                         allow_underline=True)
    assert issubclass(model_cls, BaseModel)
    return model_cls.modify_commandline_options


def create_model(opt):
    """ Create a model given the option.

        This function warps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'validate.py'
    """
    model_cls = import_class_from_module('models.' + opt.model_name + '_model', opt.model_name + 'model',
                                         allow_case=True,
                                         allow_underline=True)
    assert issubclass(model_cls, BaseModel)
    instance = model_cls(opt)
    logger.info('model [%s] was created' % type(instance).__name__)
    return instance


def get_norm_layer(norm_type='instance'):
    """ Return a normalization layer

        For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
        For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.

    :param norm_type: the name of the normalization layer: batch | instance | none
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(_):
            return nn.Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """ Initialize network weights.

    :param net: network to be initialized
    :param init_type: the name of an initialization method: normal | xavier | kaiming | orthogonal
    :param init_gain: scaling factor for normal, xavier and orthogonal.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    logger.debug('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=()):
    """ Initialize a network:
        1. register CPU/GPU device (with multi-GPU support);
        2. initialize the network weights

    :param net: the network to be initialized
    :param init_type: the name of an initialization method: normal | xavier | kaiming | orthogonal
    :param init_gain: scaling factor for normal, xavier and orthogonal.
    :param gpu_ids: which GPUs the network runs on: e.g., 0,1,2

    :return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def freeze_net(net):
    """ Freeze a network. """
    for param in net.parameters():
        param.requires_grad = False
    return net


def unfreeze_net(net):
    """ Unfreeze a network. """
    for param in net.parameters():
        param.requires_grad = True
    return net


@torch.no_grad()
def ema_update(source_model, target_model, smooth_factor=0.996):
    """ Exponential Moving Average (EMA) update."""
    source_dict = source_model.state_dict()

    new_target_dict = OrderedDict()
    for key, value in target_model.state_dict().items():
        if key in source_dict.keys():
            new_target_dict[key] = source_dict[key] * (1 - smooth_factor) + value * smooth_factor
        else:
            logger.error('During the EMA process, {} is not found in source model'.format(key))

    target_model.load_state_dict(new_target_dict)
    return target_model
