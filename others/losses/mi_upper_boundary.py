import abc

import torch
import torch.nn as nn

from others.backbones.modules.common_blocks import MultiConv
from others.losses import CustomLoss
from utils import get_class_from_subclasses


class MIUpperBoundary(CustomLoss, abc.ABC):
    def __init__(self):
        super(MIUpperBoundary, self).__init__()

    @abc.abstractmethod
    def loglikelihood(self, x, y):
        pass

    @abc.abstractmethod
    def forward(self, x, y):
        pass

    @classmethod
    def get_mi_upper_boundary_by_name(cls, name):
        return get_class_from_subclasses(cls, name)


class CLUB(MIUpperBoundary):
    def __init__(self, input_nc):
        super(CLUB, self).__init__()
        self.mu_layer = MultiConv(input_nc, input_nc, kernel_size=3, conv_block_num=2, norm_func=nn.InstanceNorm2d,
                                  last_activation=False, do_residual=True)
        self.logvar_layer = MultiConv(input_nc, input_nc, kernel_size=3, conv_block_num=2,
                                      norm_func=nn.InstanceNorm2d, last_activation=False, do_residual=True)

    def loglikelihood(self, x, y):
        mu, logvar = self.mu_layer(x), self.logvar_layer(x)
        return (- (mu - y) ** 2 / 2. / logvar.exp()).mean()

    def forward(self, x, y):
        mu, logvar = self.mu_layer(x), self.logvar_layer(x)
        positive = - (mu - y) ** 2 / 2. / logvar.exp()
        random_index = torch.randperm(x.size(0))
        while (random_index == torch.arange(x.size(0))).all():
            random_index = torch.randperm(x.size(0))
        y_neg_samples = y[random_index]
        negative = - (mu - y_neg_samples) ** 2 / 2. / logvar.exp()
        return (positive.mean(dim=[1, 2, 3]) - negative.mean(dim=[1, 2, 3])).mean()
