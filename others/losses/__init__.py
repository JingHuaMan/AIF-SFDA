import abc

from torch import nn


class CustomLoss(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, *input_val):
        pass
