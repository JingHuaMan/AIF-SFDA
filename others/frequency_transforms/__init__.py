import abc

import torch

from utils import import_class_from_module


class FrequencyTransformPrototype(abc.ABC):
    """ Prototype for frequency transforms, used to define the interface of the transforms.
        Inputs of both function() and inverse_function() should be torch.tensor with shape (b, c, w, h),
        and both procedures should be differentiable in torch backward.
    """
    @abc.abstractmethod
    def function(self, img):
        """ Transform """
        pass

    @abc.abstractmethod
    def inverse_function(self, frequency_map):
        """ Inverse transform """
        pass

    def __call__(self, img, inverse=False):
        """ Apply the transform to an image """
        return self.function(img) if not inverse else self.inverse_function(img)

    def normalize_frequency_map(self, frequency_map, visual=False):
        """ Normalize frequency map to [0, 1].
            Regardless of information loss, mainly used for visualization or DL inputs.
        """
        return torch.sigmoid(frequency_map)


def get_frequency_transform(transform_name):
    """ Get the frequency transform class by its name """
    return import_class_from_module('others.frequency_transforms.' + transform_name, transform_name, allow_case=True)
