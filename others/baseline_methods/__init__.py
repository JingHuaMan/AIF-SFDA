import importlib
import inspect

import torch

from utils import logger


def find_model_using_name(model_name):
    """Import the module "models/[model_name]_model.py".

    In the file, the class called DatasetNameModel() will
    be instantiated. It has to be a subclass of BaseModel,
    and it is case-insensitive.
    """
    model_lib = importlib.import_module('others.baseline_methods.models.segmentation')

    model = None
    for name, cls in model_lib.__dict__.items():
        if inspect.isclass(cls) and name == model_name and issubclass(cls, torch.nn.Module):
            model = cls
            break

    if model is None:
        logger.error("There should be a subclass of BaseModel with class name that matches %s in lowercase." % (
            model_name))
        exit(0)

    return model
