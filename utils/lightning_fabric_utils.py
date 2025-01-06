import logging

from lightning import Fabric

_fabric = None


def init(opt):
    global _fabric
    _fabric = Fabric(devices=opt.gpu_ids, strategy=opt.lightning_fabric_strategy)
    _fabric.launch()
    logging.getLogger("lightning.fabric").setLevel(logging.ERROR)


def get_device():
    global _fabric
    assert _fabric
    return _fabric.device


def get_fabric():
    global _fabric
    assert _fabric
    return _fabric


def process_models_and_optimizers(model_list, optimizer_list):
    global _fabric
    assert _fabric
    assert len(model_list) == len(optimizer_list)