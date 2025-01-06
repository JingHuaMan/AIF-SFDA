import torchfcn
from torch import nn


class FCN8S(torchfcn.models.FCN8s):
    def __init__(self, in_channel=3, out_channel=1):
        super(FCN8S, self).__init__(n_class=out_channel)