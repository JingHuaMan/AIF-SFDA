# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:segformer
    author: 12718
    time: 2022/4/29 16:11
    tool: PyCharm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from others.baseline_methods.models.classification.mixformer import *
from others.baseline_methods.models.classification.create_model import create_backbone

__all__ = ["segformer_b0", "segformer_b1", "segformer_b2", "segformer_b3", "segformer_b4", "segformer_b5", "SegFormer"]

# backbones = {
#     "mit_b0":mit_b0,
#     "mit_b1":mit_b1,
#     "mit_b2":mit_b2,
#     "mit_b3":mit_b3,
#     "mit_b4":mit_b4,
#     "mit_b5":mit_b5
# }

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim=256):
        super(MLP, self).__init__()
        self.proj = nn.Linear(dim, hidden_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class Decoder(nn.Module):
    def __init__(self, dims=[32, 64, 160, 256], embd_dim=256, drop_out=0.1):
        super(Decoder, self).__init__()
        self.linear1 = MLP(dims[0], embd_dim)
        self.linear2 = MLP(dims[1], embd_dim)
        self.linear3 = MLP(dims[2], embd_dim)
        self.linear4 = MLP(dims[3], embd_dim)

        self.linear_fuse = nn.Conv2d(embd_dim*4, embd_dim, 1, 1, 0)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, features):
        f1, f2, f3, f4 = features
        bs = f1.shape[0]
        _f1 = self.linear1(f1).transpose(1, 2).reshape(bs, -1, f1.shape[2], f1.shape[3])
        _f2 = self.linear2(f2).transpose(1, 2).reshape(bs, -1, f2.shape[2], f2.shape[3])
        _f3 = self.linear3(f3).transpose(1, 2).reshape(bs, -1, f3.shape[2], f3.shape[3])
        _f4 = self.linear4(f4).transpose(1, 2).reshape(bs, -1, f4.shape[2], f4.shape[3])
        fes = [_f1, ] + [F.interpolate(f, size=f1.shape[2:], mode="bilinear", align_corners=False) for f in [_f2, _f3, _f4]]
        fusion = torch.cat(fes, dim=1)
        fusion = self.linear_fuse(fusion)
        fusion = self.dropout(fusion)
        return fusion


class SegFormer(nn.Module):
    def __init__(self, img_size=512, arch="mit_b0", num_classes=19, dims=[32, 64, 160, 256], embd_dim=256, pretrained=False,
                 pretrained_weights=None, drop_rate=0.1, **kwargs):
        super(SegFormer, self).__init__()
        assert arch in ["mit_b0", "mit_b1", "mit_b2", "mit_b3", "mit_b4", "mit_b5"], "backbone arch must in " \
                                                                                     "['mit_b0', 'mit_b1', 'mit_b2', " \
                                                                                     "'mit_b3', 'mit_b4', 'mit_b5']"
        self.encoder = create_backbone(arch, img_size=img_size, **kwargs)
        if pretrained:
            assert pretrained_weights is not None
            state = torch.load(pretrained_weights, map_location="cpu")
            self.encoder.load_state_dict(state)
        del self.encoder.head

        self.decoder = Decoder(dims, embd_dim, drop_out=drop_rate)
        self.out_conv = nn.Conv2d(embd_dim, num_classes, 1, 1, 0)

    def forward(self, x):
        features = self.encoder.forward_features(x)
        decoder_feature = self.decoder(features)
        out = self.out_conv(decoder_feature)
        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
        return out


def segformer_b0(**kwargs):
    return SegFormer(dims=[32, 64, 160, 256], arch="mit_b0", embd_dim=256, drop_rate=0.1, **kwargs)


def segformer_b1(**kwargs):
    return SegFormer(dims=[64, 128, 320, 512], arch="mit_b1", embd_dim=256, drop_rate=0.1, **kwargs)

def segformer_b2(**kwargs):
    return SegFormer(dims=[64, 128, 320, 512], arch="mit_b2", embd_dim=768, drop_rate=0.1, **kwargs)

def segformer_b3(**kwargs):
    return SegFormer(dims=[64, 128, 320, 512], arch="mit_b3", embd_dim=768, drop_rate=0.1, **kwargs)

def segformer_b4(**kwargs):
    return SegFormer(dims=[64, 128, 320, 512], arch="mit_b4", embd_dim=768, drop_rate=0.1, **kwargs)

def segformer_b5(**kwargs):
    return SegFormer(dims=[64, 128, 320, 512], arch="mit_b5", embd_dim=768, drop_rate=0.1, **kwargs)

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    x = torch.randn(1, 3, 1024, 1024)
    model = segformer_b0(img_size=1024)
    out = model(x)
    print(out.shape)