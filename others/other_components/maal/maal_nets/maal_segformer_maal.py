# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from others.backbones.unet import Unet
from others.other_components.maal.maal_nets.maal_SeparableConv2d import DoubleConv

from others.other_components.maal.maal_nets.maal_backbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5


class MLP_Lin(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = self.proj(x)
        return x


class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        # self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.ln = nn.LayerNorm([c2, 128, 128])
        self.act = nn.LeakyReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.ln(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))




class PFE_module(nn.Module):
    def __init__(self, embedding_dim=768):
        super(PFE_module, self).__init__()
        self.mlp1 = MLP(input_dim=embedding_dim, embed_dim=embedding_dim)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.pool_mlp1 = MLP_Lin(input_dim=16384, embed_dim=16384 // 16)
        self.pool_relu = nn.LeakyReLU()
        self.pool_mlp2 = MLP_Lin(input_dim=16384 // 16, embed_dim=16384)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = self.mlp1(inputs)
        x_att = self.avg_pool(x)
        x_att = self.pool_mlp1(x_att.transpose(1, 2))
        x_att = self.pool_relu(x_att)
        x_att = self.pool_mlp2(x_att).transpose(1, 2)
        x_att = self.sigmoid(x_att)
        x_weighted = x * x_att.expand_as(x)
        out = x_weighted.permute(0, 2, 1).reshape(inputs.shape[0], x.shape[2], inputs.shape[2], -1)
        return out


class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
        super(SegFormerHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            c1=embedding_dim * 4,
            c2=embedding_dim,
            k=1,
        )

        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio)

        self.linear_pred_s1 = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.linear_pred_s2 = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.linear_pred_s3 = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.linear_pred_s4 = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

        self.feature_Attention_c = PFE_module()
        self.Att_c1 = MLP(input_dim=embedding_dim, embed_dim=embedding_dim)
        self.Att_c2 = MLP(input_dim=embedding_dim, embed_dim=embedding_dim)
        self.Att_c3 = MLP(input_dim=embedding_dim, embed_dim=embedding_dim)
        self.Att_c4 = MLP(input_dim=embedding_dim, embed_dim=embedding_dim)

        # self.ln1 = nn.LayerNorm([embedding_dim, 128, 128])
        # self.ln2 = nn.LayerNorm([embedding_dim, 128, 128])
        # self.ln3 = nn.LayerNorm([embedding_dim, 128, 128])
        # self.ln4 = nn.LayerNorm([embedding_dim, 128, 128])

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        pfe_c1 = self.feature_Attention_c(_c4)

        _c4 = _c4 + pfe_c1
        # _c4 = self.ln4(_c4)
        _c4 = self.Att_c4(_c4).permute(0, 2, 1).reshape(n, -1, _c4.shape[2], _c4.shape[3])

        _c3 = _c3 + pfe_c1
        # _c3 = self.ln3(_c3)
        _c3 = self.Att_c3(_c3).permute(0, 2, 1).reshape(n, -1, _c3.shape[2], _c3.shape[3])

        _c2 = _c2 + pfe_c1
        # _c2 = self.ln2(_c2)
        _c2 = self.Att_c2(_c2).permute(0, 2, 1).reshape(n, -1, _c2.shape[2], _c2.shape[3])

        _c1 = _c1 + pfe_c1
        # _c1 = self.ln1(_c1)
        _c1 = self.Att_c1(_c1).permute(0, 2, 1).reshape(n, -1, _c1.shape[2], _c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        _c1_s1 = self.dropout(_c1)
        _c1_s1 = self.linear_pred_s1(_c1_s1)

        _c2_s2 = self.dropout(_c2)
        _c2_s2 = self.linear_pred_s2(_c2_s2)

        _c3_s3 = self.dropout(_c3)
        _c3_s3 = self.linear_pred_s3(_c3_s3)

        _c4_s4 = self.dropout(_c4)
        _c4_s4 = self.linear_pred_s4(_c4_s4)

        return x, _c1_s1, _c2_s2, _c3_s3, _c4_s4


class SegFormer(nn.Module):
    def __init__(self, num_classes=21, phi='b0', pretrained=False):
        super(SegFormer, self).__init__()
        self.in_channels = {
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
            'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
        }[phi]
        self.backbone = {
            'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
            'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
        }[phi](pretrained)
        self.embedding_dim = {
            'b0': 256, 'b1': 256, 'b2': 768,
            'b3': 768, 'b4': 768, 'b5': 768,
        }[phi]
        self.decode_head = SegFormerHead(num_classes, self.in_channels, self.embedding_dim)

        self.skip_connection = Unet(3, 64)

        self.out = nn.Conv2d(64 + num_classes, num_classes, kernel_size=1)

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)

        x = self.backbone.forward(inputs)
        x, s1, s2, s3, s4 = self.decode_head.forward(x)

        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)

        s1 = F.interpolate(s1, size=(H, W), mode='bilinear', align_corners=True)
        s2 = F.interpolate(s2, size=(H, W), mode='bilinear', align_corners=True)
        s3 = F.interpolate(s3, size=(H, W), mode='bilinear', align_corners=True)
        s4 = F.interpolate(s4, size=(H, W), mode='bilinear', align_corners=True)

        skip_output = self.skip_connection(inputs)
        x = torch.cat([x, skip_output], dim=1)
        x = self.out(x)

        return x, torch.cat([s1, s2, s3, s4], dim=1)