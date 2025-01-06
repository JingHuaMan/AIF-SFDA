import random

import torch
from torch import nn


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            SeparableConv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            SeparableConv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
        )
        self.channel_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, input):
        x1 = self.conv(input)
        x2 = self.channel_conv(input)
        x = x1 + x2
        return x

# img = torch.rand([2, 512, 32, 32]).cuda()
# model = SeparableConv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False).cuda()
# print(model)
# out = model(img)
# print(out.shape)
# # print(shapes)