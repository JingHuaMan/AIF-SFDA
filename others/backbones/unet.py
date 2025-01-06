import torch
from torch import nn

from others.backbones.modules import dropout_modified
from others.backbones.modules.common_blocks import MultiConv


class Unet(nn.Module):
    ALL_OUTPUT = 0
    PART_OUTPUT = 1
    ONE_OUTPUT = 2

    def __init__(self, input_nc, output_nc, depth=5, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=dropout_modified.DROPOUT_NONE, last_layer='Sigmoid',
                 activation_func=nn.ReLU, kernel_size=3, conv_num=2, use_bias=False, output_mode=ONE_OUTPUT):
        super(Unet, self).__init__()

        if isinstance(output_mode, str):
            output_mode = getattr(Unet, output_mode.upper())
        self.output_mode = output_mode

        self.down_down = nn.MaxPool2d(2, 2)
        self.down_conv_list = nn.ModuleList(
            [MultiConv(input_nc if i == 0 else ngf * 2 ** (i - 1), ngf * 2 ** i, conv_num, kernel_size, use_bias,
                       norm_layer, activation_func, use_dropout) for i in range(depth)])

        self.up_conv_list = nn.ModuleList(
            [MultiConv(ngf * 2 ** (i - 1), ngf * 2 ** (i - 2), conv_num, kernel_size, use_bias, norm_layer,
                       activation_func,
                       use_dropout) for i in range(depth, 1, -1)])

        self.up_up_list = nn.ModuleList(
            [nn.ConvTranspose2d(ngf * 2 ** (i - 1), ngf * 2 ** (i - 2), kernel_size=2, stride=2, padding=0,
                                bias=use_bias)
             for i in range(depth, 1, -1)])

        self.out = nn.Sequential(
            nn.Conv2d(ngf, output_nc, kernel_size=1, padding=0, bias=use_bias),
            getattr(nn, last_layer)()
        )

    def forward(self, x):
        x_list = [self.down_conv_list[0](x)]
        for i in range(1, len(self.down_conv_list)):
            x_list.append(self.down_conv_list[i](self.down_down(x_list[-1])))

        y_list = [x_list[-1]]
        for i in range(len(self.up_conv_list)):
            y = torch.cat([x_list[-i - 2], self.up_up_list[i](y_list[-1])], dim=1)
            y_list.append(self.up_conv_list[i](y))

        o = self.out(y_list[-1])

        if self.output_mode == Unet.ALL_OUTPUT:
            return o, *y_list[::-1]  # from shallow to deep
        elif self.output_mode == Unet.PART_OUTPUT:
            return o, y_list[-1], x_list[0]
        else:
            return o
