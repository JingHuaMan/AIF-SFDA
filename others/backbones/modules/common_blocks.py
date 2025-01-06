from torch import nn

from others.backbones.modules import dropout_modified


class MultiConv(nn.Module):
    def __init__(self, input_nc, output_nc, conv_block_num=2, kernel_size=3, use_bias=False, norm_func=nn.BatchNorm2d,
                 activation_func=nn.ReLU, use_dropout=dropout_modified.DROPOUT_NONE, do_residual=False,
                 last_activation=True):
        super(MultiConv, self).__init__()
        assert conv_block_num >= 1
        main_conv_list = [nn.Conv2d(input_nc, output_nc, kernel_size, 1, kernel_size // 2, bias=use_bias),
                          norm_func(output_nc), activation_func(inplace=True)] + [
                             dropout_modified.get_dropout(use_dropout)]

        for i in range(conv_block_num - 1):
            main_conv_list.append(
                nn.Conv2d(output_nc, output_nc, kernel_size, 1, kernel_size // 2, bias=use_bias))
            main_conv_list.append(norm_func(output_nc))
            if i == conv_block_num - 2 and last_activation:
                main_conv_list.append(activation_func(inplace=True))
            main_conv_list.append(dropout_modified.get_dropout(use_dropout))
        self.main_conv = nn.Sequential(*main_conv_list)
        self.do_residual = do_residual

    def forward(self, x):
        output = self.main_conv(x) + (x if self.do_residual else 0)
        return output


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)
