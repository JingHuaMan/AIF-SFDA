import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init
import numpy as np

class Decoder(nn.Module):
    "Input: 64x64x128, Output: 512x512x1"
    def __init__(self, out_channel=1, do_final_relu=True):
        super(Decoder, self).__init__()
        # Input is 64x64x128
        ######   smaller model for edge and separation function#############

        self.active_elu = nn.ELU()
        self.active_relu = nn.ReLU()
        self.active_tanh = nn.Tanh()
        self.active_sigmoid = nn.Sigmoid()


        self.pool = nn.MaxPool2d(kernel_size=2)
        self.unpool = nn.MaxUnpool2d(kernel_size=2)


        self.conv_2_16 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv_16_32 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_64_32 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.deconv_128_64 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=2,
                                               output_padding=1)
        self.deconv_32_16 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.deconv_16_1 = nn.ConvTranspose2d(in_channels=16, out_channels=out_channel, kernel_size=3, padding=1, stride=2,
                                               output_padding=1)

        self.bnorm32 = nn.modules.BatchNorm2d(32)
        self.bnorm16 = nn.modules.BatchNorm2d(16)

        self.do_final_relu = do_final_relu

    def forward(self, x):   # Input is 64x64x128

        out_128x128x64 = self.deconv_128_64(x)
        out_128x128x32 = self.conv_64_32(out_128x128x64)
        out_128x128x32 = self.bnorm32( out_128x128x32)
        out_128x128x32 = self.active_relu(out_128x128x32)

        out_256x256x16_up = self.deconv_32_16(out_128x128x32)
        out_256x256x16_up = self.bnorm16(out_256x256x16_up)
        out_256x256x16_up = self.active_relu(out_256x256x16_up)

        out_512x512x1_up = self.deconv_16_1(out_256x256x16_up)
        if self.do_final_relu:
            out_512x512x1_up = self.active_relu(out_512x512x1_up)
        return out_512x512x1_up

class EdgeNet(nn.Module):

    def __init__(self):
        super(EdgeNet, self).__init__()
        # super().__init__
    ######   MOdel 2#############

        self.active_elu = nn.ELU()
        self.active_relu = nn.ReLU()
        self.active_tanh = nn.Tanh()
        self.active_sigmoid = nn.Sigmoid()

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.unpool = nn.MaxUnpool2d(kernel_size=2)

        # self.conv_1_1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.conv_1_16 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv_16_32 =  nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_32_64 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_32_16 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv_64_128 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_128_256 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_256_512 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)

        # self.deconv_128_128 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.deconv_128_128_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.deconv_256_64 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.deconv_128_32 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=3, padding=1, stride=2, output_padding=1)
        # self.deconv_32_16 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.deconv_64_16 = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.deconv_64_1 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=3, padding=1, stride=2, output_padding=1)

        # same size deconvolution
        self.deconv_same_32_16 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, padding=1, stride=1)
        self.deconv_same_16_1 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, padding=1, stride=1)

    def forward(self, x): # input 512x512x32
        # out_512x512x16 = self.conv_1_16(x)
        out_512x512x16 = self.conv_32_16(x)
        out_512x512x16 = self.active_relu(out_512x512x16)
        out_256x256x16 = self.pool(out_512x512x16)

        out_256x256x32 = self.conv_16_32(out_256x256x16)
        out_256x256x32 = self.active_relu(out_256x256x32)
        out_128x128x32 = self.pool(out_256x256x32)                                           # concatenated with out_128x128x32_up

        out_128x128x64 = self.conv_32_64(out_128x128x32)
        out_128x128x64 = self.active_relu(out_128x128x64)
        out_64x64x64 = self.pool(out_128x128x64)

        out_64x64x128 = self.conv_64_128(out_64x64x64)
        out_64x64x128 = self.active_relu(out_64x64x128)
        # out_32x32x128 = self.pool(out_64x64x128)

        # out_64x64x128_up = self.deconv_128_128(out_32x32x128)
        out_64x64x128_up = self.deconv_128_128_1(out_64x64x128)
        out_64x64x128_up = self.active_relu(out_64x64x128_up)
        out_64x64x256_cat = torch.cat((out_64x64x128_up, out_64x64x128), dim=1)

        out_128x128x64_up = self.deconv_256_64(out_64x64x256_cat)
        out_128x128x64_up = self.active_relu(out_128x128x64_up)
        out_128x128x128_cat = torch.cat((out_128x128x64_up, out_128x128x64), dim=1)                      # Concatenated with i/p

        out_256x256x32_up = self.deconv_128_32(out_128x128x128_cat)
        out_256x256x32_up = self.active_relu(out_256x256x32_up)
        out_256x256x64_cat = torch.cat((out_256x256x32_up, out_256x256x32), dim=1)

        out_512x512x16_up = self.deconv_64_16(out_256x256x64_cat)
        out_512x512x16_up = self.active_relu(out_512x512x16_up)
        out_512x512x32_cat = torch.cat((out_512x512x16_up,out_512x512x16), dim=1)

        out_512x512x16_last = self.deconv_same_32_16(out_512x512x32_cat)
        out_512x512x16_last = self.active_relu(out_512x512x16_last)
        seg = self.deconv_same_16_1(out_512x512x16_last)
        return seg

class DeconvBlock(nn.Module):
    """
        This block performs a sequential deconv(upscale)-batchnorm-relu
    """
    def __init__(self, in_ch, out_ch):
        super(DeconvBlock, self).__init__()
        self.deconv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.modules.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.deconv_block(x)

class ConvBlock(nn.Module):
    """
        This block performs a sequential conv-batchnorm-relu
    """
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.modules.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)

class UNet(nn.Module):

    def __init__(self, in_channel=1):
        super(UNet, self).__init__()

        self.active_elu = nn.ELU()
        self.active_relu = nn.ReLU()
        self.active_tanh = nn.Tanh()
        self.active_sigmoid = nn.Sigmoid()

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.unpool = nn.MaxUnpool2d(kernel_size=2)

        self.conv_block_1_16 = ConvBlock(in_ch=in_channel, out_ch=16)
        self.conv_block_16_32 = ConvBlock(in_ch=16, out_ch=32)
        self.conv_block_32_64 = ConvBlock(in_ch=32, out_ch=64)
        self.conv_block_32_16 = ConvBlock(in_ch=32, out_ch=16)
        self.conv_block_64_128 = ConvBlock(in_ch=64, out_ch=128)
        self.conv_block_128_128 = ConvBlock(in_ch=128, out_ch=128)

        self.deconv_block_256_64 = DeconvBlock(in_ch=256, out_ch=64)
        self.deconv_block_128_32 = DeconvBlock(in_ch=128, out_ch=32)
        self.deconv_block_64_16 = DeconvBlock(in_ch=64, out_ch=16)

    def forward(self, x):

        out_512x512x16 = self.conv_block_1_16(x)
        out_256x256x16 = self.pool(out_512x512x16)

        out_256x256x32 = self.conv_block_16_32(out_256x256x16)
        out_128x128x32 = self.pool(out_256x256x32)

        out_128x128x64 = self.conv_block_32_64(out_128x128x32)
        out_64x64x64 = self.pool(out_128x128x64)

        out_64x64x128 = self.conv_block_64_128(out_64x64x64)              # This is the middle layer

        out_64x64x128_up = self.conv_block_128_128(out_64x64x128)
        out_64x64x256_cat = torch.cat((out_64x64x128_up, out_64x64x128), dim=1)

        out_128x128x64_up = self.deconv_block_256_64(out_64x64x256_cat)
        out_128x128x128_cat = torch.cat((out_128x128x64_up, out_128x128x64), dim=1)

        out_256x256x32_up = self.deconv_block_128_32(out_128x128x128_cat)
        out_256x256x64_cat = torch.cat((out_256x256x32_up, out_256x256x32), dim=1)

        out_512x512x16_up = self.deconv_block_64_16(out_256x256x64_cat)
        out_512x512x32_cat = torch.cat((out_512x512x16_up,out_512x512x16), dim=1)

        out_512x512x16_last = self.conv_block_32_16(out_512x512x32_cat)   # Output Layer

        return out_512x512x16_last, out_512x512x16, out_256x256x32, out_128x128x64, out_64x64x128

class FeatureExtractor2D(nn.Module):
    """
    Input: num_ch = 16, this is generally the final output of a Unet
    Output: num_ch = 1, same size of image as the input. This module only uses 1x1 convolutions
    """
    def __init__(self):
        super(FeatureExtractor2D, self).__init__()
        self.reduction_net = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.reduction_net(x)

