import torch.nn as nn
import torch


class AWDC(nn.Module):
    def __init__(self, in_channel, ratio=4):
        super(AWDC, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=in_channel // ratio,
                               bias=False, kernel_size=1, padding=0)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv1d(in_channels=in_channel // ratio, out_channels=in_channel,
                               bias=False, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        b, c, h, w = inputs.shape
        x = self.avg_pool(inputs)
        x = x.view([b, c, 1])

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)

        x = x.view([b, c, 1, 1])

        outputs = x * inputs
        return outputs
class Discriminator_MWPD(nn.Module):

    def __init__(self, num_classes, ndf = 64):
        super(Discriminator_MWPD, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)

        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)

        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)

        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)

        self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.awdc = AWDC(in_channel=num_classes)

    def forward(self, x):
        x = self.awdc(x)

        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x