import torch
from torch import nn
from torchvision.models.resnet import conv3x3

from .modules import SCSEBlock
from .refinenet import RCU


class BR(nn.Module):
    def __init__(self, channels):
        super(BR, self).__init__()

        self.block = SCSEBlock(channels, channels)

    def forward(self, x):
        return self.block(x)


class ELUBR(nn.Module):
    expansion = 1

    def __init__(self, channels):
        super(ELUBR, self).__init__()
        self.conv1 = conv3x3(channels, channels)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ELU(inplace=True)
        self.conv2 = conv3x3(channels, channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class PyramidGCN(nn.Module):
    def __init__(self, in_channels, channels, kernel_size=None):
        super(PyramidGCN, self).__init__()

        self.gcn_1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.gcn_3 = nn.Sequential(nn.Conv2d(in_channels, channels, 1, bias=False), GCN(channels, channels, kernel_size=3))
        self.gcn_5 = nn.Sequential(nn.Conv2d(in_channels, channels, 1, bias=False), GCN(channels, channels, kernel_size=5))
        self.gcn_7 = nn.Sequential(nn.Conv2d(in_channels, channels, 1, bias=False), GCN(channels, channels, kernel_size=7))

    def forward(self, x):
        x_1 = self.gcn_1(x)
        x_3 = self.gcn_3(x)
        x_5 = self.gcn_5(x)
        x_7 = self.gcn_7(x)

        return x_1 + x_3 + x_5 + x_7


class PyramidConcatGCN(nn.Module):
    def __init__(self, in_channels, channels, kernel_size=None):
        super(PyramidConcatGCN, self).__init__()

        channels = channels // 4

        self.gcn_1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.gcn_3 = nn.Sequential(nn.Conv2d(in_channels, channels, 1, bias=False), GCN(channels, channels, kernel_size=3))
        self.gcn_5 = nn.Sequential(nn.Conv2d(in_channels, channels, 1, bias=False), GCN(channels, channels, kernel_size=5))
        self.gcn_7 = nn.Sequential(nn.Conv2d(in_channels, channels, 1, bias=False), GCN(channels, channels, kernel_size=7))

    def forward(self, x):
        x_1 = self.gcn_1(x)
        x_3 = self.gcn_3(x)
        x_5 = self.gcn_5(x)
        x_7 = self.gcn_7(x)

        return torch.cat([x_1, x_3, x_5, x_7], dim=1)


class GCN(nn.Module):
    def __init__(self, in_channels, channels, kernel_size=7):
        super(GCN, self).__init__()

        self.conv_a_1 = nn.Conv2d(in_channels, channels, kernel_size=(kernel_size, 1), padding=(kernel_size // 2, 0), bias=False)
        self.conv_a_2 = nn.Conv2d(channels, channels, kernel_size=(1, kernel_size), padding=(0, kernel_size // 2), bias=False)

        self.conv_b_1 = nn.Conv2d(in_channels, channels, kernel_size=(1, kernel_size), padding=(0, kernel_size // 2), bias=False)
        self.conv_b_2 = nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1), padding=(kernel_size // 2, 0), bias=False)

    def forward(self, x):
        x_a = self.conv_a_1(x)
        x_a = self.conv_a_2(x_a)

        x_b = self.conv_a_1(x)
        x_b = self.conv_a_2(x_b)

        return x_a + x_b


class NoUpsampleClassifier(nn.Module):
    def __init__(self, in_channels, channels):
        super(NoUpsampleClassifier, self).__init__()
        self.classifier = nn.Sequential(*[
            nn.Conv2d(in_channels, channels, kernel_size=1, bias=False),
            RCU(channels, channels),
            RCU(channels, channels),
            nn.Conv2d(channels, 1, kernel_size=1, bias=True),
        ])

    def forward(self, x):
        return self.classifier(x)


class LargeKernelMattersNet(nn.Module):
    def __init__(self, base, num_features_base=None, classifier=None, gcn=GCN, br=BR, k=128):
        super(LargeKernelMattersNet, self).__init__()

        if num_features_base is None:
            num_features_base = [256, 512, 1024, 2048]

        if classifier is None:
            classifier = nn.Sequential(
                br(k),
                nn.ConvTranspose2d(k, k, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                br(k),
                nn.Conv2d(k, 1, kernel_size=1, stride=1, padding=0, bias=True)
            )

        self.gcn_1 = gcn(num_features_base[0], k)
        self.gcn_2 = gcn(num_features_base[1], k)
        self.gcn_3 = gcn(num_features_base[2], 2*k)
        self.gcn_4 = gcn(num_features_base[3], 2*k, kernel_size=3)

        self.br_1_a = nn.Sequential(br(k), br(k))
        self.br_1_b = nn.Sequential(br(k), br(k))
        self.br_2_a = nn.Sequential(br(k), br(k))
        self.br_2_b = nn.Sequential(br(k), br(k))
        self.br_3_a = nn.Sequential(br(2*k), br(2*k))
        self.br_3_b = nn.Sequential(br(2*k), br(2*k))
        self.br_4 = nn.Sequential(br(2*k), br(2*k))

        self.deconv_1 = nn.ConvTranspose2d(k, k, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.deconv_2 = nn.ConvTranspose2d(k, k, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.deconv_3 = nn.ConvTranspose2d(2*k, k, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.deconv_4 = nn.ConvTranspose2d(2*k, 2*k, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)

        self.classifier = classifier

        self.base = base

    def forward(self, x):
        x_1, x_2, x_3, x_4 = self.base(x)

        x_1 = self.br_1_a(self.gcn_1(x_1))
        x_2 = self.br_2_a(self.gcn_2(x_2))
        x_3 = self.br_3_a(self.gcn_3(x_3))
        x_4 = self.br_4(self.gcn_4(x_4))

        x = self.deconv_4(x_4)
        x = self.deconv_3(self.br_3_b(x + x_3))
        x = self.deconv_2(self.br_2_b(x + x_2))
        x = self.deconv_1(self.br_1_b(x + x_1))

        x = self.classifier(x)

        return x
