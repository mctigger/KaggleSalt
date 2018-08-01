from torch import nn
from torch.nn import functional as F


def conv_3x3(in_channels, out_channels, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias)


class RCU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RCU, self).__init__()
        self.relu = nn.ReLU()
        self.conv_1 = conv_3x3(in_channels, out_channels)
        self.conv_2 = conv_3x3(out_channels, out_channels)

    def forward(self, residual):
        x = self.relu(residual)
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.conv_2(x)

        return x + residual


class MrF(nn.Module):
    def __init__(self, channels, low_channel_multiplier=2, low_scale_factor=2, high_scale_factor=1):
        super(MrF, self).__init__()
        self.high_scale_factor = high_scale_factor

        self.conv_high = conv_3x3(channels, channels)
        self.conv_low = conv_3x3(low_channel_multiplier * channels, channels)

        if high_scale_factor > 1:
            self.upsample_high = nn.Upsample(scale_factor=high_scale_factor)

        self.upsample_low = nn.Upsample(scale_factor=low_scale_factor)

    def forward(self, low, high):
        high = self.conv_high(high)
        if self.high_scale_factor > 1:
            high = self.upsample_high(high)
        low = self.conv_low(low)
        low = self.upsample_low(low)

        return low + high


class CRP(nn.Module):
    def __init__(self, channels):
        super(CRP, self).__init__()
        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.conv_1 = conv_3x3(channels, channels)
        self.conv_2 = conv_3x3(channels, channels)
        self.conv_3 = conv_3x3(channels, channels)

    def forward(self, residual):
        residual = self.relu(residual)
        x = self.pool(residual)
        x = self.conv_1(x)
        residual = residual + x

        x = self.pool(x)
        x = self.conv_2(x)
        residual = residual + x

        x = self.pool(x)
        x = self.conv_3(x)
        residual = residual + x

        return residual


class RefineNetBlock(nn.Module):
    def __init__(self, channels, low_channel_multiplier=2, low_scale_factor=2, high_scale_factor=1):
        super(RefineNetBlock, self).__init__()
        self.rcu_low_1 = RCU(channels * low_channel_multiplier, channels * low_channel_multiplier)
        self.rcu_low_2 = RCU(channels * low_channel_multiplier, channels * low_channel_multiplier)
        self.rcu_high_1 = RCU(channels, channels)
        self.rcu_high_2 = RCU(channels, channels)
        self.mrf = MrF(channels, low_channel_multiplier, low_scale_factor, high_scale_factor)
        self.crp = CRP(channels)
        self.out = RCU(channels, channels)

    def forward(self, low, high):
        low = self.rcu_low_1(low)
        low = self.rcu_low_2(low)
        high = self.rcu_high_1(high)
        high = self.rcu_high_2(high)
        x = self.mrf(low, high)
        x = self.crp(x)
        x = self.out(x)

        return x


class RefineNet(nn.Module):
    def __init__(self, resnet):
        super(RefineNet, self).__init__()

        self.refine_0 = RefineNetBlock(64, low_channel_multiplier=4, low_scale_factor=2, high_scale_factor=2)
        self.refine_1 = RefineNetBlock(256)
        self.refine_2 = RefineNetBlock(512)
        self.refine_3 = RefineNetBlock(1024)

        self.classifier = nn.Sequential(*[
            nn.Conv2d(64, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        ])

        self.resnet = resnet

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x_0 = x = self.resnet.maxpool(x)

        x_1 = x = self.resnet.layer1(x)
        x_2 = x = self.resnet.layer2(x)
        x_3 = x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.refine_3(x, x_3)
        x = self.refine_2(x, x_2)
        x = self.refine_1(x, x_1)
        x = self.refine_0(x, x_0)

        x = self.classifier(x)

        return F.upsample(x, scale_factor=2, mode='bilinear')

