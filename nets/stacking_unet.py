import torch
from torch import nn
from torch.nn import functional as F


def conv_3x3(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )


class UNetBlock(nn.Module):
    def __init__(self, inplanes, planes, num_layers, stride=1):
        super(UNetBlock, self).__init__()

        layers = [
            conv_3x3(inplanes, planes, stride=stride),
            *[conv_3x3(planes, planes) for _ in range(num_layers - 1)]
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self, in_channels):
        super(UNet, self).__init__()

        self.en_1 = UNetBlock(in_channels, 32, num_layers=2, stride=1)
        self.en_5 = UNetBlock(32, 64, num_layers=2, stride=2)

        self.de_1 = UNetBlock(64, 64, num_layers=2, stride=1)
        self.de_5 = UNetBlock(64 + 32, 32, num_layers=2, stride=1)

        self.classifier = nn.Sequential(*[
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        ])

    def forward(self, x):
        x = en_1 = self.en_1(x)
        x = self.en_5(x)
        x = self.de_1(x)
        x = self.de_5(torch.cat([F.upsample(x, scale_factor=2, mode='bilinear'), en_1], dim=1))

        x = self.classifier(x)

        return x