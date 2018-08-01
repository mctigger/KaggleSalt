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
    def __init__(self):
        super(UNet, self).__init__()

        self.en_1 = UNetBlock(3, 32, num_layers=2, stride=1)
        self.en_2 = UNetBlock(32, 64, num_layers=2, stride=2)
        self.en_3 = UNetBlock(64, 128, num_layers=2, stride=2)
        self.en_4 = UNetBlock(128, 256, num_layers=2, stride=2)
        self.en_5 = UNetBlock(256, 512, num_layers=2, stride=2)

        self.de_1 = UNetBlock(512, 512, num_layers=2, stride=1)
        self.de_2 = UNetBlock(512 + 256, 256, num_layers=2, stride=1)
        self.de_3 = UNetBlock(256 + 128, 128, num_layers=2, stride=1)
        self.de_4 = UNetBlock(128 + 64, 64, num_layers=2, stride=1)
        self.de_5 = UNetBlock(64 + 32, 32, num_layers=2, stride=1)

        self.classifier = nn.Sequential(*[
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        ])

    def forward(self, x):
        x = en_1 = self.en_1(x)
        x = en_2 = self.en_2(x)
        x = en_3 = self.en_3(x)
        x = en_4 = self.en_4(x)
        x = self.en_5(x)
        x = self.de_1(x)
        x = self.de_2(torch.cat([F.upsample(x, scale_factor=2, mode='bilinear'), en_4], dim=1))
        x = self.de_3(torch.cat([F.upsample(x, scale_factor=2, mode='bilinear'), en_3], dim=1))
        x = self.de_4(torch.cat([F.upsample(x, scale_factor=2, mode='bilinear'), en_2], dim=1))
        x = self.de_5(torch.cat([F.upsample(x, scale_factor=2, mode='bilinear'), en_1], dim=1))

        x = self.classifier(x)

        return x