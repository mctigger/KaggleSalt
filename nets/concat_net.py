import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import BasicBlock

def conv_3x3(in_channels, out_channels, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias)


class RefineNetUpsampleClassifier(nn.Module):
    def __init__(self, num_features, scale_factor=4):
        super(RefineNetUpsampleClassifier, self).__init__()
        self.classifier = nn.Sequential(*[
            RCU(num_features, num_features),
            RCU(num_features, num_features),
            nn.Conv2d(num_features, 1, kernel_size=1, bias=True),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        ])

    def forward(self, x):
        return self.classifier(x)


class RefineNetDetailedClassifier(nn.Module):
    def __init__(self, num_features):
        super(RefineNetDetailedClassifier, self).__init__()
        self.classifier = nn.Sequential(*[
            RCU(num_features, num_features // 2),
            RCU(num_features // 2, num_features // 2),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            RCU(num_features // 2, num_features // 4),
            RCU(num_features // 4, num_features // 4),
            nn.Conv2d(num_features, 1, kernel_size=1, bias=True),
            nn.Upsample(scale_factor=2, mode='bilinear')
        ])

    def forward(self, x):
        return self.classifier(x)


class RCU(nn.Module):
    multiplier = 1

    def __init__(self, in_channels, out_channels):
        super(RCU, self).__init__()
        self.block = BasicBlock(in_channels, out_channels)

    def forward(self, x):
        return self.block(x)


class MrF(nn.Module):
    def __init__(self, channels, scale_factors):
        super(MrF, self).__init__()
        
        paths = []
        for s in scale_factors:
            paths.append(nn.Sequential(
                conv_3x3(channels, channels),
                nn.Upsample(scale_factor=s, mode='bilinear')
            ))

        self.paths = nn.ModuleList(paths)

        self.downsample = nn.Conv2d(len(scale_factors)*channels, channels, 1, bias=False)

    def forward(self, inputs):
        x = [path(inp) for inp, path in zip(inputs, self.paths)]
        x = torch.cat(x, dim=1)
        x = self.downsample(x)

        return x


class CRP(nn.Module):
    def __init__(self, channels):
        super(CRP, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.conv_1 = conv_3x3(channels, channels)
        self.conv_2 = conv_3x3(channels, channels)
        self.conv_3 = conv_3x3(channels, channels)
        self.downsample = nn.Conv2d(4*channels, channels, 1, bias=False)

    def forward(self, residual):
        x = self.pool(residual)
        x = self.conv_1(x)
        residual = torch.cat([residual, x], dim=1)

        x = self.pool(x)
        x = self.conv_2(x)
        residual = torch.cat([residual, x], dim=1)

        x = self.pool(x)
        x = self.conv_3(x)

        x = torch.cat([residual, x], dim=1)
        x = self.downsample(x)

        return x


class RefineNetBlock(nn.Module):
    def __init__(self, channels, config, crp=CRP, rcu=RCU, dropout=0):
        super(RefineNetBlock, self).__init__()
        paths = []
        for in_channels, scale_factor in config:
            p = nn.Sequential(*[
                nn.Conv2d(in_channels, channels*rcu.multiplier, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Dropout2d(dropout),
                rcu(channels*rcu.multiplier, channels),
                rcu(channels*rcu.multiplier, channels)
            ])
            paths.append(p)

        self.paths = nn.ModuleList(paths)

        self.mrf = MrF(channels*rcu.multiplier, [scale_factor for in_channels, scale_factor in config])
        self.crp = crp(channels*rcu.multiplier)
        self.out = rcu(channels*rcu.multiplier, channels)

    def forward(self, inputs):
        paths = [path(inp) for inp, path in zip(inputs, self.paths)]
        x = self.mrf(paths)
        x = self.crp(x)
        x = self.out(x)

        return x


class RefineNet(nn.Module):
    def __init__(
            self,
            encoder,
            num_features=None,
            num_features_base=None,
            block_multiplier=4,
            crp=CRP,
            rcu=RCU,
            classifier=RefineNetUpsampleClassifier,
            dropout=0
    ):
        super(RefineNet, self).__init__()

        if num_features is None:
            num_features = [256*2, 256, 256, 256]

        if not isinstance(num_features, list):
            num_features = [num_features*2, num_features, num_features, num_features]

        if num_features_base is None:
            num_features_base = [64, 128, 256, 512]

        self.refine_0 = RefineNetBlock(num_features[0], [(block_multiplier*num_features_base[3], 1)], crp=crp, rcu=rcu, dropout=dropout)
        self.refine_1 = RefineNetBlock(num_features[1], [(block_multiplier*num_features_base[2], 1), (num_features[0]*rcu.multiplier, 2)], crp=crp, rcu=rcu, dropout=dropout)
        self.refine_2 = RefineNetBlock(num_features[2], [(block_multiplier*num_features_base[1], 1), (num_features[1]*rcu.multiplier, 2)], crp=crp, rcu=rcu, dropout=dropout)
        self.refine_3 = RefineNetBlock(num_features[3], [(block_multiplier*num_features_base[0], 1), (num_features[2]*rcu.multiplier, 2)], crp=crp, rcu=rcu, dropout=dropout)

        self.classifier = classifier(num_features[3]*rcu.multiplier)

        self.encoder = encoder

    def forward(self, x):
        x_0, x_1, x_2, x_3 = self.encoder(x)

        x = self.refine_0([x_3])
        x = self.refine_1([x_2, x])
        x = self.refine_2([x_1, x])
        x = self.refine_3([x_0, x])

        x = self.classifier(x)

        return x


class ResNetBase(nn.Module):
    def __init__(self, resnet):
        super(ResNetBase, self).__init__()
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x_0 = self.layer0(x)
        x_1 = self.layer1(x_0)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)

        return x_1, x_2, x_3, x_4


class ResNeXtBase(nn.Module):
    def __init__(self, resnet):
        super(ResNeXtBase, self).__init__()
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x_0 = self.layer0(x)
        x_1 = self.layer1(x_0)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)

        return x_1, x_2, x_3, x_4
