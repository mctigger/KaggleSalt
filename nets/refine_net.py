from torch import nn
from torch.nn import functional as F


def conv_3x3(in_channels, out_channels, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias)


class RCU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RCU, self).__init__()
        self.relu = nn.ReLU()
        self.conv_1 = conv_3x3(in_channels, out_channels, bias=True)
        self.conv_2 = conv_3x3(out_channels, out_channels)

    def forward(self, residual):
        x = self.relu(residual)
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.conv_2(x)

        return x + residual


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

    def forward(self, inputs):
        sum = None
        for inp, path in zip(inputs, self.paths):
            if sum is None:
                sum = path(inp)
            else:
                sum = sum + path(inp)

        return sum


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
    def __init__(self, channels, config):
        super(RefineNetBlock, self).__init__()
        paths = []
        for in_channels, scale_factor in config:
            p = nn.Sequential(*[
                nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
                RCU(channels, channels),
                RCU(channels, channels)
            ])
            paths.append(p)

        self.paths = nn.ModuleList(paths)

        self.mrf = MrF(channels, [scale_factor for in_channels, scale_factor in config])
        self.crp = CRP(channels)
        self.out = RCU(channels, channels)

    def forward(self, inputs):
        paths = [path(inp) for inp, path in zip(inputs, self.paths)]
        x = self.mrf(paths)
        x = self.crp(x)
        x = self.out(x)

        return x


class RefineNet(nn.Module):
    def __init__(self, resnet, num_features=256, block_multiplier=4):
        super(RefineNet, self).__init__()

        self.refine_0 = RefineNetBlock(num_features*2, [(block_multiplier*512, 1)])
        self.refine_1 = RefineNetBlock(num_features, [(block_multiplier*256, 1), (num_features*2, 2)])
        self.refine_2 = RefineNetBlock(num_features, [(block_multiplier*128, 1), (num_features, 2)])
        self.refine_3 = RefineNetBlock(num_features, [(block_multiplier*64, 1), (num_features, 2)])

        self.classifier = nn.Sequential(*[
            RCU(num_features, num_features),
            RCU(num_features, num_features),
            nn.Conv2d(num_features, 1, kernel_size=1, bias=True)
        ])

        self.resnet = resnet

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x_0 = self.resnet.layer1(x)
        x_1 = self.resnet.layer2(x_0)
        x_2 = self.resnet.layer3(x_1)
        x_3 = self.resnet.layer4(x_2)

        x = self.refine_0([x_3])
        x = self.refine_1([x_2, x])
        x = self.refine_2([x_1, x])
        x = self.refine_3([x_0, x])

        x = self.classifier(x)
        x = F.upsample(x, scale_factor=4, mode='bilinear')

        return x
