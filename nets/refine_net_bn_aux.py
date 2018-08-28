from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck


def conv_3x3(in_channels, out_channels, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias)


class RefineNetUpsampleClassifier(nn.Module):
    def __init__(self, num_features):
        super(RefineNetUpsampleClassifier, self).__init__()
        self.classifier = nn.Sequential(*[
            RCU(num_features, num_features),
            RCU(num_features, num_features),
            nn.Conv2d(num_features, 1, kernel_size=1, bias=True),
            nn.Upsample(scale_factor=4, mode='bilinear')
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


class BottleneckRCU(nn.Module):
    multiplier = 4

    def __init__(self, in_channels, out_channels):
        super(BottleneckRCU, self).__init__()
        self.block = Bottleneck(in_channels, out_channels)

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
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.conv_1 = conv_3x3(channels, channels)
        self.conv_2 = conv_3x3(channels, channels)
        self.conv_3 = conv_3x3(channels, channels)

    def forward(self, residual):
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
    def __init__(self, channels, config, crp=CRP, rcu=RCU):
        super(RefineNetBlock, self).__init__()
        paths = []
        for in_channels, scale_factor in config:
            p = nn.Sequential(*[
                nn.Conv2d(in_channels, channels*rcu.multiplier, kernel_size=1, stride=1, padding=0, bias=False),
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


class AuxilaryClassifier(nn.Module):
    def __init__(self):
        super(AuxilaryClassifier, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(2048, 1)

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


class RefineNet(nn.Module):
    def __init__(
            self,
            encoder,
            num_features=256,
            block_multiplier=4,
            crp=CRP,
            rcu=RCU,
            classifier=RefineNetUpsampleClassifier
    ):
        super(RefineNet, self).__init__()

        self.refine_0 = RefineNetBlock(num_features*2, [(block_multiplier*512, 1)], crp=crp, rcu=rcu)
        self.refine_1 = RefineNetBlock(num_features, [(block_multiplier*256, 1), (num_features*2*rcu.multiplier, 2)], crp=crp, rcu=rcu)
        self.refine_2 = RefineNetBlock(num_features, [(block_multiplier*128, 1), (num_features*rcu.multiplier, 2)], crp=crp, rcu=rcu)
        self.refine_3 = RefineNetBlock(num_features, [(block_multiplier*64, 1), (num_features*rcu.multiplier, 2)], crp=crp, rcu=rcu)

        self.aux = AuxilaryClassifier()
        self.classifier = classifier(num_features*rcu.multiplier)

        self.encoder = encoder

    def forward(self, x):
        x = self.encoder.layer0(x)
        x_0 = self.encoder.layer1(x)
        x_1 = self.encoder.layer2(x_0)
        x_2 = self.encoder.layer3(x_1)
        x_3 = self.encoder.layer4(x_2)

        x = self.refine_0([x_3])
        x = self.refine_1([x_2, x])
        x = self.refine_2([x_1, x])
        x = self.refine_3([x_0, x])

        aux = self.aux(x_3)
        x = self.classifier(x)

        return x, aux


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


class DilatedPyramidPooling(nn.Module):
    def __init__(self, channels):
        super(DilatedPyramidPooling, self).__init__()

        self.conv_a = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.conv_b_1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv_b_2 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2)
        self.conv_c_1 = nn.Conv2d(channels, channels, kernel_size=5, padding=2)
        self.conv_c_2 = nn.Conv2d(channels, channels, kernel_size=5, padding=4, dilation=2)

    def forward(self, x):
        a = self.conv_a(x)
        b_1 = self.conv_b_1(x)
        b_2 = self.conv_b_2(x)
        c_1 = self.conv_c_1(x)
        c_2 = self.conv_c_2(x)

        return a + b_1 + b_2 + c_1 + c_2 + x


class PyramidPooling(nn.Module):
    def __init__(self, channels):
        super(PyramidPooling, self).__init__()

        self.conv_a = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.conv_b = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.conv_c = nn.Sequential(
            nn.AvgPool2d(kernel_size=4, stride=4),
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        a = self.conv_a(x)
        b = self.conv_b(x)
        c = self.conv_c(x)

        return a + b + c + x


class Identity(nn.Module):
    def __init__(self, channels):
        super(Identity, self).__init__()

    def forward(self, x):

        return x