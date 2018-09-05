from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet, conv3x3


def transposed_conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, output_padding=1, bias=False)


class TransposedBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(TransposedBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = transposed_conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = nn.Sequential(
            transposed_conv3x3(inplanes, planes, stride),
            nn.BatchNorm2d(planes),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out


class TransposedBottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(TransposedBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = transposed_conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Sequential(
            transposed_conv3x3(inplanes, 4*planes, stride),
            nn.BatchNorm2d(4*planes),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out


class UResNetLayer(nn.Module):
    def __init__(self, inplanes, planes, num_layers):
        super(UResNetLayer, self).__init__()
        self.layers = nn.Sequential(*[
            BasicBlock(inplanes, planes, downsample=nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(planes)
            )),
            *[BasicBlock(planes, planes) for _ in range(num_layers - 2)],
            TransposedBasicBlock(planes, planes, stride=2)
        ])

    def forward(self, x):
        return self.layers(x)


class UResNet(nn.Module):
    def __init__(self, resnet, layers):
        super(UResNet, self).__init__()

        self.resnet = resnet

        self.de_1 = UResNetLayer(512, 256, layers[3])
        self.de_2 = UResNetLayer(256, 128, layers[2])
        self.de_3 = UResNetLayer(128, 64, layers[1])
        self.de_4 = UResNetLayer(64, 64, layers[0])

        self.classifier = nn.Sequential(*[
            nn.Conv2d(64, 1, kernel_size=1, bias=True),
        ])

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x_1 = x = self.resnet.layer1(x)
        x_2 = x = self.resnet.layer2(x)
        x_3 = x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.de_1(x)
        x = self.de_2(x + x_3)
        x = self.de_3(x + x_2)
        x = self.de_4(x + x_1)

        x = self.classifier(x)

        return F.upsample(x, scale_factor=2, mode='bilinear')


class BottleneckUResNetLayer(nn.Module):
    def __init__(self, inplanes, planes, num_layers):
        super(BottleneckUResNetLayer, self).__init__()
        self.conv_1 = Bottleneck(inplanes, planes, downsample=nn.Sequential(
            nn.Conv2d(inplanes, 4*planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(4*planes)
        ))

        self.layers = nn.Sequential(*[
            *[Bottleneck(4*planes, planes) for _ in range(num_layers - 2)]
        ])

        self.upsample = TransposedBottleneck(4*planes, planes, stride=2)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.layers(x)
        x = self.upsample(x)

        return x


class BottleneckUResNet(nn.Module):
    def __init__(self, resnet, layers):
        super(BottleneckUResNet, self).__init__()

        self.resnet = resnet

        self.conv_1 = nn.Conv2d(4*512, 512, kernel_size=1, stride=1)
        self.conv_2 = nn.Conv2d(4*256, 256, kernel_size=1, stride=1)
        self.conv_3 = nn.Conv2d(4*128, 128, kernel_size=1, stride=1)
        self.conv_4 = nn.Conv2d(4*64, 64, kernel_size=1, stride=1)

        self.de_1 = UResNetLayer(512, 256, layers[3])
        self.de_2 = UResNetLayer(256, 128, layers[2])
        self.de_3 = UResNetLayer(128, 64, layers[1])
        self.de_4 = UResNetLayer(64, 64, layers[0])

        self.classifier = nn.Sequential(*[
            nn.Conv2d(64, 1, kernel_size=1, bias=True),
        ])

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x_1 = x = self.resnet.layer1(x)
        x_2 = x = self.resnet.layer2(x)
        x_3 = x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.de_1(self.conv_1(x))
        x = self.de_2(x + self.conv_2(x_3))
        x = self.de_3(x + self.conv_3(x_2))
        x = self.de_4(x + self.conv_4(x_1))

        x = self.classifier(x)

        return F.upsample(x, scale_factor=2, mode='bilinear')
