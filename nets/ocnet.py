import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import conv3x3, BasicBlock

from .modules import ASP_OC_Module, BaseOC_Module


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation * multi_grid, dilation=dilation * multi_grid, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def _sum_each(self, x, y):
        assert (len(x) == len(y))
        z = []
        for i in range(len(x)):
            z.append(x[i] + y[i])
        return z

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, context, aux=False):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, multi_grid=(1, 1, 1))

        self.up0 = nn.Sequential(
            BasicBlock(128, 128),
        )

        self.up1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            BasicBlock(128, 128),
            BasicBlock(128, 128),
        )

        self.merge0 = nn.Sequential(
            BasicBlock(128, 128),
            BasicBlock(128, 128),
        )

        self.merge1 = nn.Sequential(
            BasicBlock(128, 128),
            BasicBlock(128, 128),
        )

        # extra added layers
        self.context = context
        self.cls = nn.Sequential(
            BasicBlock(128, 128),
            BasicBlock(128, 128),
            nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0, bias=True)
        )

        self.aux = nn.Sequential(
            ASP_OC_Module(1024, 128),
            BasicBlock(128, 128),
            BasicBlock(128, 128),
            nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=True))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        up0 = self.up0(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        up1 = self.up1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        aux = self.aux(x)
        x = self.layer4(x)
        x = self.context(x)

        x = self.merge1(up1 + F.upsample(x, scale_factor=2, mode='bilinear'))
        x = self.merge0(up0 + F.upsample(x, scale_factor=2, mode='bilinear'))

        x = self.cls(x)

        return F.upsample(x, scale_factor=2, mode='bilinear'), aux


def get_resnet101_asp_oc():
    context = nn.Sequential(
        ASP_OC_Module(2048, 128),
    )
    model = ResNet(Bottleneck, [3, 4, 23, 3], context)

    saved_state_dict = torch.load('./data/models/oc-net-resnet101-imagenet.pth')
    new_params = model.state_dict().copy()

    for i in saved_state_dict:
        i_parts = i.split('.')
        if not i_parts[0] == 'fc' and not i_parts[0] == 'last_linear' and not i_parts[0] == 'classifier':
            new_params['.'.join(i_parts[0:])] = saved_state_dict[i]

    model.load_state_dict(new_params)

    return model


def get_resnet101_asp_oc_aux():
    context = nn.Sequential(
        ASP_OC_Module(2048, 128),
    )
    model = ResNet(Bottleneck, [3, 4, 23, 3], context, True)

    saved_state_dict = torch.load('./data/models/oc-net-resnet101-imagenet.pth')
    new_params = model.state_dict().copy()

    for i in saved_state_dict:
        i_parts = i.split('.')
        if not i_parts[0] == 'fc' and not i_parts[0] == 'last_linear' and not i_parts[0] == 'classifier':
            new_params['.'.join(i_parts[0:])] = saved_state_dict[i]

    model.load_state_dict(new_params)

    return model


def get_resnet101_oc():
    context = nn.Sequential(
        nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(512),
        BaseOC_Module(in_channels=512, out_channels=512, key_channels=256, value_channels=256,
                      dropout=0.05, sizes=([1]))
    )
    model = ResNet(Bottleneck, [3, 4, 23, 3], context=context)

    saved_state_dict = torch.load('./data/models/oc-net-resnet101-imagenet.pth')
    new_params = model.state_dict().copy()

    for i in saved_state_dict:
        i_parts = i.split('.')
        if not i_parts[0] == 'fc' and not i_parts[0] == 'last_linear' and not i_parts[0] == 'classifier':
            new_params['.'.join(i_parts[0:])] = saved_state_dict[i]

    model.load_state_dict(new_params)

    return model
