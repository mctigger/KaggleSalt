from torch import nn as nn, nn

from nets.encoders.dpn import CatBnAct


class DPNBase(nn.Module):
    def __init__(self, dpn):
        super(DPNBase, self).__init__()
        idx = 1
        self.layer0 = dpn.features[0]

        idx_layer = idx + dpn.k_sec[0]
        self.layer1 = nn.Sequential(*dpn.features[idx:idx_layer])
        self.out1 = CatBnAct(256 + 80)
        idx = idx_layer

        idx_layer = idx + dpn.k_sec[1]
        self.layer2 = nn.Sequential(*dpn.features[idx:idx_layer])
        self.out2 = CatBnAct(512+192)
        idx = idx_layer

        idx_layer = idx + dpn.k_sec[2]
        self.layer3 = nn.Sequential(*dpn.features[idx:idx_layer])
        self.out3 = CatBnAct(1024+528)
        idx = idx_layer

        idx_layer = idx + dpn.k_sec[3]
        self.layer4 = nn.Sequential(*dpn.features[idx:idx_layer])
        self.out4 = CatBnAct(2048 + 640)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x1 = self.out1(x)
        x = self.layer2(x)
        x2 = self.out2(x)
        x = self.layer3(x)
        x3 = self.out3(x)
        x = self.layer4(x)
        x4 = self.out4(x)

        return x1, x2, x3, x4


class NoPoolDPNBase(nn.Module):
    def __init__(self, dpn):
        super(NoPoolDPNBase, self).__init__()
        idx = 1
        self.layer0 = nn.Sequential(
            dpn.features[0].conv,
            dpn.features[0].bn,
            dpn.features[0].act,
        )

        idx_layer = idx + dpn.k_sec[0]
        self.layer1 = nn.Sequential(*dpn.features[idx:idx_layer])
        self.out1 = CatBnAct(256 + 80)
        idx = idx_layer

        idx_layer = idx + dpn.k_sec[1]
        self.layer2 = nn.Sequential(*dpn.features[idx:idx_layer])
        self.out2 = CatBnAct(512+192)
        idx = idx_layer

        idx_layer = idx + dpn.k_sec[2]
        self.layer3 = nn.Sequential(*dpn.features[idx:idx_layer])
        self.out3 = CatBnAct(1024+528)
        idx = idx_layer

        idx_layer = idx + dpn.k_sec[3]
        self.layer4 = nn.Sequential(*dpn.features[idx:idx_layer])
        self.out4 = CatBnAct(2048 + 640)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x1 = self.out1(x)
        x = self.layer2(x)
        x2 = self.out2(x)
        x = self.layer3(x)
        x3 = self.out3(x)
        x = self.layer4(x)
        x4 = self.out4(x)

        return x1, x2, x3, x4


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


class NoPoolResNetBase(nn.Module):
    def __init__(self, resnet):
        super(NoPoolResNetBase, self).__init__()
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
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


class DRNBase(nn.Module):
    def __init__(self, resnet):
        super(DRNBase, self).__init__()
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