import torch
from torch import nn
from torch.nn import functional as F

from nets.encoders.dpn import CatBnAct
from nets.modules import SCSEModule


class DPN92Base(nn.Module):
    def __init__(self, dpn):
        super(DPN92Base, self).__init__()
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


class NoPoolDPN68Base(nn.Module):
    def __init__(self, dpn):
        super(NoPoolDPN68Base, self).__init__()
        idx = 1
        self.layer0 = nn.Sequential(
            dpn.features[0].conv,
            dpn.features[0].bn,
            dpn.features[0].act,
        )

        idx_layer = idx + dpn.k_sec[0]
        self.layer1 = nn.Sequential(*dpn.features[idx:idx_layer])
        self.out1 = CatBnAct(144)
        idx = idx_layer

        idx_layer = idx + dpn.k_sec[1]
        self.layer2 = nn.Sequential(*dpn.features[idx:idx_layer])
        self.out2 = CatBnAct(320)
        idx = idx_layer

        idx_layer = idx + dpn.k_sec[2]
        self.layer3 = nn.Sequential(*dpn.features[idx:idx_layer])
        self.out3 = CatBnAct(704)
        idx = idx_layer

        idx_layer = idx + dpn.k_sec[3]
        self.layer4 = nn.Sequential(*dpn.features[idx:idx_layer])
        self.out4 = CatBnAct(832)

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


class NoPoolDPN92Base(nn.Module):
    def __init__(self, dpn):
        super(NoPoolDPN92Base, self).__init__()
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


class NoPoolDPN98Base(nn.Module):
    def __init__(self, dpn):
        super(NoPoolDPN98Base, self).__init__()
        idx = 1
        self.layer0 = nn.Sequential(
            dpn.features[0].conv,
            dpn.features[0].bn,
            dpn.features[0].act,
        )

        idx_layer = idx + dpn.k_sec[0]
        self.layer1 = nn.Sequential(*dpn.features[idx:idx_layer])
        self.out1 = CatBnAct(336)
        idx = idx_layer

        idx_layer = idx + dpn.k_sec[1]
        self.layer2 = nn.Sequential(*dpn.features[idx:idx_layer])
        self.out2 = CatBnAct(768)
        idx = idx_layer

        idx_layer = idx + dpn.k_sec[2]
        self.layer3 = nn.Sequential(*dpn.features[idx:idx_layer])
        self.out3 = CatBnAct(1728)
        idx = idx_layer

        idx_layer = idx + dpn.k_sec[3]
        self.layer4 = nn.Sequential(*dpn.features[idx:idx_layer])
        self.out4 = CatBnAct(2688)

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


class NoPoolDPN107Base(nn.Module):
    def __init__(self, dpn):
        super(NoPoolDPN107Base, self).__init__()
        idx = 1
        self.layer0 = nn.Sequential(
            dpn.features[0].conv,
            dpn.features[0].bn,
            dpn.features[0].act,
        )

        idx_layer = idx + dpn.k_sec[0]
        self.layer1 = nn.Sequential(*dpn.features[idx:idx_layer])
        self.out1 = CatBnAct(376)
        idx = idx_layer

        idx_layer = idx + dpn.k_sec[1]
        self.layer2 = nn.Sequential(*dpn.features[idx:idx_layer])
        self.out2 = CatBnAct(1152)
        idx = idx_layer

        idx_layer = idx + dpn.k_sec[2]
        self.layer3 = nn.Sequential(*dpn.features[idx:idx_layer])
        self.out3 = CatBnAct(2432)
        idx = idx_layer

        idx_layer = idx + dpn.k_sec[3]
        self.layer4 = nn.Sequential(*dpn.features[idx:idx_layer])
        self.out4 = CatBnAct(2688)

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


class SCSENoPoolResNetBase(NoPoolResNetBase):
    def __init__(self, resnet):
        super(SCSENoPoolResNetBase, self).__init__(resnet)

        self.scse0 = SCSEModule(64)
        self.scse1 = SCSEModule(64)
        self.scse2 = SCSEModule(128)
        self.scse3 = SCSEModule(256)
        self.scse4 = SCSEModule(512)

    def forward(self, x):
        x_0 = self.scse0(self.layer0(x))
        x_1 = self.scse1(self.layer1(x_0))
        x_2 = self.scse2(self.layer2(x_1))
        x_3 = self.scse3(self.layer3(x_2))
        x_4 = self.scse4(self.layer4(x_3))

        return x_1, x_2, x_3, x_4


class NoPoolDenseNetBase(nn.Module):
    def __init__(self, resnet):
        super(NoPoolDenseNetBase, self).__init__()
        print(resnet)
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


class ConcatNoPoolResNetBase(nn.Module):
    def __init__(self, resnet):
        super(ConcatNoPoolResNetBase, self).__init__()
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

        x_2 = torch.cat([x_2, F.avg_pool2d(x_1, kernel_size=2)], dim=1)
        x_3 = torch.cat([x_3, F.avg_pool2d(x_2, kernel_size=2)], dim=1)
        x_4 = torch.cat([x_4, F.avg_pool2d(x_3, kernel_size=2)], dim=1)

        return x_1, x_2, x_3, x_4


class NoPoolResNextBase(nn.Module):
    def __init__(self, resnet):
        super(NoPoolResNextBase, self).__init__()
        self.layer0 = nn.Sequential(
            resnet.layer0.conv1,
            resnet.layer0.bn1,
            resnet.layer0.relu1,
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


class SCSENoPoolResNextBase(NoPoolResNextBase):
    def __init__(self, resnext):
        super(SCSENoPoolResNextBase, self).__init__(resnext)
        
        self.scse0 = SCSEModule(64)
        self.scse1 = SCSEModule(256)
        self.scse2 = SCSEModule(512)
        self.scse3 = SCSEModule(1024)
        self.scse4 = SCSEModule(2048)

    def forward(self, x):
        x_0 = self.scse0(self.layer0(x))
        x_1 = self.scse1(self.layer1(x_0))
        x_2 = self.scse2(self.layer2(x_1))
        x_3 = self.scse3(self.layer3(x_2))
        x_4 = self.scse4(self.layer4(x_3))

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