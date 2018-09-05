import torch
from torch import nn
from torch.nn import functional as F


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features // 4, size) for size in sizes])
        self.bottleneck_entry = nn.Conv2d(features, features // 4, kernel_size=1)
        self.bottleneck = nn.Conv2d(features // 4 * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        feats = self.bottleneck_entry(feats)
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class AtrousPSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super(AtrousPSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features // 4, size) for size in sizes])
        self.bottleneck_entry = nn.Conv2d(features, features // 4, kernel_size=1)
        self.bottleneck = nn.Conv2d(features // 4 * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        conv = nn.Conv2d(features, features, kernel_size=3, padding=size, dilation=size, bias=False)
        return conv

    def forward(self, feats):
        feats = self.bottleneck_entry(feats)
        priors = [stage(feats) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        p = F.upsample(input=x, scale_factor=2, mode='bilinear')
        return self.conv(p)


class PSPNet(nn.Module):
    def __init__(self, base, num_features=None, num_features_base=None, module=PSPModule):
        super().__init__()

        if num_features is None:
            num_features = [128, 128, 128, 128]

        if num_features_base is None:
            num_features_base = [256, 512, 1024, 2048]

        self.feats = base
        self.psp1 = module(num_features_base[0], num_features[0], (1, 4, 8, 16))
        self.psp2 = module(num_features_base[1], num_features[1], (1, 4, 8, 16))
        self.psp3 = module(num_features_base[2], num_features[2], (1, 2, 4, 8))
        self.psp4 = module(num_features_base[3], num_features[3], (1, 2, 3, 6))

        self.up_1 = PSPUpsample(128, num_features[3])
        self.up_2 = PSPUpsample(128, num_features[2])
        self.up_3 = PSPUpsample(128, num_features[1])
        self.up_4 = PSPUpsample(128, num_features[0])

        self.final = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
        )

    def forward(self, x):
        x1, x2, x3, x4 = self.feats(x)
        p1 = self.psp1(x1)
        p2 = self.psp2(x2)
        p3 = self.psp3(x3)
        p4 = self.psp4(x4)

        p = self.up_1(p4)
        p = self.up_2(p + p3)
        p = self.up_3(p + p2)
        p = self.up_4(p + p1)

        return F.upsample(self.final(p), scale_factor=2, mode='bilinear')