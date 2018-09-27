from torch import nn
from torch.nn import functional as F

from .refinenet import RefineNet, CRP, RCU, RefineNetUpsampleClassifier, RefineNetBlock, MrF


class Decoder(nn.Module):
    def __init__(self, channels):
        super(Decoder, self).__init__()

        self.transition0 = nn.Conv2d(channels, channels // 2, kernel_size=1, bias=False)
        self.block0 = nn.Sequential(
            RCU(channels // 2, channels // 2),
            RCU(channels // 2, channels // 2)
        )

        self.transition1 = nn.Conv2d(channels // 2, channels // 4, kernel_size=1, bias=False)
        self.block1 = nn.Sequential(
            RCU(channels // 4, channels // 4),
            RCU(channels // 4, channels // 4)
        )

        self.transition2 = nn.Conv2d(channels // 4, channels // 8, kernel_size=1, bias=False)
        self.block2 = nn.Sequential(
            RCU(channels // 8, channels // 8),
            RCU(channels // 8, channels // 8)
        )

        self.transition3 = nn.Conv2d(channels // 8, channels // 16, kernel_size=1, bias=False)
        self.block3 = nn.Sequential(
            RCU(channels // 16, channels // 16),
            RCU(channels // 16, channels // 16)
        )

        self.reconstruct = nn.Conv2d(channels // 16, 3, kernel_size=1, bias=True)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.transition0(x)
        x = self.block0(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.transition1(x)
        x = self.block1(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.transition2(x)
        x = self.block2(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.transition3(x)
        x = self.block3(x)

        x = self.reconstruct(x)

        return x


class SemisupervisedRefineNet(RefineNet):
    def __init__(
            self,
            encoder,
            num_features=None,
            num_features_base=None,
            block_multiplier=4,
            crp=CRP,
            rcu=RCU,
            mrf=MrF,
            classifier=RefineNetUpsampleClassifier,
            block=RefineNetBlock
    ):
        super(SemisupervisedRefineNet, self).__init__(
            encoder,
            num_features,
            num_features_base,
            block_multiplier,
            crp,
            rcu,
            mrf,
            classifier,
            block
        )

        self.decoder = Decoder(512)

    def forward(self, x):
        x_0, x_1, x_2, x_3 = self.encoder(x)

        x = self.refine_0([x_3])
        x = self.refine_1([x_2, x])
        x = self.refine_2([x_1, x])
        x = self.refine_3([x_0, x])

        x = self.classifier(x)

        return x, self.decoder(x_3)