import torch
from torch.nn import functional as F
from torch import nn

from .refinenet import RefineNetUpsampleClassifier, RCU, CRP, RefineNetBlock, RefineNet, MrF


class HypercolumnCatRefineNet(RefineNet):
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
        super(HypercolumnCatRefineNet, self).__init__(
            encoder,
            num_features,
            num_features_base,
            block_multiplier,
            crp,
            rcu,
            mrf,
            classifier,
            block)

    def forward(self, x):
        x_0, x_1, x_2, x_3 = self.encoder(x)

        r_0 = self.refine_0([x_3])
        r_1 = self.refine_1([x_2, r_0])
        r_2 = self.refine_2([x_1, r_1])
        r_3 = self.refine_3([x_0, r_2])

        hypercolumn = torch.cat([
            F.interpolate(r_0, scale_factor=8, mode='bilinear'),
            F.interpolate(r_1, scale_factor=4, mode='bilinear'),
            F.interpolate(r_2, scale_factor=2, mode='bilinear'),
            r_3
        ], dim=1)

        x = self.classifier(hypercolumn)

        return x


class AuxHypercolumnCatRefineNet(RefineNet):
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
        super(AuxHypercolumnCatRefineNet, self).__init__(
            encoder,
            num_features,
            num_features_base,
            block_multiplier,
            crp,
            rcu,
            mrf,
            classifier,
            block)


        self.cls = nn.Linear(num_features_base[3], 1)

        channels = 320
        self.segmentation = nn.Sequential(*[
            nn.Conv2d(640, channels * rcu.multiplier, kernel_size=1, bias=False),
            rcu(rcu.multiplier * channels, channels),
        ])
        self.segmentation_logit = nn.Conv2d(rcu.multiplier * channels, 1, kernel_size=1, bias=True)

        self.fuse = nn.Sequential(*[
            nn.Conv2d(321, channels * rcu.multiplier, kernel_size=1, bias=False),
            rcu(rcu.multiplier * channels, channels),
        ])
        self.fuse_logit = nn.Conv2d(rcu.multiplier * channels, 1, kernel_size=1, bias=True)


    def forward(self, x):
        x_0, x_1, x_2, x_3 = self.encoder(x)

        r_0 = self.refine_0([x_3])
        r_1 = self.refine_1([x_2, r_0])
        r_2 = self.refine_2([x_1, r_1])
        r_3 = self.refine_3([x_0, r_2])

        hypercolumn = torch.cat([
            F.interpolate(r_0, scale_factor=8, mode='bilinear'),
            F.interpolate(r_1, scale_factor=4, mode='bilinear'),
            F.interpolate(r_2, scale_factor=2, mode='bilinear'),
            r_3
        ], dim=1)

        aux_cls = self.cls(F.adaptive_avg_pool2d(x_3, output_size=1).view(x.size(0), -1))
        cls_expanded = torch.sigmoid(aux_cls).unsqueeze(2).unsqueeze(2).expand(x.size()[0], 1, hypercolumn.size()[2], hypercolumn.size()[3])

        segmentation = self.segmentation(hypercolumn)
        aux_segmentation = F.interpolate(segmentation, scale_factor=2, mode='bilinear')
        aux_segmentation = self.segmentation_logit(aux_segmentation)

        x = self.fuse(torch.cat([segmentation, cls_expanded], dim=1))
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.fuse_logit(x)

        return x, aux_segmentation, aux_cls


class HypercolumnAddRefineNet(RefineNet):
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
        super(HypercolumnAddRefineNet, self).__init__(
            encoder,
            num_features,
            num_features_base,
            block_multiplier,
            crp,
            rcu,
            mrf,
            classifier,
            block)

    def forward(self, x):
        x_0, x_1, x_2, x_3 = self.encoder(x)

        r_0 = self.refine_0([x_3])
        r_1 = self.refine_1([x_2, r_0])
        r_2 = self.refine_2([x_1, r_1])
        r_3 = self.refine_3([x_0, r_2])

        hypercolumn = \
            F.interpolate(r_0, scale_factor=8, mode='bilinear') + \
            F.interpolate(r_1, scale_factor=4, mode='bilinear') + \
            F.interpolate(r_2, scale_factor=2, mode='bilinear') + \
            r_3

        x = self.classifier(hypercolumn)

        return x
