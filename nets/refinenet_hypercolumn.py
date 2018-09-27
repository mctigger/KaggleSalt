import torch
from torch.nn import functional as F

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
