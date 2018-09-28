import torch
from torch import nn
from torch.nn import functional as F

from .refinenet import RefineNet, CRP, RCU, RefineNetUpsampleClassifier, RefineNetBlock, MrF


class AuxRefineNet(RefineNet):
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
        super(AuxRefineNet, self).__init__(
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

        self.classifier_binary = nn.Linear(num_features_base[3], 1)

    def forward(self, x):
        x_0, x_1, x_2, x_3 = self.encoder(x)

        x = self.refine_0([x_3])
        x = self.refine_1([x_2, x])
        x = self.refine_2([x_1, x])
        x = self.refine_3([x_0, x])

        cls = self.classifier_binary(F.adaptive_max_pool2d(x_3, output_size=1).view(x.size(0), -1))
        cls_expanded = torch.sigmoid(cls).unsqueeze(2).unsqueeze(2).expand_as(x)
        x = self.classifier(torch.cat([x, cls_expanded], dim=1))

        return x, cls