from torch import nn

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

        self.aux0 = nn.Conv2d(num_features[3], 1, kernel_size=1, bias=True)
        self.aux1 = nn.Conv2d(num_features[2], 1, kernel_size=1, bias=True)
        self.aux2 = nn.Conv2d(num_features[1], 1, kernel_size=1, bias=True)
        self.aux3 = nn.Conv2d(num_features[0], 1, kernel_size=1, bias=True)

    def forward(self, x):
        x_0, x_1, x_2, x_3 = self.encoder(x)

        aux0 = self.aux0(x_3)
        x = self.refine_0([x_3])
        aux1 = self.aux1(x)
        x = self.refine_1([x_2, x])
        aux2 = self.aux2(x)
        x = self.refine_2([x_1, x])
        aux3 = self.aux3(x)
        x = self.refine_3([x_0, x])

        x = self.classifier(x)

        return x, (aux0, aux1, aux2, aux3)