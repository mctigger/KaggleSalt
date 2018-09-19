from torch import nn

from .refinenet import CRP, RCU, RefineNetUpsampleClassifier, RefineNetBlock, MrF


class RefineNet(nn.Module):
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
            dropout=0
    ):
        super(RefineNet, self).__init__()

        if num_features is None:
            num_features = [256*2, 256, 256, 256]

        if not isinstance(num_features, list):
            num_features = [num_features*2, num_features, num_features, num_features]

        if num_features_base is None:
            num_features_base = [64, 128, 256, 512]

        self.refine_0 = RefineNetBlock(num_features[0], [(block_multiplier*num_features_base[3], 1)], crp=crp, rcu=rcu, mrf=mrf, dropout=dropout)
        self.refine_1 = RefineNetBlock(num_features[1], [(block_multiplier*num_features_base[2], 1), (num_features[0]*rcu.multiplier, 2)], crp=crp, rcu=rcu, mrf=mrf, dropout=dropout)
        self.refine_2 = RefineNetBlock(num_features[2], [(block_multiplier*num_features_base[1], 1), (num_features[1]*rcu.multiplier, 2)], crp=crp, rcu=rcu, mrf=mrf, dropout=dropout)
        self.refine_3 = RefineNetBlock(num_features[3], [(block_multiplier*num_features_base[0], 1), (num_features[2]*rcu.multiplier, 2)], crp=crp, rcu=rcu, mrf=mrf, dropout=dropout)

        self.aux = classifier(num_features[1])
        self.classifier = classifier(num_features[3])

        self.encoder = encoder

    def forward(self, x):
        x_0, x_1, x_2, x_3 = self.encoder(x)

        x = self.refine_0([x_3])
        x = aux = self.refine_1([x_2, x])
        x = self.refine_2([x_1, x])
        x = self.refine_3([x_0, x])

        aux = self.aux(aux)
        x = self.classifier(x)

        return x, aux