import torch
from torch import nn
from torch.nn import functional as F

from .modules import CAM_Module, PAM_Module
from .refinenet import RefineNet, CRP, RCU, RefineNetUpsampleClassifier, RefineNetBlock, MrF, conv3x3


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
        cls_expanded = torch.sigmoid(cls).unsqueeze(2).unsqueeze(2).expand(x.size()[0], 1, x.size()[2], x.size()[3])
        x = self.classifier(torch.cat([x, cls_expanded], dim=1))

        return x, cls


class AuxDualHypercolumnCatRefineNet(RefineNet):
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
        super(AuxDualHypercolumnCatRefineNet, self).__init__(
            encoder,
            num_features,
            num_features_base,
            block_multiplier,
            crp,
            rcu,
            mrf,
            classifier,
            block)

        if num_features_base is None:
            num_features_base = [64, 128, 256, 512]

        channels = block_multiplier * num_features_base[3]
        self.cam = CAM_Module(channels)
        self.pam = PAM_Module(channels)
        self.conv_cam_1 = conv3x3(channels, channels)
        self.conv_cam_2 = conv3x3(channels, channels)
        self.conv_pam_1 = conv3x3(channels, channels)
        self.conv_pam_2 = conv3x3(channels, channels)

        self.aux_pam = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(channels, 1, 1))
        self.aux_cam = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(channels, 1, 1))

    def forward(self, x):
        x_0, x_1, x_2, x_3 = self.encoder(x)

        cam = self.conv_cam_1(x_3)
        cam = self.cam(cam)
        cam = self.conv_cam_2(cam)

        pam = self.conv_pam_1(x_3)
        pam = self.pam(pam)
        pam = self.conv_pam_2(pam)

        r_0 = self.refine_0([cam + pam])
        r_1 = self.refine_1([x_2, r_0])
        r_2 = self.refine_2([x_1, r_1])
        r_3 = self.refine_3([x_0, r_2])

        hypercolumn = torch.cat([
            F.interpolate(r_1, scale_factor=4, mode='bilinear'),
            r_3
        ], dim=1)

        x = self.classifier(hypercolumn)
        aux_pam = self.aux_pam(pam)
        aux_cam = self.aux_pam(cam)

        return x, aux_pam, aux_cam