import torch
from torch import nn
from torchvision.models.resnet import BasicBlock, Bottleneck, conv3x3

from nets.encoders.senet import SEModule, SEResNetBottleneck, SEResNeXtBottleneck
from nets.modules import CAM_Module, PAM_Module, SCSEBlock, ModifiedSCSEModule, DualSCSEModule, BaseOC_Context_Module, ASP_OC_Module, PreActivationBasicBlock, PreActivationBottleneckBlock, SpatialAttentionModule, DistanceAttentionModule


def conv_3x3(in_channels, out_channels, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias)


class RCU(nn.Module):
    multiplier = 1

    def __init__(self, in_channels, out_channels):
        super(RCU, self).__init__()
        self.block = BasicBlock(in_channels, out_channels)

    def forward(self, x):
        return self.block(x)


class PreActivationRCU(nn.Module):
    multiplier = 1

    def __init__(self, in_channels, out_channels):
        super(PreActivationRCU, self).__init__()
        self.block = PreActivationBasicBlock(in_channels, out_channels)

    def forward(self, x):
        return self.block(x)


class PreActivationBottleneckRCU(nn.Module):
    multiplier = 4

    def __init__(self, in_channels, out_channels):
        super(PreActivationBottleneckRCU, self).__init__()
        self.block = PreActivationBottleneckBlock(in_channels, out_channels)

    def forward(self, x):
        return self.block(x)


class DilatedPreActivationRCU(nn.Module):
    multiplier = 1

    def __init__(self, in_channels, out_channels):
        super(DilatedPreActivationRCU, self).__init__()
        self.block = PreActivationBasicBlock(in_channels, out_channels, dilation=3)

    def forward(self, x):
        return self.block(x)


class SERCU(nn.Module):
    multiplier = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, reduction=16):
        super(SERCU, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.se_module = SEModule(out_channels * self.multiplier, reduction=reduction)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SCSERCU(nn.Module):
    multiplier = 1

    def __init__(self, *args, **kwargs):
        super(SCSERCU, self).__init__()
        self.scse_block = SCSEBlock(*args, **kwargs)

    def forward(self, x):
        return self.scse_block(x)


class ELUSCSERCU(nn.Module):
    multiplier = 1

    def __init__(self, *args, **kwargs):
        super(ELUSCSERCU, self).__init__()
        self.scse_block = SCSEBlock(*args, activation=nn.ELU, **kwargs)

    def forward(self, x):
        return self.scse_block(x)


class ELUModifiedSCSERCU(nn.Module):
    multiplier = 1

    def __init__(self, *args, **kwargs):
        super(ELUModifiedSCSERCU, self).__init__()
        self.scse_block = SCSEBlock(*args, activation=nn.ELU, scse=ModifiedSCSEModule, **kwargs)

    def forward(self, x):
        return self.scse_block(x)


class BottleneckRCU(nn.Module):
    multiplier = 4

    def __init__(self, in_channels, out_channels):
        super(BottleneckRCU, self).__init__()
        self.block = Bottleneck(in_channels, out_channels)

    def forward(self, x):
        return self.block(x)


class SEBottleneckRCU(nn.Module):
    multiplier = 4

    def __init__(self, in_channels, out_channels):
        super(SEBottleneckRCU, self).__init__()
        self.block = SEResNetBottleneck(in_channels, out_channels, groups=1, reduction=16)

    def forward(self, x):
        return self.block(x)


class SENextBottleneckRCU(nn.Module):
    multiplier = 4

    def __init__(self, in_channels, out_channels):
        super(SENextBottleneckRCU, self).__init__()
        self.block = SEResNeXtBottleneck(in_channels, out_channels, groups=16, reduction=16)

    def forward(self, x):
        return self.block(x)


class MrF(nn.Module):
    def __init__(self, channels, scale_factors, align_corners=False):
        super(MrF, self).__init__()
        
        paths = []
        for s in scale_factors:
            paths.append(nn.Sequential(
                conv_3x3(channels, channels),
                nn.Upsample(scale_factor=s, mode='bilinear', align_corners=align_corners)
            ))

        self.paths = nn.ModuleList(paths)

    def forward(self, inputs):
        sum = None
        for inp, path in zip(inputs, self.paths):
            if sum is None:
                sum = path(inp)
            else:
                sum = sum + path(inp)

        return sum


class TransposedConvolutionMrF(nn.Module):
    def __init__(self, channels, scale_factors):
        super(TransposedConvolutionMrF, self).__init__()

        paths = []
        for s in scale_factors:
            if s == 1:
                paths.append(conv3x3(channels, channels))
            else:
                paths.append(nn.ConvTranspose2d(channels, channels, kernel_size=3, stride=s, padding=1, output_padding=1, bias=False))

        self.paths = nn.ModuleList(paths)

    def forward(self, inputs):
        sum = None
        for inp, path in zip(inputs, self.paths):
            if sum is None:
                sum = path(inp)
            else:
                sum = sum + path(inp)

        return sum


class DualGatedTranspose(nn.Module):
    def __init__(self, channels, scale):
        super(DualGatedTranspose, self).__init__()

        self.transpose = nn.ConvTranspose2d(channels, channels, kernel_size=3, stride=scale, padding=1, output_padding=1, bias=False)
        self.dual_gate = DualSCSEModule(channels)
        self.rcu = RCU(channels, channels)

    def forward(self, high, low):
        low = self.transpose(low)
        high = self.dual_gate(high, low)
        high = self.rcu(high)

        return high + low


class GatedTransposedConvolutionMrF(nn.Module):
    def __init__(self, channels, scale_factors):
        super(GatedTransposedConvolutionMrF, self).__init__()

        paths = []
        for s in scale_factors:
            if s == 1:
                paths.append(conv3x3(channels, channels))
            else:
                paths.append(DualGatedTranspose(channels, s))

        self.paths = nn.ModuleList(paths)

    def forward(self, inputs):
        sum = None
        for inp, path in zip(inputs, self.paths):
            if sum is None:
                sum = path(inp)
            else:
                sum = path(sum, inp)

        return sum


class CRP(nn.Module):
    def __init__(self, channels):
        super(CRP, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.conv_1 = conv_3x3(channels, channels)
        self.conv_2 = conv_3x3(channels, channels)
        self.conv_3 = conv_3x3(channels, channels)

    def forward(self, residual):
        x = self.pool(residual)
        x = self.conv_1(x)
        residual = residual + x

        x = self.pool(x)
        x = self.conv_2(x)
        residual = residual + x

        x = self.pool(x)
        x = self.conv_3(x)
        residual = residual + x

        return residual


class OriginalCRP(nn.Module):
    def __init__(self, channels):
        super(OriginalCRP, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.conv_1 = conv_3x3(channels, channels)
        self.conv_2 = conv_3x3(channels, channels)
        self.conv_3 = conv_3x3(channels, channels)

    def forward(self, residual):
        residual = self.relu(residual)

        x = self.pool(residual)
        x = self.conv_1(x)
        residual = residual + x

        x = self.pool(x)
        x = self.conv_2(x)
        residual = residual + x

        x = self.pool(x)
        x = self.conv_3(x)
        residual = residual + x

        return residual


class IdentityCRP(nn.Module):
    def __init__(self, channels):
        super(IdentityCRP, self).__init__()

    def forward(self, residual):

        return residual


class CRP4(nn.Module):
    def __init__(self, channels):
        super(CRP4, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.conv_1 = conv_3x3(channels, channels)
        self.conv_2 = conv_3x3(channels, channels)
        self.conv_3 = conv_3x3(channels, channels)
        self.conv_4 = conv_3x3(channels, channels)

    def forward(self, residual):
        x = self.pool(residual)
        x = self.conv_1(x)
        residual = residual + x

        x = self.pool(x)
        x = self.conv_2(x)
        residual = residual + x

        x = self.pool(x)
        x = self.conv_3(x)
        residual = residual + x

        x = self.pool(x)
        x = self.conv_4(x)
        residual = residual + x

        return residual


class AverageCRP(nn.Module):
    def __init__(self, channels):
        super(AverageCRP, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.conv_1 = conv_3x3(channels, channels)
        self.conv_2 = conv_3x3(channels, channels)
        self.conv_3 = conv_3x3(channels, channels)

    def forward(self, residual):
        x = self.pool(residual)
        x = self.conv_1(x)
        residual = residual + x

        x = self.pool(x)
        x = self.conv_2(x)
        residual = residual + x

        x = self.pool(x)
        x = self.conv_3(x)
        residual = residual + x

        return residual


class SpatialAttentionContext(nn.Module):
    def __init__(self, channels):
        super(SpatialAttentionContext, self).__init__()

        self.attention_module = SpatialAttentionModule(channels, size=(8, 8))

    def forward(self, x):
        return self.attention_module(x)


class DistanceCRP(nn.Module):
    def __init__(self, channels):
        super(DistanceCRP, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.conv_1 = conv_3x3(channels, channels)
        self.conv_2 = conv_3x3(channels, channels)
        self.conv_3 = conv_3x3(channels, channels)

        self.distance_attention = DistanceAttentionModule(channels)

    def forward(self, residual):
        x = self.pool(residual)
        x = self.conv_1(x)
        residual = residual + x

        x = self.pool(x)
        x = self.conv_2(x)
        residual = residual + x

        x = self.pool(x)
        x = self.conv_3(x)
        residual = residual + x

        x = self.distance_attention(residual)
        residual = x + residual

        return residual


class AspOC(nn.Module):
    def __init__(self, channels):
        super(AspOC, self).__init__()
        self.psab = ASP_OC_Module(channels, channels, dilations=(6, 12, 20))
        self.conv_bn = nn.Sequential(
            nn.Conv2d(2*channels, channels, kernel_size=1, stride=1, bias=False, padding=0),
            nn.BatchNorm2d(channels)
        )

    def forward(self, residual):
        x = self.psab(residual)
        x = torch.cat([x, residual], dim=1)
        x = self.conv_bn(x)

        return x


class OC(nn.Module):
    def __init__(self, channels):
        super(OC, self).__init__()
        self.oc = BaseOC_Context_Module(channels, channels, key_channels=channels // 2, value_channels=channels, sizes=[1, 2])
        self.conv = conv3x3(channels, channels)

    def forward(self, residual):
        x = self.conv(self.oc(residual))

        return x


class DualAttentation(nn.Module):
    def __init__(self, channels):
        super(DualAttentation, self).__init__()
        self.cam = CAM_Module(channels)
        self.pam = PAM_Module(channels)
        self.conv_cam_1 = conv3x3(channels, channels)
        self.conv_cam_2 = conv3x3(channels, channels)
        self.conv_pam_1 = conv3x3(channels, channels)
        self.conv_pam_2 = conv3x3(channels, channels)

    def forward(self, x):
        cam = self.conv_cam_1(x)
        cam = self.cam(cam)
        cam = self.conv_cam_2(cam)

        pam = self.conv_pam_1(x)
        pam = self.pam(pam)
        pam = self.conv_pam_2(pam)

        return cam + pam


class RefineNetUpsampleClassifier(nn.Module):
    def __init__(self, in_channels, channels=None, scale_factor=4, rcu=RCU, align_corners=False):
        super(RefineNetUpsampleClassifier, self).__init__()

        if channels is None:
            channels = in_channels

        self.classifier = nn.Sequential(*[
            rcu(rcu.multiplier * channels, channels),
            rcu(rcu.multiplier * channels, channels),
            nn.Conv2d(rcu.multiplier * channels, 1, kernel_size=1, bias=True),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=align_corners)
        ])

    def forward(self, x):
        return self.classifier(x)


class DropoutRefineNetUpsampleClassifier(nn.Module):
    def __init__(self, in_channels, channels=None, scale_factor=4, rcu=RCU, dropout=0):
        super(DropoutRefineNetUpsampleClassifier, self).__init__()

        if channels is None:
            channels = in_channels

        self.classifier = nn.Sequential(*[
            nn.Conv2d(in_channels, channels * rcu.multiplier, kernel_size=1, bias=False),
            rcu(rcu.multiplier * channels, channels),
            nn.Dropout2d(dropout),
            nn.Conv2d(rcu.multiplier * channels, 1, kernel_size=1, bias=True),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        ])

    def forward(self, x):
        return self.classifier(x)


class SmallDropoutRefineNetUpsampleClassifier(nn.Module):
    def __init__(self, channels, scale_factor=4, rcu=RCU, dropout=0):
        super(SmallDropoutRefineNetUpsampleClassifier, self).__init__()

        self.classifier = nn.Sequential(*[
            conv_3x3(rcu.multiplier * channels, rcu.multiplier * channels),
            nn.BatchNorm2d(rcu.multiplier * channels),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(rcu.multiplier * channels, 1, kernel_size=1, bias=True),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        ])

    def forward(self, x):
        return self.classifier(x)


class DADropoutRefineNetUpsampleClassifier(nn.Module):
    def __init__(self, channels, scale_factor=4, rcu=RCU, dropout=0):
        super(DADropoutRefineNetUpsampleClassifier, self).__init__()

        self.da = DualAttentation(channels)

        self.classifier = nn.Sequential(*[
            self.da,
            conv_3x3(rcu.multiplier * channels, rcu.multiplier * channels),
            nn.BatchNorm2d(rcu.multiplier * channels),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(rcu.multiplier * channels, 1, kernel_size=1, bias=True),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        ])

    def forward(self, x):
        return self.classifier(x)


class DistanceRefineNetUpsampleClassifier(nn.Module):
    def __init__(self, in_channels, channels=None, scale_factor=4, rcu=RCU):
        super(DistanceRefineNetUpsampleClassifier, self).__init__()

        if channels is None:
            channels = in_channels

        self.distance_module = DistanceAttentionModule(rcu.multiplier * channels)
        self.rcu1 = rcu(rcu.multiplier * channels, channels)
        self.rcu2 = rcu(rcu.multiplier * channels, channels)
        self.classifier = nn.Conv2d(rcu.multiplier * channels, 1, kernel_size=1, bias=True)
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear')

    def forward(self, x):
        x = x + self.distance_module(x)
        x = self.rcu1(x)
        x = self.rcu2(x)
        x = self.classifier(x)
        x = self.upsample(x)

        return x


class SmallOCRefineNetUpsampleClassifier(nn.Module):
    def __init__(self, channels, scale_factor=4, rcu=RCU, dropout=0):
        super(SmallOCRefineNetUpsampleClassifier, self).__init__()

        self.classifier = nn.Sequential(*[
            ASP_OC_Module(rcu.multiplier * channels, rcu.multiplier * channels),
            conv_3x3(rcu.multiplier * channels, rcu.multiplier * channels),
            nn.BatchNorm2d(rcu.multiplier * channels),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(rcu.multiplier * channels, 1, kernel_size=1, bias=True),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        ])

    def forward(self, x):
        return self.classifier(x)


class OCRefineNetUpsampleClassifier(nn.Module):
    def __init__(self, in_channels, channels=None, scale_factor=4, rcu=RCU):
        super(OCRefineNetUpsampleClassifier, self).__init__()

        if channels is None:
            channels = in_channels

        self.oc = nn.Sequential(
            ASP_OC_Module(rcu.multiplier * channels, rcu.multiplier * channels),
        )
        self.rcu1 = rcu(rcu.multiplier * channels, channels)
        self.rcu2 = rcu(rcu.multiplier * channels, channels)
        self.classifier = nn.Conv2d(rcu.multiplier * channels, 1, kernel_size=1, bias=True)
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear')

    def forward(self, x):
        x = x + self.oc(x)
        x = self.rcu1(x)
        x = self.rcu2(x)
        x = self.classifier(x)
        x = self.upsample(x)

        return x


class ModifiedRefineNetUpsampleClassifier(nn.Module):
    def __init__(self, in_channels, channels=None, scale_factor=4, rcu=RCU):
        super(ModifiedRefineNetUpsampleClassifier, self).__init__()

        if channels is None:
            channels = in_channels

        self.classifier = nn.Sequential(*[
            nn.Conv2d(in_channels, channels * rcu.multiplier, kernel_size=1, bias=False),
            rcu(rcu.multiplier * channels, channels),
            rcu(rcu.multiplier * channels, channels),
            nn.Conv2d(rcu.multiplier * channels, 1, kernel_size=1, bias=True),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        ])

    def forward(self, x):
        return self.classifier(x)


class OldRefineNetBlock(nn.Module):
    def __init__(self, channels, config, crp=CRP, rcu=RCU, mrf=MrF):
        super(OldRefineNetBlock, self).__init__()
        paths = []
        for in_channels, scale_factor in config:
            p = nn.Sequential(*[
                nn.Conv2d(in_channels, channels*rcu.multiplier, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Dropout2d(0),
                rcu(channels*rcu.multiplier, channels),
                rcu(channels*rcu.multiplier, channels)
            ])
            paths.append(p)

        self.paths = nn.ModuleList(paths)

        self.mrf = mrf(channels*rcu.multiplier, [scale_factor for in_channels, scale_factor in config])
        self.crp = crp(channels*rcu.multiplier)
        self.out = rcu(channels*rcu.multiplier, channels)

    def forward(self, inputs):
        paths = [path(inp) for inp, path in zip(inputs, self.paths)]
        x = self.mrf(paths)
        x = self.crp(x)
        x = self.out(x)

        return x


class RefineNetBlock(nn.Module):
    def __init__(self, channels, config, crp=CRP, rcu=RCU, mrf=MrF):
        super(RefineNetBlock, self).__init__()
        paths = []
        for in_channels, scale_factor in config:
            p = nn.Sequential(*[
                nn.Conv2d(in_channels, channels*rcu.multiplier, kernel_size=1, stride=1, padding=0, bias=False),
                rcu(channels*rcu.multiplier, channels),
                rcu(channels*rcu.multiplier, channels)
            ])
            paths.append(p)

        self.paths = nn.ModuleList(paths)

        self.mrf = mrf(channels*rcu.multiplier, [scale_factor for in_channels, scale_factor in config])
        self.crp = crp(channels*rcu.multiplier)
        self.out = rcu(channels*rcu.multiplier, channels)

    def forward(self, inputs):
        paths = [path(inp) for inp, path in zip(inputs, self.paths)]
        x = self.mrf(paths)
        x = self.crp(x)
        x = self.out(x)

        return x


class SCSERefineNetBlock(nn.Module):
    def __init__(self, channels, config, crp=CRP, rcu=RCU, mrf=MrF):
        super(SCSERefineNetBlock, self).__init__()
        paths = []
        for in_channels, scale_factor in config:
            p = nn.Sequential(*[
                nn.Conv2d(in_channels, channels*rcu.multiplier, kernel_size=1, stride=1, padding=0, bias=False),
                rcu(channels*rcu.multiplier, channels),
                SCSERCU(channels*rcu.multiplier, channels)
            ])
            paths.append(p)

        self.paths = nn.ModuleList(paths)

        self.mrf = mrf(channels*rcu.multiplier, [scale_factor for in_channels, scale_factor in config])
        self.crp = crp(channels*rcu.multiplier)
        self.out = rcu(channels*rcu.multiplier, channels)

    def forward(self, inputs):
        paths = [path(inp) for inp, path in zip(inputs, self.paths)]
        x = self.mrf(paths)
        x = self.crp(x)
        x = self.out(x)

        return x


class DropoutSCSERefineNetBlock(nn.Module):
    def __init__(self, channels, config, crp=CRP, rcu=RCU, mrf=MrF, dropout=0.1):
        super(DropoutSCSERefineNetBlock, self).__init__()
        paths = []
        for i, (in_channels, scale_factor) in enumerate(config):
            p = [
                nn.Conv2d(in_channels, channels*rcu.multiplier, kernel_size=1, stride=1, padding=0, bias=False)
            ]

            if i == 0:
                p += [nn.Dropout2d(dropout)]

            p += [
                rcu(channels*rcu.multiplier, channels),
                rcu(channels*rcu.multiplier, channels)
            ]

            p = nn.Sequential(*p)
            paths.append(p)

        self.paths = nn.ModuleList(paths)

        self.mrf = mrf(channels*rcu.multiplier, [scale_factor for in_channels, scale_factor in config])
        self.crp = crp(channels*rcu.multiplier)
        self.out = SCSERCU(channels*rcu.multiplier, channels)

    def forward(self, inputs):
        paths = [path(inp) for inp, path in zip(inputs, self.paths)]
        x = self.mrf(paths)
        x = self.crp(x)
        x = self.out(x)

        return x


class DeepRefineNetBlock(RefineNetBlock):
    def __init__(self, channels, config, crp=CRP, rcu=RCU, mrf=MrF):
        super(DeepRefineNetBlock, self).__init__(channels, config, crp, rcu, mrf)

        self.out = nn.Sequential(
            rcu(channels * rcu.multiplier, channels),
            rcu(channels * rcu.multiplier, channels),
            rcu(channels * rcu.multiplier, channels),
        )


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
            block=RefineNetBlock
    ):
        super(RefineNet, self).__init__()

        if num_features is None:
            num_features = [256*2, 256, 256, 256]

        if not isinstance(num_features, list):
            num_features = [num_features*2, num_features, num_features, num_features]

        if num_features_base is None:
            num_features_base = [64, 128, 256, 512]

        if not isinstance(crp, list):
            crp = [crp, crp, crp, crp]

        self.refine_0 = block(num_features[0], [(block_multiplier*num_features_base[3], 1)], crp[0], rcu, mrf)
        self.refine_1 = block(num_features[1], [(block_multiplier*num_features_base[2], 1), (num_features[0]*rcu.multiplier, 2)], crp[1], rcu, mrf)
        self.refine_2 = block(num_features[2], [(block_multiplier*num_features_base[1], 1), (num_features[1]*rcu.multiplier, 2)], crp[2], rcu, mrf)
        self.refine_3 = block(num_features[3], [(block_multiplier*num_features_base[0], 1), (num_features[2]*rcu.multiplier, 2)], crp[3], rcu, mrf)

        self.classifier = classifier(num_features[3])

        self.encoder = encoder

    def forward(self, x):
        x_0, x_1, x_2, x_3 = self.encoder(x)

        x = self.refine_0([x_3])
        x = self.refine_1([x_2, x])
        x = self.refine_2([x_1, x])
        x = self.refine_3([x_0, x])

        x = self.classifier(x)

        return x


class HighResRefineNet(nn.Module):
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
        super(HighResRefineNet, self).__init__()

        if num_features is None:
            num_features = [256*2, 256, 256, 256]

        if not isinstance(num_features, list):
            num_features = [num_features*2, num_features, num_features, num_features]

        if num_features_base is None:
            num_features_base = [64, 128, 256, 512]

        if not isinstance(crp, list):
            crp = [crp, crp, crp, crp]

        self.refine_0 = block(num_features[0], [(block_multiplier*num_features_base[3], 1)], crp[0], rcu, mrf)
        self.refine_1 = block(num_features[1], [(block_multiplier*num_features_base[2], 1), (num_features[0]*rcu.multiplier, 2)], crp[1], rcu, mrf)
        self.refine_2 = block(num_features[2], [(block_multiplier*num_features_base[1], 1), (num_features[1]*rcu.multiplier, 2)], crp[2], rcu, mrf)
        self.refine_3 = block(num_features[3], [(block_multiplier*num_features_base[0], 1), (num_features[2]*rcu.multiplier, 1)], crp[3], rcu, mrf)

        self.classifier = classifier(num_features[3])

        self.encoder = encoder

    def forward(self, x):
        x_0, x_1, x_2, x_3 = self.encoder(x)

        x = self.refine_0([x_3])
        x = self.refine_1([x_2, x])
        x = self.refine_2([x_1, x])
        x = self.refine_3([x_0, x])

        x = self.classifier(x)

        return x


class HighwayRefineNetBlock(nn.Module):
    def __init__(self, channels, channels_highway, config, rcu=RCU, rcu_highway=RCU, crp=CRP, mrf=MrF):
        super(HighwayRefineNetBlock, self).__init__()
        paths = []
        for in_channels, scale_factor in config:
            modules = []

            if in_channels != channels:
                modules = modules + [
                    rcu_highway(channels_highway * rcu_highway.multiplier, channels_highway),
                    rcu_highway(channels_highway * rcu_highway.multiplier, channels_highway)
                ]

            else:
                modules = modules + [
                    nn.Conv2d(in_channels, channels * rcu.multiplier, kernel_size=1, stride=1, padding=0, bias=False),
                    rcu(channels * rcu.multiplier, channels),
                    rcu(channels * rcu.multiplier, channels)
                ]

            if in_channels != channels_highway*rcu_highway.multiplier:
                modules.append(nn.Conv2d(channels*rcu.multiplier, channels_highway * rcu_highway.multiplier, kernel_size=1, stride=1, padding=0, bias=False))

            p = nn.Sequential(*modules)
            paths.append(p)

        self.paths = nn.ModuleList(paths)

        self.mrf = mrf(channels * rcu_highway.multiplier, [scale_factor for in_channels, scale_factor in config])
        self.crp = crp(channels * rcu_highway.multiplier)
        self.out = rcu_highway(channels_highway * rcu_highway.multiplier, channels_highway)

    def forward(self, inputs):
        paths = [path(inp) for inp, path in zip(inputs, self.paths)]
        x = self.mrf(paths)
        x = self.crp(x)
        x = self.out(x)

        return x


class HighwayRefineNet(nn.Module):
    def __init__(
            self,
            encoder,
            num_features=None,
            num_features_base=None,
            block_multiplier=4,
            crp=CRP,
            rcu=RCU,
            rcu_highway=BottleneckRCU,
            channels_highway=64,
            mrf=MrF,
            classifier=RefineNetUpsampleClassifier
    ):
        super(HighwayRefineNet, self).__init__()

        if num_features is None:
            num_features = [256*2, 256, 256, 256]

        if not isinstance(num_features, list):
            num_features = [num_features*2, num_features, num_features, num_features]

        if num_features_base is None:
            num_features_base = [64, 128, 256, 512]

        if not isinstance(crp, list):
            crp = [crp, crp, crp, crp]

        self.refine_0 = HighwayRefineNetBlock(num_features[0], channels_highway, [(block_multiplier*num_features_base[3], 1)],
                                              rcu, rcu_highway, crp[0], mrf)
        self.refine_1 = HighwayRefineNetBlock(
            num_features[1],
            channels_highway,
            [(block_multiplier*num_features_base[2], 1), (num_features[0]*rcu.multiplier, 2)],
            rcu,
            rcu_highway,
            crp[1],
            mrf
        )
        self.refine_2 = HighwayRefineNetBlock(
            num_features[2],
            channels_highway,
            [(block_multiplier*num_features_base[1], 1), (num_features[1]*rcu.multiplier, 2)],
            rcu,
            rcu_highway,
            crp[2],
            mrf
        )
        self.refine_3 = HighwayRefineNetBlock(
            num_features[3],
            channels_highway,
            [(block_multiplier*num_features_base[0], 1), (num_features[2]*rcu.multiplier, 2)],
            rcu,
            rcu_highway,
            crp[3],
            mrf
        )

        self.classifier = classifier(num_features[3]*rcu_highway.multiplier)

        self.encoder = encoder

    def forward(self, x):
        x_0, x_1, x_2, x_3 = self.encoder(x)

        x = self.refine_0([x_3])
        x = self.refine_1([x_2, x])
        x = self.refine_2([x_1, x])
        x = self.refine_3([x_0, x])

        x = self.classifier(x)

        return x


class DualScaleRefineNet(nn.Module):
    def __init__(
            self,
            encoder1,
            encoder2,
            num_features=None,
            num_features_base=None,
            block_multiplier=4,
            crp=CRP,
            rcu=RCU,
            mrf=MrF,
            classifier=RefineNetUpsampleClassifier,
            block=RefineNetBlock
    ):
        super(DualScaleRefineNet, self).__init__()

        if num_features is None:
            num_features = [256*2, 256, 256, 256]

        if not isinstance(num_features, list):
            num_features = [num_features*2, num_features, num_features, num_features]

        if num_features_base is None:
            num_features_base = [64, 128, 256, 512]

        self.scale = nn.AvgPool2d(2)

        self.refine_0 = block(num_features[0], [
            (block_multiplier * num_features_base[3], 1),
            (block_multiplier * num_features_base[3], 2),
            (block_multiplier * num_features_base[2], 1),
        ], crp, rcu, mrf)

        self.refine_1 = block(num_features[1], [
            (block_multiplier*num_features_base[2], 1),
            (num_features[0] * rcu.multiplier, 2),
            (block_multiplier*num_features_base[1], 1),
        ], crp, rcu, mrf)
        self.refine_2 = block(num_features[2], [(block_multiplier*num_features_base[1], 1), (num_features[1]*rcu.multiplier, 2)], crp, rcu, mrf)
        self.refine_3 = block(num_features[3], [(block_multiplier*num_features_base[0], 1), (num_features[2]*rcu.multiplier, 2)], crp, rcu, mrf)

        self.classifier = classifier(num_features[3])

        self.encoder1 = encoder1
        self.encoder2 = encoder2

    def forward(self, x):
        x_0, x_1, x_2, x_3 = self.encoder1(x)
        x_0_half, x_1_half, x_2_half, x_3_half = self.encoder2(x)

        x = self.refine_0([x_3, x_3_half, x_2_half])
        x = self.refine_1([x_2, x, x_1_half])
        x = self.refine_2([x_1, x])
        x = self.refine_3([x_0, x])

        x = self.classifier(x)

        return x


class DilatedPyramidPooling(nn.Module):
    def __init__(self, channels):
        super(DilatedPyramidPooling, self).__init__()

        self.conv_a = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.conv_b_1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv_b_2 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2)
        self.conv_c_1 = nn.Conv2d(channels, channels, kernel_size=5, padding=2)
        self.conv_c_2 = nn.Conv2d(channels, channels, kernel_size=5, padding=4, dilation=2)

    def forward(self, x):
        a = self.conv_a(x)
        b_1 = self.conv_b_1(x)
        b_2 = self.conv_b_2(x)
        c_1 = self.conv_c_1(x)
        c_2 = self.conv_c_2(x)

        return a + b_1 + b_2 + c_1 + c_2 + x


class PyramidPooling(nn.Module):
    def __init__(self, channels):
        super(PyramidPooling, self).__init__()

        self.conv_a = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.conv_b = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.conv_c = nn.Sequential(
            nn.AvgPool2d(kernel_size=4, stride=4),
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        a = self.conv_a(x)
        b = self.conv_b(x)
        c = self.conv_c(x)

        return a + b + c + x
