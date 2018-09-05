from torch import nn
from torchvision.models.resnet import BasicBlock

from .modules import SCSEBlock


class BR(nn.Module):
    def __init__(self, channels):
        super(BR, self).__init__()

        self.block = SCSEBlock(channels, channels)

    def forward(self, x):
        return self.block(x)


class GCN(nn.Module):
    def __init__(self, in_channels, channels, kernel_size=7):
        super(GCN, self).__init__()

        self.conv_a_1 = nn.Conv2d(in_channels, channels, kernel_size=(kernel_size, 1), padding=(kernel_size // 2, 0), stride=1, bias=False)
        self.conv_a_2 = nn.Conv2d(channels, channels, kernel_size=(1, kernel_size), padding=(0, kernel_size // 2), stride=1, bias=False)

        self.conv_b_1 = nn.Conv2d(in_channels, channels, kernel_size=(1, kernel_size), padding=(0, kernel_size // 2), stride=1, bias=False)
        self.conv_b_2 = nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1), padding=(kernel_size // 2, 0), stride=1, bias=False)

    def forward(self, x):
        x_a = self.conv_a_1(x)
        x_a = self.conv_a_2(x_a)

        x_b = self.conv_a_1(x)
        x_b = self.conv_a_2(x_b)

        return x_a + x_b


class LargeKernelMattersNet(nn.Module):
    def __init__(self, base, num_features_base=None):
        super(LargeKernelMattersNet, self).__init__()

        if num_features_base is None:
            num_features_base = [256, 512, 1024, 2048]

        k = 128

        self.gcn_1 = GCN(num_features_base[0], k)
        self.gcn_2 = GCN(num_features_base[1], k)
        self.gcn_3 = GCN(num_features_base[2], k)
        self.gcn_4 = GCN(num_features_base[3], k, kernel_size=3)

        self.br_1_a = BR(k)
        self.br_1_b = BR(k)
        self.br_2_a = BR(k)
        self.br_2_b = BR(k)
        self.br_3_a = BR(k)
        self.br_3_b = BR(k)
        self.br_4 = BR(k)

        self.deconv_1 = nn.ConvTranspose2d(k, k, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.deconv_2 = nn.ConvTranspose2d(k, k, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.deconv_3 = nn.ConvTranspose2d(k, k, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.deconv_4 = nn.ConvTranspose2d(k, k, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)

        self.classifier = nn.Sequential(
            BR(k),
            nn.ConvTranspose2d(k, k, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            BR(k),
            nn.Conv2d(k, 1, kernel_size=1, stride=1, padding=0, bias=True)
        )

        self.base = base

    def forward(self, x):
        x_1, x_2, x_3, x_4 = self.base(x)

        x_1 = self.br_1_a(self.gcn_1(x_1))
        x_2 = self.br_2_a(self.gcn_2(x_2))
        x_3 = self.br_3_a(self.gcn_3(x_3))
        x_4 = self.br_4(self.gcn_4(x_4))

        x = self.deconv_4(x_4)
        x = self.deconv_3(self.br_3_b(x + x_3))
        x = self.deconv_2(self.br_2_b(x + x_2))
        x = self.deconv_1(self.br_1_b(x + x_1))

        x = self.classifier(x)

        return x
