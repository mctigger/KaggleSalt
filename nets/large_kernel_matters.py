import torch
from torch import nn
from torchvision.models.resnet import BasicBlock


class BR(nn.Module):
    def __init__(self, channels):
        super(BR, self).__init__()

        self.block = BasicBlock(channels, channels)

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
    def __init__(self, resnet):
        super(LargeKernelMattersNet, self).__init__()
        k = 64
        self.resnet = resnet

        self.gcn_1 = GCN(4*64, k)
        self.gcn_2 = GCN(4*128, k)
        self.gcn_3 = GCN(4*256, k)
        self.gcn_4 = GCN(4*512, k, kernel_size=3)

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

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x_1 = self.resnet.layer1(x)
        x_2 = self.resnet.layer2(x_1)
        x_3 = self.resnet.layer3(x_2)
        x_4 = self.resnet.layer4(x_3)

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
