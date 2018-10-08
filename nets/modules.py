import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)


class PreActivationBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(PreActivationBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x = x + residual

        return x


class PreActivationBottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActivationBottleneckBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)

        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv3(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x = x + residual

        return x


class SpatialAttentionModule(nn.Module):
    def __init__(self, channels, size):
        super(SpatialAttentionModule, self).__init__()

        self.downsample = nn.AdaptiveAvgPool2d(size)
        self.squeeze = nn.Conv2d(channels, 1, kernel_size=1, bias=False)

        self.mlp = nn.Sequential(
            nn.BatchNorm1d(size ** 2),
            nn.ReLU(),
            nn.Linear(size ** 2, size ** 2, bias=False),
            nn.BatchNorm1d(size ** 2),
            nn.ReLU(),
            nn.Linear(size ** 2, size ** 2, bias=False)
        )
        self.scale = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        inp = x
        x = self.downsample(x)
        x = self.mlp(x.view(b, -1)).view(b, 1, h, w)
        x = F.interpolate(x, size=(h, w), mode='bilinear')
        x = self.scale(x)
        x = x * inp

        return x


class DistanceAttentionModule(nn.Module):
    def __init__(self, channels):
        super(DistanceAttentionModule, self).__init__()

        self.conv = conv3x3(channels + 1, channels)

        self.relu = nn.ReLU()

    def create_distance_tensor(self, size):
        horizontal = torch.cat([torch.arange(size // 2, 0, -1), torch.arange(1, size // 2 + 1)])\
            .unsqueeze(0)\
            .expand(size, size)\
            .float()

        vertical = torch.cat([torch.arange(size // 2, 0, -1), torch.arange(1, size // 2 + 1)])\
            .unsqueeze(1)\
            .expand(size,size)\
            .float()

        distance = torch.sqrt(vertical * vertical + horizontal * horizontal)
        normalized_distance = distance / torch.max(distance)

        return normalized_distance

    def forward(self, x):
        distance = self.create_distance_tensor(x.size(2)).unsqueeze(0).unsqueeze(0).expand(x.size(0), -1, -1, -1).to(x.device)
        x = torch.cat([x, distance], dim=1)
        x = self.conv(x)
        return x


class SCSEModule(nn.Module):
    def __init__(self, channel, reduction=16, activation=nn.ReLU):
        super(SCSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel // reduction)),
                                                activation(inplace=True),
                                                nn.Linear(int(channel // reduction), channel),
                                                nn.Sigmoid())

        self.spatial_se = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                        nn.Sigmoid())

    def forward(self, x):
        bahs, chs, _, _ = x.size()

        # Returns a new tensor with the same data as the self tensor but of a different size.
        chn_se = self.avg_pool(x).view(bahs, chs)
        chn_se = self.channel_excitation(chn_se).view(bahs, chs, 1, 1)
        chn_se = x * chn_se

        spa_se = self.spatial_se(x)
        spa_se = x * spa_se
        return chn_se + spa_se



class ModifiedSCSEModule(nn.Module):
    def __init__(self, channel, reduction=16, activation=nn.ReLU):
        super(ModifiedSCSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel // reduction)),
                                                activation(inplace=True),
                                                nn.Linear(int(channel // reduction), channel),
                                                nn.Sigmoid())

        self.spatial_se = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                        nn.Sigmoid())

    def forward(self, x):
        bahs, chs, _, _ = x.size()

        # Returns a new tensor with the same data as the self tensor but of a different size.
        chn_se = self.avg_pool(x).view(bahs, chs)
        chn_se = self.channel_excitation(chn_se).view(bahs, chs, 1, 1)

        spa_se = self.spatial_se(x)
        return x * chn_se * spa_se



class DualSCSEModule(nn.Module):
    def __init__(self, channel, reduction=16, activation=nn.ReLU):
        super(DualSCSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel // reduction)),
                                                activation(inplace=True),
                                                nn.Linear(int(channel // reduction), channel),
                                                nn.Sigmoid())

        self.spatial_se = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                        nn.Sigmoid())

    def forward(self, high, low):
        bahs, chs, _, _ = low.size()

        # Returns a new tensor with the same data as the self tensor but of a different size.
        chn_se = self.avg_pool(low).view(bahs, chs)
        chn_se = self.channel_excitation(chn_se).view(bahs, chs, 1, 1)
        chn_se = high * chn_se

        spa_se = self.spatial_se(low)
        spa_se = high * spa_se
        return chn_se + spa_se


class SCSEBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, activation=nn.ReLU, scse=SCSEModule):
        super(SCSEBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = activation(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.scse_module = scse(planes, reduction=reduction, activation=activation)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.scse_module(out) + residual
        out = self.relu(out)

        return out


class _PyramidSelfAttentionBlock(nn.Module):
    '''
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    '''

    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1):
        super(_PyramidSelfAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels == None:
            self.out_channels = in_channels
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels)
        )
        self.f_query = self.f_key
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
                                 kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
                           kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3)

        local_x = []
        local_y = []
        step_h, step_w = h // self.scale, w // self.scale
        for i in range(0, self.scale):
            for j in range(0, self.scale):
                start_x, start_y = i * step_h, j * step_w
                end_x, end_y = min(start_x + step_h, h), min(start_y + step_w, w)
                if i == (self.scale - 1):
                    end_x = h
                if j == (self.scale - 1):
                    end_y = w
                local_x += [start_x, end_x]
                local_y += [start_y, end_y]

        value = self.f_value(x)
        query = self.f_query(x)
        key = self.f_key(x)

        local_list = []
        local_block_cnt = 2 * self.scale * self.scale
        for i in range(0, local_block_cnt, 2):
            value_local = value[:, :, local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1]]
            query_local = query[:, :, local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1]]
            key_local = key[:, :, local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1]]

            h_local, w_local = value_local.size(2), value_local.size(3)
            value_local = value_local.contiguous().view(batch_size, self.value_channels, -1)
            value_local = value_local.permute(0, 2, 1)

            query_local = query_local.contiguous().view(batch_size, self.key_channels, -1)
            query_local = query_local.permute(0, 2, 1)
            key_local = key_local.contiguous().view(batch_size, self.key_channels, -1)

            sim_map = torch.matmul(query_local, key_local)
            sim_map = (self.key_channels ** -.5) * sim_map
            sim_map = F.softmax(sim_map, dim=-1)

            context_local = torch.matmul(sim_map, value_local)
            context_local = context_local.permute(0, 2, 1).contiguous()
            context_local = context_local.view(batch_size, self.value_channels, h_local, w_local)
            local_list.append(context_local)

        context_list = []
        for i in range(0, self.scale):
            row_tmp = []
            for j in range(0, self.scale):
                row_tmp.append(local_list[j + i * self.scale])
            context_list.append(torch.cat(row_tmp, 3))

        context = torch.cat(context_list, 2)
        context = self.W(context)

        return context


class PyramidSelfAttentionBlock2D(_PyramidSelfAttentionBlock):
    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1):
        super(PyramidSelfAttentionBlock2D, self).__init__(in_channels,
                                                          key_channels,
                                                          value_channels,
                                                          out_channels,
                                                          scale)


class _SelfAttentionBlock(nn.Module):
    '''
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    '''

    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1):
        super(_SelfAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels

        if out_channels is None:
            self.out_channels = in_channels

        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels)
        )
        self.f_query = self.f_key
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
                                 kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
                           kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        value = self.f_value(x).view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)
        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(x).view(batch_size, self.key_channels, -1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        context = self.W(context)
        if self.scale > 1:
            context = F.upsample(input=context, size=(h, w), mode='bilinear', align_corners=True)
        return context


class SelfAttentionBlock2D(_SelfAttentionBlock):
    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1):
        super(SelfAttentionBlock2D, self).__init__(in_channels,
                                                   key_channels,
                                                   value_channels,
                                                   out_channels,
                                                   scale)

class BaseOC_Module(nn.Module):
    """
    Implementation of the BaseOC module
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: we choose 0.05 as the default value.
        size: you can apply multiple sizes. Here we only use one size.
    Return:
        features fused with Object context information.
    """

    def __init__(self, in_channels, out_channels, key_channels, value_channels, dropout, sizes=([1])):
        super(BaseOC_Module, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, out_channels, key_channels, value_channels, size) for size in sizes])
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout)
        )

    def _make_stage(self, in_channels, output_channels, key_channels, value_channels, size):
        return SelfAttentionBlock2D(in_channels,
                                    key_channels,
                                    value_channels,
                                    output_channels,
                                    size)

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        return output


class BaseOC_Context_Module(nn.Module):
    """
    Output only the context features.
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: specify the dropout ratio
        fusion: We provide two different fusion method, "concat" or "add"
        size: we find that directly learn the attention weights on even 1/8 feature maps is hard.
    Return:
        features after "concat" or "add"
    """

    def __init__(self, in_channels, out_channels, key_channels, value_channels, dropout=0, sizes=([1])):
        super(BaseOC_Context_Module, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, out_channels, key_channels, value_channels, size) for size in sizes])
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout)
        )

    def _make_stage(self, in_channels, output_channels, key_channels, value_channels, size):
        return SelfAttentionBlock2D(in_channels,
                                    key_channels,
                                    value_channels,
                                    output_channels,
                                    size)

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        output = self.conv_bn_dropout(context)
        return output


class ASP_OC_Module(nn.Module):
    def __init__(self, features, out_features=512, dilations=(12, 24, 36)):
        super(ASP_OC_Module, self).__init__()
        self.context = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=1, dilation=1, bias=True),
                                     nn.BatchNorm2d(out_features),
                                     BaseOC_Context_Module(in_channels=out_features, out_channels=out_features,
                                                           key_channels=out_features // 2, value_channels=out_features, sizes=([2])))
        self.conv2 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
                                   nn.BatchNorm2d(out_features))
        self.conv3 = nn.Sequential(
            nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
            nn.BatchNorm2d(out_features))
        self.conv4 = nn.Sequential(
            nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
            nn.BatchNorm2d(out_features))
        self.conv5 = nn.Sequential(
            nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
            nn.BatchNorm2d(out_features))

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(out_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(out_features),
            nn.Dropout2d(0.1)
        )

    def _cat_each(self, feat1, feat2, feat3, feat4, feat5):
        assert (len(feat1) == len(feat2))
        z = []
        for i in range(len(feat1)):
            z.append(torch.cat((feat1[i], feat2[i], feat3[i], feat4[i], feat5[i]), 1))
        return z

    def forward(self, x):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')

        feat1 = self.context(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)

        if isinstance(x, Variable):
            out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        elif isinstance(x, tuple) or isinstance(x, list):
            out = self._cat_each(feat1, feat2, feat3, feat4, feat5)
        else:
            raise RuntimeError('unknown input type')

        output = self.conv_bn_dropout(out)
        return output



class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class DANetModule(nn.Module):
    def __init__(self, channels):
        super(DANetModule, self).__init__()

        self.cam = CAM_Module(channels)
        self.pam = PAM_Module(channels)

        self.conv_cam1 = conv3x3(channels, channels)
        self.conv_cam2 = conv3x3(channels, channels)

        self.conv_pam1 = conv3x3(channels, channels)
        self.conv_pam2 = conv3x3(channels, channels)

    def forward(self, x):
        cam = self.conv_cam1(x)
        cam = self.cam(cam)
        cam = self.conv_cam2(cam)

        pam = self.conv_pam1(x)
        pam = self.pam(pam)
        pam = self.conv_pam2(pam)

        return cam + pam