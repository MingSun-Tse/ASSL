import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2D_WN(nn.Conv2d):
    '''Conv2D with weight normalization.
    '''
    def __init__(self, 
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros',  # TODO: refine this type
        device=None,
        dtype=None
    ):
        super(Conv2D_WN, self).__init__(in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, dilation=dilation, groups=groups, 
            bias=bias, padding_mode=padding_mode)

        # set up the scale variable in weight normalization
        self.wn_scale = nn.Parameter(torch.ones(out_channels), requires_grad=True)
        self.init_wn()
    
    def init_wn(self):
        """initialize the wn parameters"""
        for i in range(self.weight.size(0)):
            self.wn_scale.data[i] = torch.norm(self.weight.data[i])

    def forward(self, input):
        w = F.normalize(self.weight, dim=(1,2,3))
        w = w * self.wn_scale.view(-1,1,1,1)
        return F.conv2d(input, w, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

def wn_conv(in_channels, out_channels, kernel_size, bias=True):
    return Conv2D_WN(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)




class LiteUpsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, n_out=3, bn=False, act=False, bias=True):

        m = []
        m.append(conv(n_feats, n_out*(scale ** 2), 3, bias))
        m.append(nn.PixelShuffle(scale))
        # if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
        #     for _ in range(int(math.log(scale, 2))):
        #         m.append(conv(n_feats, 4 * n_out, 3, bias))
        #         m.append(nn.PixelShuffle(2))
        #         if bn:
        #             m.append(nn.BatchNorm2d(n_out))
        #         if act == 'relu':
        #             m.append(nn.ReLU(True))
        #         elif act == 'prelu':
        #             m.append(nn.PReLU(n_out))

        # elif scale == 3:
        #     m.append(conv(n_feats, 9 * n_out, 3, bias))
        #     m.append(nn.PixelShuffle(3))
        #     if bn:
        #         m.append(nn.BatchNorm2d(n_out))
        #     if act == 'relu':
        #         m.append(nn.ReLU(True))
        #     elif act == 'prelu':
        #         m.append(nn.PReLU(n_out))
        # else:
        #     raise NotImplementedError

        super(LiteUpsampler, self).__init__(*m)

