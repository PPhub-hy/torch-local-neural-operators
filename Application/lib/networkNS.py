import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import pywt
from scipy import signal, stats
from functools import partial
from lib.utils import generate_wavelet_filters_2D



class NetNS_InN_legendre(nn.Module):

    def __init__(self, num_blocks, In_length):
        super(NetNS_InN_legendre, self).__init__()
        self.num_blocks = num_blocks
        with_bias = False#True#

        self.filter_d = None
        self.filter_r = None

        self.n = 16
        self.m = 8
        self.k = 2
        print('legendre params: n:{} m:{} k:{}'.format(self.n,self.m,self.k))
        file_path = 'lib/legendres/LegendreConv{}.mat'
        self.filter_r, self.filter_d, self.l_filters = generte_legendre_filters_2D(file_path,
                                                                                   n=self.n,
                                                                                   m=self.m)
        self.modes = self.filter_d.shape[0]

        self.convs = []
        self.WPDlayers = []
        self.linears = []
        self.linearModes = []
        first_channel = 6 * In_length
        out_channel = 3
        if not with_p:
            first_channel = 2 * In_length
            out_channel = 2
        channel = 40
        self.first_channel = first_channel
        self.channel = channel
        self.first_conv = nn.Conv2d(first_channel, channel, kernel_size=3, bias=with_bias)
        for i in range(num_blocks):
            self.WPDlayers.append(
                nn.Conv2d(channel * self.modes, channel * self.modes, kernel_size = 1, groups=self.modes, bias=with_bias)
            )
            self.linearModes.append(
                nn.Conv2d(self.modes * channel, self.modes * channel, kernel_size=1, groups=channel, bias=with_bias)
            )
            print('WPDlayer {}, modes {}'.format(i + 1, self.modes))



            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(channel, channel, kernel_size=3, bias=with_bias),
                    nn.GELU(),
                    nn.Conv2d(channel, channel, kernel_size=3, bias=with_bias),
                            )
                        )

        self.conv11 = nn.Conv2d(channel, 128,  kernel_size=1, bias=with_bias)
        self.convout = nn.Conv2d(128, out_channel, kernel_size=1, bias=with_bias)

        self.convs = nn.ModuleList(self.convs)
        self.WPDlayers = nn.ModuleList(self.WPDlayers)
        self.linears = nn.ModuleList(self.linears)
        self.linearModes = nn.ModuleList(self.linearModes)

        self.RR = self.n // self.k * (self.k - 1)
        self.recept = self.RR * self.num_blocks + 1
        print('Range of receptive domain: {}'.format(self.recept))

        self.filter_r = self.filter_r.cuda()
        self.filter_d = self.filter_d.cuda()

    def get_padding_R(self, l_input):
        n = self.n
        if l_input < n:
            return n - l_input

        assert n % 2 == 0
        n = n // 2

        return n - l_input % n

    def forward(self, input):
        recept = self.recept
        left = recept
        right = recept

        x = self.first_conv(input)
        b = x.shape[0]
        c = x.shape[1]

        for idx in range(self.num_blocks):
            RR = self.RR

            linear_x = self.convs[idx](x)
            rc = 2

            l1 = x.shape[-2]
            l2 = x.shape[-1]
            x = x.reshape(b * c, 1, l1, l2)
            Legendre_x = F.conv2d(x, self.filter_d / self.k, stride=self.n // self.k)
            ll1 = Legendre_x.shape[-2]
            ll2 = Legendre_x.shape[-1]
            if self.recept == 0:
                print('remaining space: ', ll1, '*', ll2)
            Legendre_x = Legendre_x.reshape(b, c, self.modes, ll1, ll2)
            Legendre_x = Legendre_x.reshape(b, c * self.modes, ll1, ll2)

            Legendre_x = Legendre_x.reshape(b, c, self.modes, ll1, ll2)
            Legendre_x = Legendre_x.permute(0, 2, 1, 3, 4)
            Legendre_x = Legendre_x.reshape(b, c * self.modes, ll1, ll2)

            Legendre_x2 = self.linearModes[idx](Legendre_x)
            Legendre_x2 = Legendre_x2.reshape(b, self.modes, c, ll1, ll2)
            Legendre_x2 = Legendre_x2.permute(0, 2, 1, 3, 4)
            Legendre_x2 = Legendre_x2.reshape(b * c, self.modes, ll1, ll2)

            Legendre_x = Legendre_x2 + Legendre_x2

            Legendre_x = F.conv_transpose2d(Legendre_x, self.filter_r / self.k, stride=self.n // self.k)
            Legendre_x = Legendre_x.reshape(b, c, l1, l2)[:,:,RR:-RR,RR:-RR] + linear_x[:,:,RR-rc:-RR+rc,RR-rc:-RR+rc]

            x = F.gelu(Legendre_x)

        x = self.conv11(x)
        x = F.gelu(x)
        x = self.convout(x)

        return x[:,:,:,:]