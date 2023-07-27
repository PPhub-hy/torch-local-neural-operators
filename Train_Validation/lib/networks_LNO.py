import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.utils import generte_legendre_filters_2D, generte_legendre_filters_1D
import math

class NetNS_InN_legendre(nn.Module):

    def __init__(self, num_blocks, In_length, cheb = False):
        super(NetNS_InN_legendre, self).__init__()
        self.num_blocks = num_blocks
        with_bias = False#True#

        self.filter_d = None
        self.filter_r = None

        self.n = 12
        self.m = 6
        self.k = 2
        print('legendre params: n:{} m:{} k:{}'.format(self.n,self.m,self.k))
        if cheb:
            file_path = 'lib/chebyshevs/ChebyshevConv{}.mat'
            print("*"*5, 'using Chebyshev filters', "*"*5)
        else:
            file_path = 'lib/legendres/LegendreConv{}.mat'
            print("*"*5,'using Legendre filters',"*"*5)
        self.filter_d, self.filter_r, self.l_filters = generte_legendre_filters_2D(file_path,
                                                                                   n=self.n,
                                                                                   m=self.m)
        self.modes = self.filter_d.shape[0]

        self.convs = []
        self.WPDlayers = []
        self.linears = []
        self.linearModes = []

        first_channel = 2 * In_length
        out_channel = 2

        channel = 40
        self.first_channel = first_channel
        self.channel = channel
        self.first_conv = nn.Conv2d(first_channel, channel, kernel_size=3, bias=with_bias)
        for i in range(num_blocks):
            self.linearModes.append(
                nn.Conv2d(self.modes * channel, self.modes * channel, kernel_size=1, groups=channel, bias=with_bias)
            )
            print('spectral layer {}, modes {}'.format(i + 1, self.modes))

            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(channel, channel, kernel_size=3, bias=with_bias),
                    nn.GELU(),
                    nn.Conv2d(channel, channel, kernel_size=3, bias=with_bias),
                            )
                        )

        self.conv11 = nn.Conv2d(channel, 128,  kernel_size=1, bias=with_bias)
        self.convout = nn.Conv2d(128, out_channel, kernel_size=1, bias=with_bias)

        self.first_conv.weight = nn.Parameter(self.first_conv.weight * math.sqrt(3))
        for i in range(num_blocks):
            self.linearModes[i].weight = nn.Parameter(self.linearModes[i].weight * math.sqrt(3))
            self.convs[i][0].weight = nn.Parameter(self.convs[i][0].weight * math.sqrt(6))
            self.convs[i][2].weight = nn.Parameter(self.convs[i][2].weight * math.sqrt(3))
        self.conv11.weight = nn.Parameter(self.conv11.weight * math.sqrt(6))
        self.convout.weight = nn.Parameter(self.convout.weight * math.sqrt(3))

        self.convs = nn.ModuleList(self.convs)
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
        r1 = self.get_padding_R(input.shape[-1] + 2 * recept - 2)
        left = recept
        right = recept + r1
        r2 = self.get_padding_R(input.shape[-2] + 2 * recept - 2)
        down = recept
        up = recept + r2

        input = F.pad(input, (left, right, down, up), "circular")
        x = self.first_conv(input)
        b = x.shape[0]
        c = x.shape[1]

        for idx in range(self.num_blocks):
            RR = self.RR

            linear_x = self.convs[idx](x)
            rc = 2

            l1 = x.shape[-1]
            l2 = x.shape[-2]
            x = x.reshape(b * c, 1, l2, l1)
            Legendre_x = F.conv2d(x, self.filter_d / self.k, stride=self.n // self.k)  #shape: [b * c, self.modes, _, _]
            ll1 = Legendre_x.shape[-1]
            ll2 = Legendre_x.shape[-2]
            if self.recept == 0:
                print('remaining space: ', ll1)

            Legendre_x = Legendre_x.reshape(b, c, self.modes, ll2, ll1)
            Legendre_x = Legendre_x.permute(0, 2, 1, 3, 4)
            Legendre_x = Legendre_x.reshape(b, c * self.modes, ll2, ll1)

            Legendre_x2 = self.linearModes[idx](Legendre_x)
            Legendre_x2 = Legendre_x2.reshape(b, self.modes, c, ll2, ll1)
            Legendre_x2 = Legendre_x2.permute(0, 2, 1, 3, 4)
            Legendre_x2 = Legendre_x2.reshape(b * c, self.modes, ll2, ll1)

            Legendre_x = Legendre_x2*2

            Legendre_x = F.conv_transpose2d(Legendre_x, self.filter_r / self.k, stride=self.n // self.k)
            Legendre_x = Legendre_x.reshape(b, c, l2, l1)[:,:,RR:-RR,RR:-RR] + linear_x[:,:,RR-rc:-RR+rc,RR-rc:-RR+rc]

            x = F.gelu(Legendre_x)

        x = self.conv11(x)
        x = F.gelu(x)
        x = self.convout(x)

        return x[:,:,:-r2,:-r1]

class NetWave_InN_legendre(nn.Module):

    def __init__(self, num_blocks, cheb = False):
        super(NetWave_InN_legendre, self).__init__()
        self.num_blocks = num_blocks
        with_bias = False

        self.filter_d = None
        self.filter_r = None

        self.n = 12
        self.m = 4
        self.k = 2
        print('legendre params: n:{} m:{} k:{}'.format(self.n,self.m,self.k))
        if cheb:
            file_path = 'lib/chebyshevs/ChebyshevConv{}.mat'
            print("*"*5, 'using Chebyshev filters', "*"*5)
        else:
            file_path = 'lib/legendres/LegendreConv{}.mat'
            print("*"*5,'using Legendre filters',"*"*5)
        self.filter_d, self.filter_r, self.l_filters = generte_legendre_filters_2D(file_path,
                                                                                   n=self.n,
                                                                                   m=self.m)
        self.modes = self.filter_d.shape[0]

        self.convs = []
        self.WPDlayers = []
        self.linears = []
        self.linearModes = []
        first_channel = 2
        out_channel = 2
        channel = 40
        self.first_channel = first_channel
        self.channel = channel
        self.first_conv = nn.Conv2d(first_channel, channel, kernel_size=3, bias=with_bias)
        for i in range(num_blocks):
            self.linearModes.append(
                nn.Conv2d(self.modes * channel, self.modes * channel, kernel_size=1, groups=channel, bias=with_bias)
            )
            print('spectral layer {}, modes {}'.format(i + 1, self.modes))

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
        if l_input % n == 0:
            return 0
        else:
            return n - l_input % n

    def forward(self, input):
        input[:, 1:2, :, :] = input[:, 1:2, :, :] / 10
        p = input[:, 0:1, :, :]
        dp = input[:, 1:2, :, :]

        recept = self.recept
        r1 = self.get_padding_R(input.shape[-1] + 2 * recept - 2)
        left = recept
        right = recept + r1
        r2 = self.get_padding_R(input.shape[-2] + 2 * recept - 2)
        down = recept
        up = recept + r2

        x = F.pad(input, (left, right, down, up), "circular")

        x = self.first_conv(x)
        b = x.shape[0]
        c = x.shape[1]

        for idx in range(self.num_blocks):
            RR = self.RR

            linear_x = self.convs[idx](x)
            rc = 2

            l1 = x.shape[-1]
            l2 = x.shape[-2]
            x = x.reshape(b * c, 1, l2, l1)
            Legendre_x = F.conv2d(x, self.filter_d / self.k, stride=self.n // self.k)  #shape: [b * c, self.modes, _, _]
            ll1 = Legendre_x.shape[-1]
            ll2 = Legendre_x.shape[-2]
            if self.recept == 0:
                print('remaining space: ', ll1)

            Legendre_x = Legendre_x.reshape(b, c, self.modes, ll2, ll1)
            Legendre_x = Legendre_x.permute(0, 2, 1, 3, 4)
            Legendre_x = Legendre_x.reshape(b, c * self.modes, ll2, ll1)

            Legendre_x2 = self.linearModes[idx](Legendre_x)
            Legendre_x2 = Legendre_x2.reshape(b, self.modes, c, ll2, ll1)
            Legendre_x2 = Legendre_x2.permute(0, 2, 1, 3, 4)
            Legendre_x2 = Legendre_x2.reshape(b * c, self.modes, ll2, ll1)

            Legendre_x = Legendre_x2*2

            Legendre_x = F.conv_transpose2d(Legendre_x, self.filter_r / self.k, stride=self.n // self.k) #shape: [b * c, self.modes, l, l]
            Legendre_x = Legendre_x.reshape(b, c, l2, l1)[:,:,RR:-RR,RR:-RR] + linear_x[:,:,RR-rc:-RR+rc,RR-rc:-RR+rc]

            x = F.gelu(Legendre_x)

        x = self.conv11(x)
        x = F.gelu(x)
        x = self.convout(x)

        x[:, 1:2, :, :] = x[:, 1:2, :, :] * 10

        return x[:,:,:-r2,: -r1]

'''
The network used for Burgers2D problem is identical with the 'NetNS_InN_legendre'
'''

class NetBurgers1D_InN_legendre(nn.Module):

    def __init__(self, num_blocks, In_length, cheb = False):
        super(NetBurgers1D_InN_legendre, self).__init__()
        self.num_blocks = num_blocks
        with_bias = False

        self.filter_d = None
        self.filter_r = None

        self.n = 12
        self.m = 6
        self.k = 2
        print('legendre params: n:{} m:{} k:{}'.format(self.n,self.m,self.k))
        if cheb:
            file_path = 'lib/chebyshevs/ChebyshevConv{}.mat'
            print("*"*5, 'using Chebyshev filters', "*"*5)
        else:
            file_path = 'lib/legendres/LegendreConv{}.mat'
            print("*"*5,'using Legendre filters',"*"*5)
        self.filter_d, self.filter_r, self.l_filters = generte_legendre_filters_1D(file_path,
                                                                                   n=self.n,
                                                                                   m=self.m)
        self.modes = self.filter_d.shape[0]

        self.convs = []
        self.WPDlayers = []
        self.linears = []
        self.linearModes = []
        first_channel = 1 * In_length
        out_channel = 1
        channel = 20
        self.first_channel = first_channel
        self.channel = channel
        self.first_conv = nn.Conv1d(first_channel, channel, kernel_size=3, bias=with_bias)
        for i in range(num_blocks):
            self.linearModes.append(
                nn.Conv1d(self.modes * channel, self.modes * channel, kernel_size=1, groups=channel, bias=with_bias)
            )
            print('spectral layer {}, modes {}'.format(i + 1, self.modes))

            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(channel, channel, kernel_size=3, bias=with_bias),
                    nn.GELU(),
                    nn.Conv1d(channel, channel, kernel_size=3, bias=with_bias),
                            )
                        )

        self.conv11 = nn.Conv1d(channel, 128,  kernel_size=1, bias=with_bias)
        self.convout = nn.Conv1d(128, out_channel, kernel_size=1, bias=with_bias)

        self.convs = nn.ModuleList(self.convs)
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
        r = self.get_padding_R(input.shape[-1] + 2 * recept - 2)
        left = recept
        right = recept + r
        input = F.pad(input, (left, right), "circular")

        x = self.first_conv(input)
        b = x.shape[0]
        c = x.shape[1]

        for idx in range(self.num_blocks):
            RR = self.RR

            linear_x = self.convs[idx](x)
            rc = 2

            l = x.shape[-1]
            x = x.reshape(b * c, 1, l)
            Legendre_x = F.conv1d(x, self.filter_d / self.k, stride=self.n // self.k)  #shape: [b * c, self.modes, _, _]
            ll = Legendre_x.shape[-1]
            if self.recept == 0:
                print('remaining space: ', ll)

            Legendre_x = Legendre_x.reshape(b, c, self.modes, ll)
            Legendre_x = Legendre_x.permute(0, 2, 1, 3)
            Legendre_x = Legendre_x.reshape(b, c * self.modes, ll)

            Legendre_x2 = self.linearModes[idx](Legendre_x)
            Legendre_x2 = Legendre_x2.reshape(b, self.modes, c, ll)
            Legendre_x2 = Legendre_x2.permute(0, 2, 1, 3)
            Legendre_x2 = Legendre_x2.reshape(b * c, self.modes, ll)

            Legendre_x = Legendre_x2*2

            Legendre_x = F.conv_transpose1d(Legendre_x, self.filter_r / self.k, stride=self.n // self.k) #shape: [b * c, self.modes, l, l]
            Legendre_x = Legendre_x.reshape(b, c, l)[:,:,RR:-RR] + linear_x[:,:,RR-rc:-RR+rc]

            x = F.gelu(Legendre_x)

        x = self.conv11(x)
        x = F.gelu(x)
        x = self.convout(x)

        return x[:,:,:-r]

