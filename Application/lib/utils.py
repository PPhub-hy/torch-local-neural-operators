import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt


def generate_wavelet_filters_2D(wl, level, modes_a, modes_b):
    dec_l, dec_h, rec_l, rec_h = wl.filter_bank

    l = len(dec_l)

    rec_l = rec_l[::-1]
    rec_h = rec_h[::-1]

    dec_l = torch.Tensor(dec_l).reshape(1, l)
    dec_h = torch.Tensor(dec_h).reshape(1, l)
    rec_l = torch.Tensor(rec_l).reshape(1, l)
    rec_h = torch.Tensor(rec_h).reshape(1, l)

    filters_d = []
    filters_r = []

    filters_d += [torch.mul(dec_l.T, dec_l).reshape(1, 1, l, l)]
    filters_d += [torch.mul(dec_l.T, dec_h).reshape(1, 1, l, l)]
    filters_d += [torch.mul(dec_h.T, dec_l).reshape(1, 1, l, l)]
    filters_d += [torch.mul(dec_h.T, dec_h).reshape(1, 1, l, l)]

    filters_r += [torch.mul(rec_l.T, rec_l).reshape(1, 1, l, l)]
    filters_r += [torch.mul(rec_l.T, rec_h).reshape(1, 1, l, l)]
    filters_r += [torch.mul(rec_h.T, rec_l).reshape(1, 1, l, l)]
    filters_r += [torch.mul(rec_h.T, rec_h).reshape(1, 1, l, l)]

    def get_idxs_2D(level):
        Is = []
        for i in range(4 ** level):
            idxs = []
            idx = i
            for k in range(level):
                kk = level - k - 1
                h_or_l = idx // 4 ** kk
                idx = idx % 4 ** kk
                idxs.append(h_or_l)
            Is.append(idxs)
        return Is

    def mix_filter(idxs, filters):
        this_filter = None
        for k in range(level):
            h_or_l = idxs[-1 - k]
            if this_filter is None:
                this_filter = filters[h_or_l]
            else:
                this_filter = F.conv_transpose2d(this_filter, filters[h_or_l], stride=2)
        return this_filter

    Is = get_idxs_2D(level)

    filter_d = None
    filter_r = None
    for i in range(len(Is)):
        Dist1 = 0
        Dist2 = 0
        for j in range(len(Is[i])):
            a1 = 0
            a2 = 0
            if Is[i][j] == 1:
                a1 = 1
            elif Is[i][j] == 2:
                a2 = 1
            elif Is[i][j] == 3:
                a1 = 1
                a2 = 1
            Dist1 += 2 ** (level - 1 - j) * a1
            Dist2 += 2 ** (level - 1 - j) * a2

        if  min(Dist1 ,Dist2) < modes_b and max(Dist1 ,Dist2) < modes_a:
            this_filter_d = mix_filter(Is[i], filters_d)
            if filter_d is None:
                filter_d = this_filter_d
            else:
                filter_d = torch.cat((filter_d, this_filter_d), dim=0)

            this_filter_r = mix_filter(Is[i], filters_r)
            if filter_r is None:
                filter_r = this_filter_r
            else:
                filter_r = torch.cat((filter_r, this_filter_r), dim=0)

    return filter_d, filter_r, l



if __name__ == '__main__':
    output = torch.rand(1, 2, 3, 3)
    label = torch.ones(1, 2, 3, 3)
    print(output)
    print(label)
    H1= H1Loss(output, label, 3)
    #print(H1)
    L2 = L2Loss(output, label, 3)
    #print(L2)
