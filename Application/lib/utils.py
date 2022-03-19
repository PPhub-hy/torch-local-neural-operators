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

def H1Loss(output, label, NG):
    assert output.shape == label.shape
    u_output = output[:, 0:1, :, :]
    v_output = output[:, 1:2, :, :]
    u_label = label[:, 0:1, :, :]
    v_label = label[:, 1:2, :, :]

    dux_output = F.pad(u_output, (0, 2, 0, 0), "circular")-F.pad(u_output, (1, 1, 0, 0), "circular")
    dux_output = dux_output[:, :, :, :-2] * NG / 2
    duy_output = F.pad(u_output, (0, 0, 0, 2), "circular") - F.pad(u_output, (0, 0, 1, 1), "circular")
    duy_output = duy_output[:, :, :-2, :] * NG / 2

    dvx_output = F.pad(v_output, (0, 2, 0, 0), "circular") - F.pad(v_output, (1, 1, 0, 0), "circular")
    dvx_output = dvx_output[:, :, :, :-2] * NG / 2
    dvy_output = F.pad(v_output, (0, 0, 0, 2), "circular") - F.pad(v_output, (0, 0, 1, 1), "circular")
    dvy_output = dvy_output[:, :, :-2, :] * NG / 2

    dux_label = F.pad(u_label, (0, 2, 0, 0), "circular") - F.pad(u_label, (1, 1, 0, 0), "circular")
    dux_label = dux_label[:, :, :, :-2] * NG / 2
    duy_label = F.pad(u_label, (0, 0, 0, 2), "circular") - F.pad(u_label, (0, 0, 1, 1), "circular")
    duy_label = duy_label[:, :, :-2, :] * NG / 2

    dvx_label = F.pad(v_label, (0, 2, 0, 0), "circular") - F.pad(v_label, (1, 1, 0, 0), "circular")
    dvx_label = dvx_label[:, :, :, :-2] * NG / 2
    dvy_label = F.pad(v_label, (0, 0, 0, 2), "circular") - F.pad(v_label, (0, 0, 1, 1), "circular")
    dvy_label = dvy_label[:, :, :-2, :] * NG / 2

    #print(dux_output)
    #print(dux_label)

    H1 = (u_output-u_label)**2 +(v_output-v_label)**2
    H1 = H1 + (dux_output - dux_label)**2 + (dvx_output - dvx_label)**2 + (duy_output - duy_label)**2 + (dvy_output - dvy_label)**2
    H1 = torch.sum(torch.sum(H1, -1), -1) / NG / NG
    H1 = torch.sqrt(H1)
    #print(H1.shape)
    return torch.mean(H1)

def L2Loss(output, label, NG):
    assert output.shape == label.shape
    u_output = output[:, 0:1, :, :]
    v_output = output[:, 1:2, :, :]
    u_label = label[:, 0:1, :, :]
    v_label = label[:, 1:2, :, :]

    L2 = (u_output-u_label)**2 +(v_output-v_label)**2
    #print(L2)
    L2 = torch.sum(torch.sum(L2, -1), -1) / NG / NG
    #print(L2)
    L2 = torch.sqrt(L2)
    return torch.mean(L2)


if __name__ == '__main__':
    output = torch.rand(1, 2, 3, 3)
    label = torch.ones(1, 2, 3, 3)
    print(output)
    print(label)
    H1= H1Loss(output, label, 3)
    #print(H1)
    L2 = L2Loss(output, label, 3)
    #print(L2)
