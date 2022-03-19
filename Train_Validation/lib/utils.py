import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as scio

def generte_legendre_filters_2D(file, n, m):
    '''
    :param file: the path of the legendre filters
    :param n: the length of the legendre filters
    :param m: the number of modes reserved
    :return: the filters for decomposition and reconstruction, the length of the filters
    '''
    filters_1D = scio.loadmat(file.format(n))

    window = np.append(np.linspace(0, 1, n // 2), np.linspace(1, 0, n // 2))


    filters_forward = []
    filters_backward = []
    for i in range(m):
        filters_forward.append(filters_1D['forward'][i,:])
        filters_backward.append(filters_1D['backward'][i,:])
        assert len(filters_1D['forward'][i,:]) == n

    filters_d = []
    filters_r = []
    for i in range(m):
        filters_forward[i] =  torch.Tensor(filters_forward[i]).reshape(1, n)# * window
        filters_backward[i] = torch.Tensor(filters_backward[i]).reshape(1, n)

    for i in range(m):
        for j in range(m):
            filters_d += [torch.mul(filters_forward[i].T, filters_forward[j]).reshape(1, 1, n, n)]
            filters_r += [torch.mul(filters_backward[i].T, filters_backward[j]).reshape(1, 1, n, n)]

    filter_d = torch.cat(tuple(filters_d), dim=0)
    filter_r = torch.cat(tuple(filters_r), dim=0)
    return filter_d, filter_r, n

def generte_legendre_filters_1D(file, n, m):
    '''
    :param file: the path of the legendre filters
    :param n: the length of the legendre filters
    :param m: the number of modes reserved
    :return: the filters for decomposition and reconstruction, the length of the filters
    '''
    filters_1D = scio.loadmat(file.format(n))

    window = np.append(np.linspace(0, 1, n // 2), np.linspace(1, 0, n // 2))


    filters_forward = []
    filters_backward = []
    for i in range(m):
        filters_forward.append(filters_1D['forward'][i,:])
        filters_backward.append(filters_1D['backward'][i,:])
        assert len(filters_1D['forward'][i,:]) == n

    filters_d = []
    filters_r = []
    for i in range(m):
        filters_forward[i] =  torch.Tensor(filters_forward[i]).reshape(1, n)# * window
        filters_backward[i] = torch.Tensor(filters_backward[i]).reshape(1, n)

    for i in range(m):
        filters_d += [filters_forward[i].reshape(1, 1, n,)]
        filters_r += [filters_backward[i].reshape(1, 1, n)]

    filter_d = torch.cat(tuple(filters_d), dim=0)
    filter_r = torch.cat(tuple(filters_r), dim=0)
    return filter_d, filter_r, n

