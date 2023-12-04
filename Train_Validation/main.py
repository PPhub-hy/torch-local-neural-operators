from lib.networks_LNO import NetNS_InN_legendre, NetWave_InN_legendre, NetBurgers1D_InN_legendre

from Data.DatasetNS import NS_Dataset2D
from Data.DatasetWave import Wave_Dataset2D
from Data.DatasetBurgers2D import Burgers_Dataset2D
from Data.DatasetBurgers import Burgers_Dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import random
import math
import scipy.io as scio
import numpy as np

PROBLEM ='NS' #'NS' #'Burgers1D'#'Burgers2D' #'Wave'

if PROBLEM == 'NS' or PROBLEM == 'Burgers2D':
    from lib.train import train_iterative_NS_InN as train
    from lib.test import test_iterative_NS_InN as test
elif PROBLEM == 'Wave':
    from lib.train import train_iterative_Wave_InN as train
    from lib.test import test_iterative_Wave_InN as test
elif PROBLEM == 'Burgers1D':
    from lib.train import train_iterative_Burgers1D_InN as train
    from lib.test import test_iterative_Burgers1D_InN as test

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--out_name", help="name of this try")
args = parser.parse_args()

in_length = None
# network parameters
if PROBLEM == 'NS' or PROBLEM == 'Burgers2D':
    Re = 500
    t_interval = 5
    in_length = 1
elif PROBLEM == 'Wave':
    t_interval = 10
    in_length = 1
elif PROBLEM =='Burgers1D':
    t_interval =5
    in_length =1

# training parameters
learning_rate = 0.001
weight_decay = 1e-4
lr_min = 0
momentum = 0.9
batch_size = 4
print_frequency = 50

rounds = 10
epochs = 20#15
epochs_overall = rounds * epochs


def get_parameter_number(network):
    total_num = sum(p.numel() for p in network.parameters())
    trainable_num = sum(p.numel() for p in network.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def train_test_save():
    if PROBLEM == 'NS':
        orders_all = [i for i in range(1, 12)] #Re500-mini
        #orders_all = [i for i in range(1, 111)] #Re1000
        #orders_all = [i for i in range(1, 136)] #Re500
        #orders_all = [i for i in range(1, 211)] #Re100
        #orders_all = [i for i in range(1, 511)] #Re50

        orders_test = [i for i in range(1, 11)]
        orders_train = [order for order in orders_all if order not in orders_test]

        print('orders_train = ', orders_train)
        print('orders_test = ', orders_test)

        dataset = NS_Dataset2D(data_dir='./Data/',data_name='NS128Re{}t1000'.format(Re), orders_train=orders_train, orders_test=orders_test,
                               t_interval=t_interval)

        network = NetNS_InN_legendre(num_blocks=4, In_length=in_length)
        #network = NetNS_InN_legendre(num_blocks=4, In_length=in_length, cheb=True)

    elif PROBLEM == 'Wave':
        orders_all = [i for i in range(11, 211)]
        orders_test = [i for i in range(1, 11)]
        orders_train = [order for order in orders_all if order not in orders_test]

        dataset = Wave_Dataset2D(data_dir='./Data/', data_name='Wave128t1000', orders_train=orders_train,
                               orders_test=orders_test,
                               t_interval=t_interval)

        network = NetWave_InN_legendre(num_blocks=4)
        #network = NetWave_InN_legendre(num_blocks=4, cheb=True)

    elif PROBLEM =='Burgers2D':
        orders_all = [i for i in range(11, 211)]
        orders_test = [i for i in range(1, 11)]
        orders_train = [order for order in orders_all if order not in orders_test]

        dataset = Burgers_Dataset2D(data_dir='./Data/', data_name='Burgers2D128Re100t1000', orders_train=orders_train,
                                 orders_test=orders_test,
                                 t_interval=t_interval)

        network = NetNS_InN_legendre(num_blocks=4, In_length=in_length)
        #network = NetNS_InN_legendre(num_blocks=4, In_length=in_length, cheb=True)

    elif PROBLEM == 'Burgers1D':
        orders_all = [i for i in range(11, 211)]
        orders_test = [i for i in range(1, 11)]
        orders_train = [order for order in orders_all if order not in orders_test]

        dataset = Burgers_Dataset(data_dir='./Data/', data_name='Burgers128Re100t1000', orders_train=orders_train,
                                 orders_test=orders_test,
                                 t_interval=t_interval)

        network = NetBurgers1D_InN_legendre(num_blocks=4, In_length=1)
        #network = NetBurgers1D_InN_legendre(num_blocks=4, In_length=1, cheb=True)

    print(dataset.cache_names)
    print(dataset.sample_nums)

    print(network)
    print(get_parameter_number(network))
    network = network.cuda()
    optimizer = optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7, last_epoch=-1)

    recurrent = 10
    for r in range(rounds):
        train_generator = dataset.data_generator_series(out_length=recurrent,in_length=in_length, batch_size=batch_size)
        train(network=network, batch_size=batch_size,
              epochs=epochs, max_ep=epochs_overall, last_ep=r*epochs, optimizer=optimizer,
              dataset=dataset, train_gen=train_generator, round=recurrent,
              print_frequency=print_frequency,
              In_length=in_length)
        scheduler.step()

    torch.save(network, 'models/' + args.out_name + '_model.pp')

    test_round = 10
    long_round = 100
    test_generator = dataset.data_generator_series(out_length=test_round, in_length=in_length, batch_size=batch_size, split='test')
    test(network, dataset, test_generator, test_round, long_round, args.out_name, In_length=in_length)


def load_test(out_name):
    model_file = out_name + '_model.pp'
    network = torch.load('models/' + model_file)
    print(network)
    network = network.cuda()

    if PROBLEM == 'NS':
        orders_test = [i for i in range(1, 11)]
        ground_truth_name = 'NS128Re{}t1000'.format(Re)

        dataset = NS_Dataset2D(data_dir='./Data/', data_name='NS128Re{}t1000'.format(Re), orders_train=[],
                               orders_test=orders_test,
                               t_interval=t_interval)

    elif PROBLEM == 'Wave':
        orders_test = [i for i in range(1, 11)]
        ground_truth_name = 'Wave128t1000'

        dataset = Wave_Dataset2D(data_dir='./Data/', data_name='Wave128t1000', orders_train=[],
                               orders_test=orders_test,
                               t_interval=t_interval)

    elif PROBLEM == 'Burgers2D':
        orders_test = [i for i in range(1, 11)]
        ground_truth_name = 'Burgers2D128Re100t1000'

        dataset = Burgers_Dataset2D(data_dir='./Data/', data_name='Burgers2D128Re100t1000'.format(Re),
                               orders_train=[], orders_test=orders_test,
                               t_interval=t_interval)

    elif PROBLEM == 'Burgers1D':
        orders_test = [i for i in range(1, 11)]
        ground_truth_name = 'Burgers128Re100t1000'

        dataset = Burgers_Dataset(data_dir='./Data/', data_name='Burgers128Re100t1000', orders_train=[],
                                 orders_test=orders_test,
                                 t_interval=t_interval)

    test_round = 10
    long_round = 100
    test_generator = dataset.data_generator_series(out_length=test_round, in_length=in_length, batch_size=batch_size,
                                                   split='test')
    test(network, dataset, test_generator, test_round, long_round, args.out_name, In_length=in_length)
    if PROBLEM == 'NS':
        MeanSquareError(out_name, ground_truth_name, orders_test)

def MeanSquareError(out_name, ground_truth_name, orders_test):
    output = scio.loadmat('outputs/' + out_name + '.mat')['output']
    NG = output.shape[-1]
    test_t = [9, 19, 39, 99]
    MSE = np.zeros(len(test_t))
    for k in range(len(orders_test)):
        order = orders_test[k]
        ground_truth = scio.loadmat('./Data/' + ground_truth_name + '/' + ground_truth_name + '_' + str(order) + '.mat')
        i = 0
        for ii in test_t:
            ground_truth_u = ground_truth['u'][(ii+1) * t_interval, :].reshape(NG, NG)
            ground_truth_v = ground_truth['v'][(ii+1) * t_interval, :].reshape(NG, NG)
            MSE[i] += np.sqrt(np.sum((output[ii, k, 0, :, :]-ground_truth_u)**2+(output[ii, k, 1, :, :]-ground_truth_v)**2)/NG/NG)
            i += 1
    MSE = MSE/len(orders_test)
    print('MSE(t=0.5, 1, 2, 5)={}'.format(MSE))



if __name__ == '__main__':
    train_test_save()
    load_test(args.out_name)
    #load_test_new_sample()
