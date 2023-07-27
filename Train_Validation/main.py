from lib.networks_LNO import NetNS_InN_legendre, NetWave_InN_legendre, NetBurgers1D_InN_legendre
from lib.networks_FNO import NetNS_InN_fourior_L, NetWave_InN_fourior_L, NetBurgers1D_InN_fourior_L

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

PROBLEM ='Wave' #'NS' #'Burgers1D'#'Burgers2D' #'Wave'

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
    Re = 1000
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
        orders_all = [i for i in range(11, 12)] #Re500-mini
        #orders_all = [i for i in range(11, 111)] #Re1000
        #orders_all = [i for i in range(11, 136)] #Re500
        #orders_all = [i for i in range(11, 211)] #Re100
        #orders_all = [i for i in range(11, 511)] #Re50

        orders_test = [i for i in range(1, 11)]
        orders_train = [order for order in orders_all if order not in orders_test]

        print('orders_train = ', orders_train)
        print('orders_test = ', orders_test)

        dataset = NS_Dataset2D(data_dir='/data/LNOdata/',data_name='NS128Re{}t1000'.format(Re), orders_train=orders_train, orders_test=orders_test,
                               t_interval=t_interval)

        network = NetNS_InN_legendre(num_blocks=4, In_length=in_length)
        #network = NetNS_InN_legendre(num_blocks=4, In_length=in_length, cheb=True)
        #network = NetNS_InN_fourior_L(12, 20)

    elif PROBLEM == 'Wave':
        orders_all = [i for i in range(11, 211)]
        orders_test = [i for i in range(1, 11)]
        orders_train = [order for order in orders_all if order not in orders_test]

        dataset = Wave_Dataset2D(data_dir='/data/LNOdata/', data_name='Wave128t1000', orders_train=orders_train,
                               orders_test=orders_test,
                               t_interval=t_interval)

        #network = NetWave_InN_legendre(num_blocks=4)
        #network = NetWave_InN_legendre(num_blocks=4, cheb=True)
        network = NetWave_InN_fourior_L(12, 20)

    elif PROBLEM =='Burgers2D':
        orders_all = [i for i in range(11, 211)]
        orders_test = [i for i in range(1, 11)]
        orders_train = [order for order in orders_all if order not in orders_test]

        dataset = Burgers_Dataset2D(data_dir='/data/LNOdata/', data_name='Burgers2D128Re100t1000', orders_train=orders_train,
                                 orders_test=orders_test,
                                 t_interval=t_interval)

        #network = NetNS_InN_legendre(num_blocks=4, In_length=in_length)
        #network = NetNS_InN_legendre(num_blocks=4, In_length=in_length, cheb=True)
        network = NetNS_InN_fourior_L(12, 20)

    elif PROBLEM == 'Burgers1D':
        orders_all = [i for i in range(11, 211)]
        orders_test = [i for i in range(1, 11)]
        orders_train = [order for order in orders_all if order not in orders_test]

        dataset = Burgers_Dataset(data_dir='/data/LNOdata/', data_name='Burgers128Re100t1000', orders_train=orders_train,
                                 orders_test=orders_test,
                                 t_interval=t_interval)

        network = NetBurgers1D_InN_legendre(num_blocks=4, In_length=1)
        #network = NetBurgers1D_InN_legendre(num_blocks=4, In_length=1, cheb=True)
        #network = NetBurgers1D_InN_fourior_L(12,20)

    print(dataset.cache_names)
    print(dataset.sample_nums)

    print(network)
    print(get_parameter_number(network))
    network = network.cuda()
    optimizer = optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #optimizer = optim.SGD(network.parameters(), lr=0.005, momentum=momentum,
    #                             weight_decay=weight_decay)
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

        dataset = NS_Dataset2D(data_dir='/data/LNOdata/', data_name='NS128Re{}t1000'.format(Re), orders_train=[],
                               orders_test=orders_test,
                               t_interval=t_interval)

    elif PROBLEM == 'Wave':
        orders_test = [i for i in range(1, 11)]
        ground_truth_name = 'Wave128t1000'

        dataset = Wave_Dataset2D(data_dir='/data/LNOdata/', data_name='Wave128t1000', orders_train=[],
                               orders_test=orders_test,
                               t_interval=t_interval)

    elif PROBLEM == 'Burgers2D':
        orders_test = [i for i in range(1, 11)]
        ground_truth_name = 'Burgers2D128Re100t1000'

        dataset = Burgers_Dataset2D(data_dir='/data/LNOdata/', data_name='Burgers2D128Re100t1000'.format(Re),
                               orders_train=[], orders_test=orders_test,
                               t_interval=t_interval)

    elif PROBLEM == 'Burgers1D':
        orders_test = [i for i in range(1, 11)]
        ground_truth_name = 'Burgers128Re100t1000'

        dataset = Burgers_Dataset(data_dir='/data/LNOdata/', data_name='Burgers128Re100t1000', orders_train=[],
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
        ground_truth = scio.loadmat('/data/LNOdata/' + ground_truth_name + '/' + ground_truth_name + '_' + str(order) + '.mat')
        i = 0
        for ii in test_t:
            ground_truth_u = ground_truth['u'][(ii+1) * t_interval, :].reshape(NG, NG)
            ground_truth_v = ground_truth['v'][(ii+1) * t_interval, :].reshape(NG, NG)
            MSE[i] += np.sqrt(np.sum((output[ii, k, 0, :, :]-ground_truth_u)**2+(output[ii, k, 1, :, :]-ground_truth_v)**2)/NG/NG)
            i += 1
    MSE = MSE/len(orders_test)
    print('MSE(t=0.5, 1, 2, 5)={}'.format(MSE))

def load_test_new_sample():
    with torch.no_grad():
        print(PROBLEM)
        if PROBLEM == 'NS':
            sample_dir = '/data/LNOdata/NS128Re{}t1000'.format(Re) + '/NS128Re{}t1000'.format(Re) + '_test1.mat'
            if Re == 100:
                model_file = 'InComNS_Re100_200ep_LNO-Legendre_n12m6k2_Adam0.001+step_realLoss_interval50_baseline_NormSigma_test3_model.pp'
            if Re == 500:
                model_file = 'InComNS_Re500_200ep_LNO-Legendre_n16m8k2_Adam0.001+step_realLoss_interval50_baseline_NormSigma_test6_model.pp'
            if Re == 1000:
                model_file = 'InComNS_Re1000_200ep_LNO-Legendre_n24m8k2_Adam0.001+step_realLoss_interval50_baseline_NormSigma_test8_model.pp'
        elif PROBLEM == 'Wave':
            sample_dir = '/data/LNOdata/Wave128t1000/Wave128t1000_test1.mat'
            model_file = 'Wave_200ep_LNO-Legendre_n12m4k2_interval100_recheck_test3_model.pp'
        elif PROBLEM == 'Burgers2D':
            sample_dir = '/data/LNOdata/Burgers2D128Re100t1000/Burgers2D128Re100t1000_test1.mat'
            model_file = 'Burgers2D_200ep_LNO-Legendre_n12m6k2_Adam0.001+step_realLoss_interval50_test3_model.pp'
        elif PROBLEM == 'Burgers1D':
            sample_dir = '/data/LNOdata/Burgers128Re100t1000/Burgers362_test4.mat'
            model_file = 'Burgers1D_200ep_LNO-Legendre_n12m6k2_Adam0.001+step_realLoss_interval50_test3_model.pp'


        network = torch.load('models/' + model_file)

        long_round = 100
        record_round = 1
        data = scio.loadmat(sample_dir)
        if PROBLEM == 'Burgers1D':
            Length_x = 362

            inp = np.zeros((1, 1 * in_length, Length_x), np.float32)
            for ii in range(in_length):
                inp[0, 0 + ii * 1, :] = data['u'][t_interval * ii, :-1]
            inp = torch.from_numpy(inp).cuda()

            data_matlab = torch.zeros(long_round // record_round, inp.shape[0], 1, Length_x)
            for i in range(long_round):
                output = network(inp)
                if i % record_round == 0:
                    data_matlab[i // record_round, :, :, :] = output[:, :, :]
                new_in = torch.zeros(inp.shape[0], 1 * in_length, Length_x).cuda()
                home = 1 * (in_length - 1)
                new_in[:, :home, :] = inp[:, 1:, :]
                new_in[:, home:home + 1, :] = output
                inp = new_in

        else:
            Length_y = 181
            Length_x = 128

            inp = np.zeros((1, 2 * in_length, Length_y, Length_x), np.float32)
            if PROBLEM == 'NS' or PROBLEM == 'Burgers2D':
                for ii in range(in_length):
                    inp[0, 0 + ii * 2, :, :] = data['u'][t_interval * ii, :].reshape((Length_y, Length_x))
                    inp[0, 1 + ii * 2, :, :] = data['v'][t_interval * ii, :].reshape((Length_y, Length_x))
            elif PROBLEM == 'Wave':
                for ii in range(in_length):
                    inp[0, 0 + ii * 2, :, :] = data['p'][t_interval * ii, :].reshape((Length_y, Length_x))
                    inp[0, 1 + ii * 2, :, :] = data['dp'][t_interval * ii, :].reshape((Length_y, Length_x))
            inp = torch.from_numpy(inp).cuda()

            data_matlab = torch.zeros(long_round // record_round, inp.shape[0], 2, Length_y, Length_x)
            for ii in range(long_round):
                output = network(inp)
                if ii % record_round == 0:
                    data_matlab[ii // record_round, :, :, :, :] = output[:, :, :, :]
                new_in = torch.zeros(inp.shape[0], 2 * in_length, Length_y, Length_x).cuda()
                home = 2 * (in_length - 1)
                new_in[:, :home, :, :] = inp[:, 2:, :, :]
                new_in[:, home:home + 2, :, :] = output
                inp = new_in

        if PROBLEM == 'NS':
            scio.savemat('outputs/NS128Re{}t1000'.format(Re) + '_output_test1.mat', mdict={"output": data_matlab.detach().numpy()})
        elif PROBLEM == 'Wave':
            scio.savemat('outputs/Wave128t1000_output_recheck_test1.mat', mdict={"output": data_matlab.detach().numpy()})
        elif PROBLEM == 'Burgers2D':
            scio.savemat('outputs/Burgers2D128Re100t1000_output_test1.mat', mdict={"output": data_matlab.detach().numpy()})
        elif PROBLEM == 'Burgers1D':
            scio.savemat('outputs/Burgers128Re100t1000_output_test4.mat', mdict={"output": data_matlab.detach().numpy()})


if __name__ == '__main__':
    train_test_save()
    load_test(args.out_name)
    #load_test_new_sample()
