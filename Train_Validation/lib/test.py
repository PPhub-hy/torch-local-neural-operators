import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio

def test_iterative_NS_InN(network, dataset, test_gen, round, long_round, out_name, In_length):
    N = 128
    length_in = In_length

    network.eval()
    data_input, data_output = next(test_gen)
    batch_size = data_input.shape[0]
    iterations = sum(dataset.sample_nums['test']) // batch_size
    losses = [0 for _ in range(round)]
    with torch.no_grad():
        for _ in range(iterations):
            assert len(data_output) == round

            data_input = data_input.cuda()
            for idx in range(len(data_output)):
                data_output[idx] = data_output[idx].cuda()

            next_in = data_input
            for j in range(round):
                output = network(next_in)
                diff_norm = torch.norm(output[:, 0:2, :, :].reshape(batch_size, - 1) \
                                       - data_output[j][:, 0:2, :, :].reshape(batch_size, -1), 2, 1)
                out_norm = torch.norm(data_output[j][:, 0:2, :, :].reshape(batch_size, -1), 2, 1)
                loss = torch.mean(diff_norm/out_norm)
                losses[j] += loss.item() / iterations

                if j + 1 <= round:
                    new_in = torch.zeros(batch_size, 2 * length_in, N, N).cuda()
                    home = 2 * (length_in - 1)
                    new_in[:, :home, :, :] = next_in[:, 2:, :, :]
                    new_in[:, -2:, :, :] = output
                    next_in = new_in.detach()

            data_input, data_output = next(test_gen)

        print('loss=', losses)

        record_round = 1
        NG = data_input.shape[3]
        inp = dataset.load_test_input(length_in)
        inp = inp.cuda()

        data_matlab = torch.zeros(long_round // record_round, inp.shape[0], 2, NG, NG)
        for i in range(long_round):
            output = network(inp)
            if i % record_round == 0:
                data_matlab[i // record_round, :, :, :, :] = output[:, :, :, :]
            new_in = torch.zeros(inp.shape[0], 2 * length_in, NG, NG).cuda()
            home = 2 * (length_in - 1)
            new_in[:, :home, :, :] = inp[:, 2:, :, :]
            new_in[:, home:home + 2, :, :] = output
            inp = new_in

    scio.savemat('outputs/'+out_name, mdict={"output": data_matlab.detach().numpy()})

def test_iterative_Wave_InN(network, dataset, test_gen, round, long_round, out_name, In_length):
    N = 128
    length_in = In_length

    network.eval()
    data_input, data_output = next(test_gen)
    batch_size = data_input.shape[0]
    iterations = sum(dataset.sample_nums['test']) // batch_size
    losses = [0 for _ in range(round)]
    with torch.no_grad():
        for _ in range(iterations):
            assert len(data_output) == round

            data_input = data_input.cuda()
            for idx in range(len(data_output)):
                data_output[idx] = data_output[idx].cuda()

            next_in = data_input
            for j in range(round):
                output = network(next_in)
                diff_norm = torch.norm(output[:, 0:1, :, :].reshape(batch_size, - 1) \
                                       - data_output[j][:, 0:1, :, :].reshape(batch_size, -1), 2, 1)
                out_norm = torch.norm(data_output[j][:, 0:1, :, :].reshape(batch_size, -1), 2, 1)
                loss = torch.mean(diff_norm/out_norm)
                losses[j] += loss.item() / iterations

                if j + 1 <= round:
                    new_in = torch.zeros(batch_size, 2 * length_in, N, N).cuda()
                    home = 2 * (length_in - 1)
                    new_in[:, :home, :, :] = next_in[:, 2:, :, :]
                    new_in[:, -2:, :, :] = output

                    next_in = new_in.detach()

            data_input, data_output = next(test_gen)

        print('loss=', losses)

        record_round = 1
        NG = data_input.shape[3]
        inp = dataset.load_test_input(length_in)
        inp = inp.cuda()

        data_matlab = torch.zeros(long_round // record_round, inp.shape[0], 2, NG, NG)
        for i in range(long_round):
            output = network(inp)
            if i % record_round == 0:
                data_matlab[i // record_round, :, :, :, :] = output[:, :, :, :]
            new_in = torch.zeros(inp.shape[0], 2 * length_in, NG, NG).cuda()
            home = 2 * (length_in - 1)
            new_in[:, :home, :, :] = inp[:, 2:, :, :]
            new_in[:, home:home + 2, :, :] = output
            inp = new_in
    scio.savemat('outputs/'+out_name, mdict={"output": data_matlab.detach().numpy()})

def test_iterative_Burgers1D_InN(network, dataset, test_gen, round, long_round, out_name, In_length):
    N = 128
    length_in = In_length

    network.eval()
    data_input, data_output = next(test_gen)
    batch_size = data_input.shape[0]
    iterations = sum(dataset.sample_nums['test']) // batch_size
    losses = [0 for _ in range(round)]
    with torch.no_grad():
        for _ in range(iterations):
            assert len(data_output) == round

            data_input = data_input.cuda()
            for idx in range(len(data_output)):
                data_output[idx] = data_output[idx].cuda()

            next_in = data_input
            for j in range(round):
                output = network(next_in)
                diff_norm = torch.norm(output[:, 0:1, :].reshape(batch_size, - 1) \
                                       - data_output[j][:, 0:1, :].reshape(batch_size, -1), 2, 1)
                out_norm = torch.norm(data_output[j][:, 0:1, :].reshape(batch_size, -1), 2, 1)
                loss = torch.mean(diff_norm/out_norm)
                losses[j] += loss.item() / iterations

                if j + 1 <= round:
                    new_in = torch.zeros(batch_size, 1 * length_in, N).cuda()
                    home = 1 * (length_in - 1)
                    new_in[:, :home, :] = next_in[:, 1:, :]
                    new_in[:, -1:, :] = output
                    next_in = new_in.detach()

            data_input, data_output = next(test_gen)

        print('loss=', losses)

        record_round = 1
        NG = data_input.shape[-1]
        inp = dataset.load_test_input(length_in)
        inp = inp.cuda()


        data_matlab = torch.zeros(long_round // record_round, inp.shape[0], 1, NG)
        for i in range(long_round):
            output = network(inp)
            if i % record_round == 0:
                data_matlab[i // record_round, :, :, :] = output[:, :, :]
            new_in = torch.zeros(inp.shape[0], 1 * length_in, NG).cuda()
            home = 1 * (length_in - 1)
            new_in[:, :home, :] = inp[:, 1:, :]
            new_in[:, home:home + 1, :] = output
            inp = new_in

    scio.savemat('outputs/'+out_name, mdict={"output": data_matlab.detach().numpy()})
