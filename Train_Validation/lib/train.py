import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np
import scipy.io as io

def train_iterative_NS_InN(network, batch_size,
                           epochs, max_ep, last_ep, optimizer,
                           dataset, train_gen, round,
                           print_frequency,
                           In_length):
    N = 128
    length_in = In_length
    network.train()
    iterations = 500

    losses_epoch = [0 for _ in range(round)]
    optimizer.zero_grad()
    for e in range(epochs):
        Lr = optimizer.param_groups[0]['lr']
        print(' ')
        print('*' * 20)
        print('epoch {}/{}:'.format(last_ep + e + 1, max_ep))
        print('Lr： {}'.format(Lr))
        last_time = time.time()
        losses = [0 for _ in range(round)]
        losses_uv = [0 for _ in range(round)]
        for i in range(iterations):
            data_input, data_output = next(train_gen)

            assert len(data_output) == round

            data_input = data_input.cuda()
            for idx in range(len(data_output)):
                data_output[idx] = data_output[idx].cuda()

            loss_b = 0
            next_in = data_input
            for j in range(round):
                output = network(next_in)
                diff_norm_uv = torch.norm(output[:, 0:2, :, :].reshape(batch_size, - 1) \
                                          - data_output[j][:, 0:2, :, :].reshape(batch_size, -1), 2, 1)
                out_norm_uv = torch.norm(data_output[j][:, 0:2, :, :].reshape(batch_size, -1), 2, 1)
                loss_uv = torch.mean(diff_norm_uv / out_norm_uv)

                loss = torch.mean(diff_norm_uv)
                loss_b += loss

                losses_uv[j] += loss_uv.item() / print_frequency

                losses[j] += loss.item() / print_frequency
                losses_epoch[j] += loss.item() / iterations

                if j + 1 <= round:
                    new_in = torch.zeros(batch_size, 2 * length_in, N, N).cuda()
                    home = 2 * (length_in - 1)
                    new_in[:, :home, :, :] = next_in[:, 2:, :, :]
                    new_in[:, -2:, :, :] = output
                    next_in = new_in

            loss_b.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 5)

            optimizer.step()
            optimizer.zero_grad()

            if i % print_frequency == print_frequency - 1:
                print('iteration={}/{}'.format(i + 1, iterations))
                print('loss={}({})'.format(losses, losses_uv))
                print('time costs per iteration: {:.2f}'.format((time.time() - last_time) / print_frequency))
                last_time = time.time()
                losses = [0 for _ in range(round)]
                losses_uv = [0 for _ in range(round)]

        print('losses_epoch=', losses_epoch)
        losses_epoch = [0 for _ in range(round)]

def train_iterative_Wave_InN(network, batch_size,
                           epochs, max_ep, last_ep, optimizer,
                           dataset, train_gen, round,
                           print_frequency,
                           In_length):
    N = 128
    length_in = In_length
    network.train()
    iterations = 500

    losses_epoch = [0 for _ in range(round)]
    optimizer.zero_grad()
    for e in range(epochs):
        Lr = optimizer.param_groups[0]['lr']
        print(' ')
        print('*' * 20)
        print('epoch {}/{}:'.format(last_ep + e + 1, max_ep))
        print('Lr： {}'.format(Lr))
        last_time = time.time()
        losses = [0 for _ in range(round)]
        losses_p = [0 for _ in range(round)]
        for i in range(iterations):
            data_input, data_output = next(train_gen)

            assert len(data_output) == round

            data_input = data_input.cuda()
            for idx in range(len(data_output)):
                data_output[idx] = data_output[idx].cuda()

            loss_b = 0
            next_in = data_input
            for j in range(round):
                output = network(next_in)
                diff_norm_p = torch.norm(output[:, 0:1, :, :].reshape(batch_size, - 1) \
                                          - data_output[j][:, 0:1, :, :].reshape(batch_size, -1), 2, 1)
                out_norm_p = torch.norm(data_output[j][:, 0:1, :, :].reshape(batch_size, -1), 2, 1)

                loss_p = torch.mean(diff_norm_p / out_norm_p)

                loss = torch.mean(diff_norm_p)
                loss_b += loss

                losses_p[j] += loss_p.item() / print_frequency

                losses[j] += loss.item() / print_frequency
                losses_epoch[j] += loss.item() / iterations

                if j + 1 <= round:
                    new_in = torch.zeros(batch_size, 2 * length_in, N, N).cuda()
                    home = 2 * (length_in - 1)
                    new_in[:, :home, :, :] = next_in[:, 2:, :, :]
                    new_in[:, -2:, :, :] = output
                    next_in = new_in

            loss_b.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 5)

            optimizer.step()
            optimizer.zero_grad()

            if i % print_frequency == print_frequency - 1:
                print('iteration={}/{}'.format(i + 1, iterations))
                print('loss={}({})'.format(losses, losses_p))
                print('time costs per iteration: {:.2f}'.format((time.time() - last_time) / print_frequency))
                last_time = time.time()
                losses = [0 for _ in range(round)]
                losses_p = [0 for _ in range(round)]

        print('losses_epoch=', losses_epoch)
        losses_epoch = [0 for _ in range(round)]

def train_iterative_Burgers1D_InN(network, batch_size,
                           epochs, max_ep, last_ep, optimizer,
                           dataset, train_gen, round,
                           print_frequency,
                           In_length):
    N = 128
    length_in = In_length
    network.train()
    iterations = 500

    losses_epoch = [0 for _ in range(round)]
    optimizer.zero_grad()
    for e in range(epochs):
        Lr = optimizer.param_groups[0]['lr']
        print(' ')
        print('*' * 20)
        print('epoch {}/{}:'.format(last_ep + e + 1, max_ep))
        print('Lr： {}'.format(Lr))
        last_time = time.time()
        losses = [0 for _ in range(round)]
        losses_u = [0 for _ in range(round)]
        for i in range(iterations):
            data_input, data_output = next(train_gen)

            assert len(data_output) == round

            data_input = data_input.cuda()
            for idx in range(len(data_output)):
                data_output[idx] = data_output[idx].cuda()

            loss_b = 0
            next_in = data_input
            for j in range(round):
                output = network(next_in)

                diff_norm_u = torch.norm(output[:, 0:1, :].reshape(batch_size, - 1) \
                                          - data_output[j][:, 0:1, :].reshape(batch_size, -1), 2, 1)
                out_norm_u = torch.norm(data_output[j][:, 0:1, :].reshape(batch_size, -1), 2, 1)
                loss_u = torch.mean(diff_norm_u / out_norm_u)


                loss = torch.mean(diff_norm_u)
                loss_b += loss

                losses_u[j] += loss_u.item() / print_frequency

                losses[j] += loss.item() / print_frequency
                losses_epoch[j] += loss.item() / iterations

                if j + 1 <= round:
                    new_in = torch.zeros(batch_size, 1 * length_in, N).cuda()
                    home = 1 * (length_in - 1)
                    new_in[:, :home, :] = next_in[:, 1:, :]
                    new_in[:, -1:, :] = output
                    next_in = new_in

            loss_b.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 5)

            optimizer.step()
            optimizer.zero_grad()

            if i % print_frequency == print_frequency - 1:
                print('iteration={}/{}'.format(i + 1, iterations))
                print('loss={}({})'.format(losses, losses_u))
                print('time costs per iteration: {:.2f}'.format((time.time() - last_time) / print_frequency))
                last_time = time.time()
                losses = [0 for _ in range(round)]
                losses_u = [0 for _ in range(round)]

        print('losses_epoch=', losses_epoch)
        losses_epoch = [0 for _ in range(round)]