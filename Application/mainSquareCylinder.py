import torch
import numpy as np
import scipy.io as scio
import gc
import torch.nn.functional as F

with torch.no_grad():
    model_file = 'NS_Re100_model.pp'
    network = torch.load('models/' + model_file)
    network.eval()

    L = 30
    NG = L
    NG_L = 600
    NG_D = 750
    NG_U = 750
    NG_R = 1800
    Length_x = NG_L + NG + NG_R
    Length_y = NG_U + NG + NG_D
    delta_x = 2 / 128

    u_lid = 40 / 100 / 2 * 128 / L

    Length_x += network.get_padding_R(Length_x + 2 * network.recept + 1 - 2)
    Length_y += network.get_padding_R(Length_y + 2 * network.recept + 1 - 2)

    xMin = 0
    xMax = xMin + delta_x * NG
    yMin = -0.5
    yMax = yMin + delta_x * NG
    print('length_x=', Length_x)
    print('length_y=', Length_y)


    '''Initial condition'''
    u_NN = np.ones((Length_y + 1, Length_x + 1), dtype='float32') * u_lid
    v_NN = np.ones((Length_y + 1, Length_x + 1), dtype='float32') * u_lid * 0
    u_NN = torch.from_numpy(u_NN).cuda()
    v_NN = torch.from_numpy(v_NN).cuda()

    r = network.recept
    output = torch.zeros(1, 2, Length_y + 1, Length_x + 1).cuda()

    for cycle in range(2000):

        u_NN = F.pad(u_NN, (r, r, r, r), mode='constant', value=u_lid)
        v_NN = F.pad(v_NN, (r, r, r, r), mode='constant', value=u_lid * 0)

        input = torch.stack((u_NN, v_NN))
        input = torch.unsqueeze(input, 0)

        output[:, :, :, :] = network(input[:, :, :, :])
        u_NN = output[0, 0, :, :]
        v_NN = output[0, 1, :, :]

        u_NN[NG_D:NG_D + NG + 1, NG_L:NG_L + NG + 1] = 0
        v_NN[NG_D:NG_D + NG + 1, NG_L:NG_L + NG + 1] = 0

        if cycle % 20 == 0:
            print('Cycle=', cycle)
            scio.savemat('SC_' + str(cycle) + '.mat', mdict={'u': u_NN.cpu().detach().numpy(),
                                                             'v': v_NN.cpu().detach().numpy()})

        gc.collect()


