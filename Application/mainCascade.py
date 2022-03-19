import torch
import numpy as np
import scipy.io as scio
import gc
import time
import torch.nn.functional as F
from IBMInterpolation import *




with torch.no_grad():

    alpha = -10 / 180 * np.pi  # Angle of attack

    model_file = 'No3-0102_model.pp'
    network = torch.load('models/' + model_file)
    network.eval()

    NG_L = 400
    NG_D = 32
    NG_U = 32
    NG_R = 1200
    Length_x = NG_L+NG_R
    Length_y = NG_U+NG_D
    delta_x = 2/128

    u_lid = 1

    Length_x += network.get_padding_R(Length_x+2*network.recept + 1 - 2)
    Length_y += network.get_padding_R(Length_y+2*network.recept + 1 - 2)


    xMin = 0
    xMax = xMin + delta_x * Length_x
    yMin = -0.5
    yMax = yMin + delta_x * Length_y
    print('length_x=',Length_x)
    print('length_y=',Length_y)

    '''Load airfoil'''
    filename = 'NACA0012_20.mat'
    this_raw_data = scio.loadmat(filename)
    delta_s = this_raw_data['delta_s']
    p_e = np.zeros((Length_y + 1, Length_x + 1,2), dtype='float32')
    for i in range(Length_y+1):
        for j in range(Length_x+1):
            p_e[i,j,0]=j*delta_x+xMin
            p_e[i, j, 1] = i * delta_x + yMin
    p_e = p_e.reshape([(Length_y + 1)*(Length_x + 1),2])
    p_l = this_raw_data['p_l']
    p_l[:,0] += xMin+NG_L*delta_x
    p_l[:, 1] += yMin + Length_y/2 * delta_x

    IBM = ClassicIBM(delta_x, delta_s, p_e, p_l)

    '''Initial condition'''
    u_NN = np.ones((Length_y + 1, Length_x + 1), dtype='float32')*u_lid*np.cos(alpha)
    v_NN = np.ones((Length_y + 1, Length_x + 1), dtype='float32')*u_lid*np.sin(alpha)
    u_NN = torch.from_numpy(u_NN).cuda()
    v_NN = torch.from_numpy(v_NN).cuda()


    r = network.recept
    output = torch.zeros(1, 2, Length_y + 1, Length_x + 1).cuda()
    for cycle in range(2000):
        u_NN = F.pad(u_NN,(r,r,0,0),mode='constant', value=u_lid*np.cos(alpha))
        v_NN = F.pad(v_NN, (r, r, 0, 0), mode='constant', value=u_lid*np.sin(alpha))
        input = torch.stack((u_NN, v_NN))
        input = torch.unsqueeze(input, 0)
        input = F.pad(input, (0, 0, r, r), mode='circular')

        output = network(input)

        u_NN = output[0, 0, :, :]
        v_NN = output[0, 1, :, :]
        u_NN, v_NN = IBM.step(u_NN,v_NN,Length_x, Length_y)

        if cycle % 20 == 0:
            print('Cycle=',cycle)
            scio.savemat('Cascade_' + str(cycle) + '.mat', mdict={'u': u_NN.cpu().detach().numpy(),
                                                              'v': v_NN.cpu().detach().numpy()})
        gc.collect()


