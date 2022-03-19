import numpy as np
import random
import torch
import scipy.io as scio
import gc
import pickle
import os

NG = 128 #一个方向划分的单元数

class Wave_Dataset2D:

    def __init__(self,
                 data_dir,
                 data_name,
                 orders_train,
                 orders_test,
                 t_interval):
        self.L = NG
        self.data_name = data_name
        self.data_dir = data_dir + data_name + '/'
        self.cache_dir = self.data_dir + 'Wave_cache_inter{}_test{}/'.format(t_interval, orders_test)
        self.param_dir = self.data_dir + 'Wave_cache_inter{}_test{}/params.param'.format(t_interval, orders_test)
        self.orders_train = orders_train
        self.orders_test = orders_test
        self.all_orders = orders_train + orders_test
        self.data_lists = {'train': [], 'test': []}
        self.cache_names = {'train': [], 'test': []}
        self.sample_nums = {'train': [], 'test': []}
        self.test_in = {order: [] for order in orders_test}
        self.t_interval = t_interval
        self.part_num = 0

        if os.path.exists(self.cache_dir):
            self.param_load_init(self.param_dir)
        else:
            os.makedirs(self.cache_dir)
            self.load_dataset_caches()
            self.param_save(self.param_dir)

    def param_load_init(self, path):
        with open(path, 'rb') as f:
            parameters = pickle.load(f)

        self.data_dir = parameters['data_dir']
        self.orders_train = parameters['orders_train']
        assert self.orders_test == parameters['orders_test']
        self.all_orders = parameters['all_orders']
        self.cache_names = parameters['cache_names']
        self.sample_nums = parameters['sample_nums']
        self.test_in = parameters['test_in']
        self.part_num = parameters['part_num']
        self.t_interval = parameters['t_interval']

        print(' dataset parameters loaded from {}'.format(path))

    def param_save(self, path):
        parameters = {}
        parameters['data_dir'] = self.data_dir
        parameters['orders_train'] = self.orders_train
        parameters['orders_test'] = self.orders_test
        parameters['all_orders'] = self.all_orders
        parameters['cache_names'] = self.cache_names
        parameters['sample_nums'] = self.sample_nums
        parameters['test_in'] = self.test_in
        parameters['part_num'] = self.part_num
        parameters['t_interval'] = self.t_interval

        with open(path, 'wb') as f:
            pickle.dump(parameters, f)
        print(' dataset parameters saved!')

    def save_cache_file(self, split, test_order = None):
        self.part_num += 1
        cache_name = 'part{}_{}'.format(self.part_num, split)
        if split=='test':
            cache_name += test_order
        cache_file = self.cache_dir + cache_name + '.Wavecache'
        with open(cache_file, 'wb') as f:
            pickle.dump(self.data_lists[split], f)
        self.cache_names[split].append(cache_name)
        self.sample_nums[split] += [len(self.data_lists[split])]
        print('cache file saved at ', cache_file)
        self.data_lists[split] = []

    def load_cache_file(self, cache_name):
        cache_file = self.cache_dir + cache_name + '.Wavecache'
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        return data

    def data_augmentation(self, p0, dp0, type):
        if type == 0:
            return p0, dp0
        else:
            pp = np.zeros(NG * NG, np.float32)
            dpp = np.zeros(NG * NG, np.float32)

        if type == 1:
            for i in range(NG):
                for j in range(NG):
                    pp[i * NG + j] = p0[j * NG + NG - i - 1]
                    dpp[i * NG + j] = dp0[j * NG + NG - i - 1]
            return pp, dpp
            #return u0, v0, p0

        elif type == 2:
            for i in range(NG):
                for j in range(NG):
                    pp[i * NG + j] = p0[(NG - i - 1) * NG + NG - j - 1]
                    dpp[i * NG + j] = dp0[(NG - i - 1) * NG + NG - j - 1]
            return pp, dpp

        elif type == 3:
            for i in range(NG):
                for j in range(NG):
                    pp[i * NG + j] = p0[(NG - j - 1) * NG + i]
                    dpp[i * NG + j] = dp0[(NG - j - 1) * NG + i]
            return pp, dpp

        elif type == 4:
            for i in range(NG):
                for j in range(NG):
                    pp[i * NG + j] = p0[i * NG + NG - j - 1]
                    dpp[i * NG + j] = dp0[i * NG + NG - j - 1]
            return pp, dpp

        elif type == 5:
            for i in range(NG):
                for j in range(NG):
                    pp[i * NG + j] = p0[(NG - i - 1) * NG + j]
                    dpp[i * NG + j] = dp0[(NG - i - 1) * NG + j]
            return pp, dpp

        elif type == 6:
            for i in range(NG):
                for j in range(NG):
                    pp[i * NG + j] = p0[(NG - j - 1) * NG + NG - i - 1]
                    dpp[i * NG + j] = dp0[(NG - j - 1) * NG + NG - i - 1]
            return pp, dpp

        elif type == 7:
            for i in range(NG):
                for j in range(NG):
                    pp[i * NG + j] = p0[j * NG + i]
                    dpp[i * NG + j] = dp0[j * NG + i]
            return pp, dpp

    def load_dataset_caches(self):
        sampling_gap = self.t_interval
        dataname = self.data_name
        sample_length = 30

        for order in self.all_orders:
            filename = self.data_dir + dataname + '_' + str(order) + '.mat' #.zfill(2)
            this_raw_data = scio.loadmat(filename)

            if order in self.orders_train:
                split = 'train'
            elif order in self.orders_test:
                split = 'test'

            print('loading {} for {}'.format(filename, split))
            for aug_i in range(8):
                print('augmenting type {}'.format(aug_i))
                this_list = []
                for i in range(this_raw_data['p'].shape[0] // sampling_gap):
                    idx = i * sampling_gap
                    p = this_raw_data['p'][idx, :]
                    dp = this_raw_data['dp'][idx, :]
                    p, dp = self.data_augmentation(p, dp, aug_i)
                    this_list.append({'p': p.copy().reshape((NG, NG)),
                                      'dp': dp.copy().reshape((NG, NG))})

                    if len(this_list) == sample_length:
                        self.data_lists[split].append(this_list)
                        if idx >= this_raw_data['p'].shape[0] - sampling_gap * sample_length:
                            break
                        else:
                            this_list = []
                #if split == 'test':
                #    break

            del this_raw_data
            gc.collect()

            if split == 'test':
                self.save_cache_file(split, str(order))
                continue
            if len(self.data_lists[split]) >= 1000:
                self.save_cache_file(split)

        if not len(self.data_lists['train']) == 0:
            self.save_cache_file('train')

        in_interval = self.t_interval
        self.t_interval = int(self.t_interval / sampling_gap)
        assert self.t_interval * sampling_gap == in_interval

    def load_test_input(self, in_length):
        test_input = np.zeros((len(self.cache_names['test']), 2 * in_length, NG, NG), np.float32)
        for idx in range(len(self.cache_names['test'])):
            data = self.load_cache_file(self.cache_names['test'][idx])
            for ii in range(in_length):
                test_input[idx][0 + ii] = data[0][0 + self.t_interval * ii]['p']
                test_input[idx][1 + ii] = data[0][0 + self.t_interval * ii]['dp']
        return torch.from_numpy(test_input)

    def data_generator_series(self, out_length, in_length, batch_size, split='train'):

        assert out_length >= 1
        length = in_length + out_length - 1
        cache_names = self.cache_names[split].copy()

        cache_idx = -1
        data_idx = -1
        data = [0]

        while True:
            batch_in = np.zeros((batch_size, 2 * in_length, NG, NG), np.float32)
            batch_out = [np.zeros((batch_size, 2, NG, NG), np.float32) for _ in range(out_length)]

            for i in range(batch_size):
                data_idx = (data_idx + 1) % len(data)
                if data_idx == 0:
                    cache_idx = (cache_idx + 1) % len(cache_names)
                    if cache_idx == 0:
                        random.shuffle(cache_names)
                        print('cache index shuffled! ')
                        data = self.load_cache_file(cache_names[cache_idx])
                    else:
                        data = self.load_cache_file(cache_names[cache_idx])
                    random.shuffle(data)
                    print('data shuffled! ')

                start_idx = random.randint(0, len(data[0]) - length - 1)
                for ii in range(in_length):
                    batch_in[i][0 + ii] = data[data_idx][start_idx + self.t_interval * ii]['p']
                    batch_in[i][1 + ii] = data[data_idx][start_idx + self.t_interval * ii]['dp']
                for j in range(out_length):
                    batch_out[j][i][0] = data[data_idx][start_idx + self.t_interval * (j + in_length)]['p']
                    batch_out[j][i][1] = data[data_idx][start_idx + self.t_interval * (j + in_length)]['dp']
            for idx in range(len(batch_out)):
                batch_out[idx] = torch.from_numpy(batch_out[idx])

            yield torch.from_numpy(batch_in), batch_out

if __name__ == '__main__':
    dataset = Burgers_Dataset2D(data_dir='./', orders_train=[1], orders_test=[1], t_interval=1)
    # print(dataset.raw_datas[1]['u'].shape)

    x, y =next(dataset.data_generator())
    print(x[0][0])
    print(x[0][1])
    print(x[0][4])
