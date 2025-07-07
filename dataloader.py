import numpy as np
import torch
import h5py
from sklearn.preprocessing import MinMaxScaler
import random
import warnings
warnings.filterwarnings("ignore")


class TrainDataset_All(torch.utils.data.Dataset):
    def __init__(self, X, Y, Miss_list, Idxs):
        self.X = X
        self.Y = Y
        self.Miss_list = Miss_list
        self.Idxs = Idxs
        self.view_size = len(X)

    def __getitem__(self, index):
        return [self.X[i][index] for i in range(self.view_size)], \
               [self.Y[i][index] for i in range(self.view_size)], \
               [self.Miss_list[i][index] for i in range(self.view_size)], \
               self.Idxs[index]

    def __len__(self):
        return self.X[0].shape[0]


def get_mask(view_num, alldata_len, missing_rate):
    miss_mat = np.ones((alldata_len, view_num))
    b=((10 - 10 * missing_rate) / 10) * alldata_len
    miss_begin = int(b)
    for i in range(miss_begin, alldata_len):
        missdata = np.random.randint(0, high=view_num, size=view_num - 1)
        miss_mat[i, missdata] = 0
    miss_mat = torch.tensor(miss_mat, dtype=torch.int)

    return miss_mat


def load_data(data_name, missrate):
    path = 'D:/MultiView Dataset/'
    data = h5py.File(path + data_name + ".mat")
    X, Y = [], []
    Label = np.array(data['Y']).T
    Label = Label.reshape(Label.shape[0])
    mm = MinMaxScaler()

    for i in range(data['X'].shape[1]):
        diff_view = data[data['X'][0, i]]
        diff_view = np.array(diff_view, dtype=np.float32).T
        std_view = mm.fit_transform(diff_view)
        X.append(std_view)
        Y.append(Label)

    input_dims = []
    for i in range(len(X)):
        input_dims.append(X[i].shape[1])

    unique = np.unique(Y[0])
    cluster_num = np.size(unique, axis=0)
    data_num = len(Y[0])
    view_num = len(X)
    view_dims = input_dims

    index = [i for i in range(data_num)]
    np.random.shuffle(index)
    for v in range(view_num):
        X[v] = X[v][index]
        Y[v] = Y[v][index]

    Miss_mat = get_mask(view_num, data_num, missrate)
    Miss_vecs = [row for row in Miss_mat.T]

    return X, Y, Miss_vecs, cluster_num, data_num, view_num, view_dims




