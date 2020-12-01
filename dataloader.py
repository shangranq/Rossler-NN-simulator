from torch.utils.data import Dataset, DataLoader
from RosslerSolver import simulator
import numpy as np
from utils import read_json


class Rossler:
    def __init__(self, config):
        x0, y0, z0 = config['x0'], config['y0'], config['z0']
        a,  b,  c  = config['a'], config['b'], config['c']
        step, N, s = config['time_step'], config['total_steps'], config['stride']
        X, Y, Z = simulator(x0, y0, z0, a, b, c, step, N)
        self.train_X, self.valid_X, self.test_X = X[:N//2:s], X[N//2:N*3//4:s], X[N*3//4::s]
        self.train_Y, self.valid_Y, self.test_Y = Y[:N//2:s], Y[N//2:N*3//4:s], Y[N*3//4::s]
        self.train_Z, self.valid_Z, self.test_Z = Z[:N//2:s], Z[N//2:N*3//4:s], Z[N*3//4::s]


class dataset(Dataset):
    def __init__(self, X, Y, Z, window_size):
        self.X, self.Y, self.Z = np.array(X), np.array(Y), np.array(Z)
        self.w_size = window_size

    def __len__(self):
        return self.X.shape[0] - self.w_size

    def __getitem__(self, idx):
        X_input, X_output = self.X[idx:idx + self.w_size - 1], self.X[idx + self.w_size - 1:idx + self.w_size]
        Y_input, Y_output = self.Y[idx:idx + self.w_size - 1], self.Y[idx + self.w_size - 1:idx + self.w_size]
        Z_input, Z_output = self.Z[idx:idx + self.w_size - 1], self.Z[idx + self.w_size - 1:idx + self.w_size]
        return X_input, Y_input, Z_input, X_output, Y_output, Z_output


if __name__ == '__main__':
    config = read_json('config.json')["data"]
    data = Rossler(config)
    train_data = dataset(data.train_X, data.train_Y, data.train_Z, config['w_size'])
    train_dataloader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    for X_input, Y_input, Z_input, X_output, Y_output, Z_output in train_dataloader:
        print(X_input.shape, X_output.shape)





