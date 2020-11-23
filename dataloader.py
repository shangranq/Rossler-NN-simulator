from torch.utils.data import Dataset
from RosslerSolver import simulator
import numpy as np

class Rossler:
    def __init__(self, config):
        x0, y0, z0 = config['x0'], config['y0'], config['z0']
        a,  b,  c  = config['a'], config['b'], config['c']
        step,  N   = config['time_step'], config['total_steps']
        X, Y, Z = simulator(x0, y0, z0, a, b, c, step, N)
        self.train_X, self.valid_X, self.test_X = X[:N//2], X[N//2:N*3//4], X[N*3//4:]
        self.train_Y, self.valid_Y, self.test_Y = Y[:N//2], Y[N//2:N*3//4], Y[N*3//4:]
        self.train_Z, self.valid_Z, self.test_Z = Z[:N//2], Z[N//2:N*3//4], Z[N*3//4:]


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





