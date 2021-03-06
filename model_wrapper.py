from dataloader import Rossler, dataset
from torch.utils.data import DataLoader
from visual import show_window, show_a_test_window, show_long_window, trajectory, project_xy, project_yz, project_xz
from models import MLP, MLP_3D, LSTM, Transformer
import torch
import torch.nn as nn
import os
import numpy as np
import torch.optim as optim
from utils import cast_to_float
from glob import glob
from torch.utils.tensorboard import SummaryWriter
from scipy.interpolate import interp1d


class ModelDev:

    def __init__(self, config):
        self.config = config
        self.prepare_dataloaders(config['data'])

        # self.model = MLP(config['MLP'])
        # self.model = MLP_3D(config['MLP'])
        # self.model = LSTM(config['LSTM'])
        self.model = Transformer(config['Trans'])
        print(self.model)

        self.model_name = config['train']['model_name']

        self.checkpoint_dir = './checkpoint_dir/{}/'.format(self.model_name)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.tb_log_dir = './tb_log/{}/'.format(self.model_name)
        if not os.path.exists(self.tb_log_dir):
            os.mkdir(self.tb_log_dir)

        self.optimal_metric = 100000
        self.cur_metric = 100000

        self.loss = nn.MSELoss()
        self.optim = optim.Adam(self.model.parameters(), lr=self.config['train']['lr'], betas=(0.5, 0.999))


    def prepare_dataloaders(self, config):
        data = Rossler(config)
        train_data = dataset(data.train_X, data.train_Y, data.train_Z, config['w_size'])
        self.train_dataloader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, drop_last=True)
        valid_data = dataset(data.valid_X, data.valid_Y, data.valid_Z, config['w_size'])
        self.valid_dataloader = DataLoader(valid_data, batch_size=config['batch_size'], shuffle=False, drop_last=True)
        test_data  = dataset(data.test_X, data.test_Y, data.test_Z, config['w_size'])
        self.test_dataloader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False, drop_last=True)
        self.data = data


    def train(self):
        self.writer = SummaryWriter(self.tb_log_dir)
        for self.epoch in range(self.config['train']['epochs']):
            self.train_on_epoch()
            self.cur_metric = self.valid_on_epoch()
            print(self.cur_metric)
            if self.needToSave():
                self.saveWeights()


    def train_on_epoch(self):
        self.model.train(False)
        LOSS = []
        for X_i, Y_i, Z_i, X_o, Y_o, Z_o in self.train_dataloader:
            X_i, Y_i, Z_i, X_o, Y_o, Z_o = cast_to_float(X_i, Y_i, Z_i, X_o, Y_o, Z_o)
            X_i += torch.normal(0, 0.1, X_i.shape)
            self.model.zero_grad()
            pred = self.model(X_i)
            loss = self.loss(pred, X_o)
            loss.backward()
            self.optim.step()
            LOSS.append(loss.data.cpu().numpy())
        self.writer.add_scalar('train Loss', np.mean(LOSS), self.epoch)


    def valid_on_epoch(self):
        self.model.train(False)
        LOSS = []
        with torch.no_grad():
            for X_i, Y_i, Z_i, X_o, Y_o, Z_o in self.valid_dataloader:
                X_i, Y_i, Z_i, X_o, Y_o, Z_o = cast_to_float(X_i, Y_i, Z_i, X_o, Y_o, Z_o)
                pred = self.model(X_i)
                loss = self.loss(pred, X_o)
                LOSS.append(loss.data.cpu().numpy())
        self.writer.add_scalar('valid Loss', np.mean(LOSS), self.epoch)
        return np.mean(LOSS)


    def cast_to_float(self, X_i, Y_i, Z_i, X_o, Y_o, Z_o):
        X_i = X_i.float()
        Y_i = Y_i.float()
        Z_i = Z_i.float()
        X_o = X_o.float()
        Y_o = Y_o.float()
        Z_o = Z_o.float()
        return X_i, Y_i, Z_i, X_o, Y_o, Z_o


    def needToSave(self):
        if self.cur_metric < self.optimal_metric:
            self.optimal_metric = self.cur_metric
            return True
        return False


    def saveWeights(self, clean_previous=True):
        if clean_previous:
            files = glob(self.checkpoint_dir + '*.pth')
            for f in files:
                os.remove(f)
        torch.save(self.model.state_dict(), '{}model_{}.pth'.format(self.checkpoint_dir, self.epoch))


    def test_MSE(self):
        self.model.train(False)
        self.load_weights()
        LOSS = []
        with torch.no_grad():
            for X_i, Y_i, Z_i, X_o, Y_o, Z_o in self.test_dataloader:
                X_i, Y_i, Z_i, X_o, Y_o, Z_o = cast_to_float(X_i, Y_i, Z_i, X_o, Y_o, Z_o)
                pred = self.model(X_i)
                loss = self.loss(pred, X_o)
                LOSS.append(loss.data.cpu().numpy())
        return np.mean(LOSS)


    def test_a_window(self):
        self.model.train(False)
        self.load_weights()
        idx = 0
        with torch.no_grad():
            for X_i, Y_i, Z_i, X_o, Y_o, Z_o in self.test_dataloader:
                X_i, Y_i, Z_i, X_o, Y_o, Z_o = cast_to_float(X_i, Y_i, Z_i, X_o, Y_o, Z_o)
                pred = self.model(X_i)
                show_a_test_window(X_i.data.numpy()[0, :], X_o.data.numpy()[0, :], pred.data.numpy()[0, :], idx, self.config['data']['stride'])
                idx += 1


    def test_long_window(self, length):
        self.model.train(False)
        self.load_weights()
        for start_idx in [100, 200, 300, 400, 500]:
            X_I = self.data.test_X[start_idx:start_idx+length]
            X_pred = X_I[:self.config['data']['w_size']-1]
            with torch.no_grad():
                while len(X_pred) < len(X_I):
                    nparray = np.array(X_pred[-self.config['data']['w_size']+1:])
                    nparray = np.expand_dims(nparray, axis=0)
                    torchTensor = torch.FloatTensor(nparray)
                    pred = self.model(torchTensor).data.squeeze().numpy().tolist()
                    X_pred.append(pred)
            show_long_window(X_I, X_pred, self.config['data']['stride'], self.config['data']['w_size'], start_idx, self.config['train']['model_name'])


    def load_weights(self):
        target_file = list(glob(self.checkpoint_dir + 'model*.pth'))[0]
        print('loading ', target_file)
        weights = torch.load(target_file)
        self.model.load_state_dict(weights)



class ModelDev_3D(ModelDev):

    def train_on_epoch(self):
        self.model.train(False)
        LOSS = []
        for X_i, Y_i, Z_i, X_o, Y_o, Z_o in self.train_dataloader:
            X_i, Y_i, Z_i, X_o, Y_o, Z_o = cast_to_float(X_i, Y_i, Z_i, X_o, Y_o, Z_o)
            X_i, Y_i, Z_i = self.augment(X_i, Y_i, Z_i)
            self.model.zero_grad()
            pred = self.model(X_i, Y_i, Z_i)
            label = torch.cat((X_o, Y_o, Z_o), dim=1)
            loss = self.loss(pred, label)
            loss.backward()
            self.optim.step()
            LOSS.append(loss.data.cpu().numpy())
        self.writer.add_scalar('train Loss', np.mean(LOSS), self.epoch)

    def valid_on_epoch(self):
        self.model.train(False)
        LOSS = []
        with torch.no_grad():
            for X_i, Y_i, Z_i, X_o, Y_o, Z_o in self.valid_dataloader:
                X_i, Y_i, Z_i, X_o, Y_o, Z_o = cast_to_float(X_i, Y_i, Z_i, X_o, Y_o, Z_o)
                pred = self.model(X_i, Y_i, Z_i)
                label = torch.cat((X_o, Y_o, Z_o), dim=1)
                loss = self.loss(pred, label)
                LOSS.append(loss.data.cpu().numpy())
        self.writer.add_scalar('valid Loss', np.mean(LOSS), self.epoch)
        return np.mean(LOSS)

    def test_MSE(self):
        self.model.train(False)
        self.load_weights()
        LOSS = []
        with torch.no_grad():
            for X_i, Y_i, Z_i, X_o, Y_o, Z_o in self.test_dataloader:
                X_i, Y_i, Z_i, X_o, Y_o, Z_o = cast_to_float(X_i, Y_i, Z_i, X_o, Y_o, Z_o)
                pred = self.model(X_i, Y_i, Z_i)
                label = torch.cat((X_o, Y_o, Z_o), dim=1)
                loss = self.loss(pred, label)
                LOSS.append(loss.data.cpu().numpy())
        return np.mean(LOSS)

    def augment(self, X_i, Y_i, Z_i):
        X_i += torch.normal(0, 0.1, X_i.shape)
        Y_i += torch.normal(0, 0.1, Y_i.shape)
        Z_i += torch.normal(0, 0.1, Z_i.shape)
        return X_i, Y_i, Z_i

    def test_long_window(self, length):
        self.model.train(False)
        self.load_weights()
        for start_idx in [100, 200, 300, 400, 500]:
            X_I = self.data.test_X[start_idx:start_idx+length]
            Y_I = self.data.test_Y[start_idx:start_idx+length]
            Z_I = self.data.test_Z[start_idx:start_idx+length]
            X_pred = X_I[:self.config['data']['w_size']-1]
            Y_pred = Y_I[:self.config['data']['w_size']-1]
            Z_pred = Z_I[:self.config['data']['w_size']-1]
            with torch.no_grad():
                while len(X_pred) < len(X_I):
                    x_i = self.prepare_tensor(X_pred[-self.config['data']['w_size']+1:])
                    y_i = self.prepare_tensor(Y_pred[-self.config['data']['w_size']+1:])
                    z_i = self.prepare_tensor(Z_pred[-self.config['data']['w_size']+1:])
                    pred = self.model(x_i, y_i, z_i).data.squeeze().numpy().tolist()
                    X_pred.append(pred[0])
                    Y_pred.append(pred[1])
                    Z_pred.append(pred[2])
            show_long_window(X_I, X_pred, self.config['data']['stride'], self.config['data']['w_size'], start_idx, self.config['train']['model_name'], 'x')
            show_long_window(Y_I, Y_pred, self.config['data']['stride'], self.config['data']['w_size'], start_idx, self.config['train']['model_name'], 'y')
            show_long_window(Z_I, Z_pred, self.config['data']['stride'], self.config['data']['w_size'], start_idx, self.config['train']['model_name'], 'z')

    def show_3D_trajectory(self):
        self.model.train(False)
        self.load_weights()
        X_I = self.data.test_X[:100]
        Y_I = self.data.test_Y[:100]
        Z_I = self.data.test_Z[:100]
        X_pred = X_I[:self.config['data']['w_size'] - 1]
        Y_pred = Y_I[:self.config['data']['w_size'] - 1]
        Z_pred = Z_I[:self.config['data']['w_size'] - 1]
        with torch.no_grad():
            while len(X_pred) < 1000:
                x_i = self.prepare_tensor(X_pred[-self.config['data']['w_size'] + 1:])
                y_i = self.prepare_tensor(Y_pred[-self.config['data']['w_size'] + 1:])
                z_i = self.prepare_tensor(Z_pred[-self.config['data']['w_size'] + 1:])
                pred = self.model(x_i, y_i, z_i).data.squeeze().numpy().tolist()
                X_pred.append(pred[0])
                Y_pred.append(pred[1])
                Z_pred.append(pred[2])
        t = [i for i in range(len(X_pred))]
        fx = interp1d(t, X_pred, kind='cubic')
        fy = interp1d(t, Y_pred, kind='cubic')
        fz = interp1d(t, Z_pred, kind='cubic')
        t_new = np.linspace(0, len(X_pred)-1, num=10 * len(X_pred), endpoint=True)
        X_new = fx(t_new)
        Y_new = fy(t_new)
        Z_new = fz(t_new)
        trajectory(X_new, Y_new, Z_new, self.config['train']['model_name'])
        project_yz(X_new, Y_new, Z_new, self.config['train']['model_name'])
        project_xz(X_new, Y_new, Z_new, self.config['train']['model_name'])
        project_xy(X_new, Y_new, Z_new, self.config['train']['model_name'])


    def prepare_tensor(self, pred):
        nparray = np.array(pred[-self.config['data']['w_size'] + 1:])
        nparray = np.expand_dims(nparray, axis=0)
        return torch.FloatTensor(nparray)
