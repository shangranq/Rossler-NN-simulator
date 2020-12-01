from dataloader import Rossler, dataset
from torch.utils.data import DataLoader
from visual import show_window
from models import MLP
import torch
import torch.nn as nn
import os
import numpy as np
import torch.optim as optim
from utils import cast_to_float
from glob import glob
from torch.utils.tensorboard import SummaryWriter


class ModelDev:

    def __init__(self, config):
        self.config = config
        self.prepare_dataloaders(config['data'])
        self.model = MLP(config['MLP'])
        print(self.model)

        self.model_name = config['train']['model_name']

        self.checkpoint_dir = './checkpoint_dir/{}/'.format(self.model_name)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.tb_log_dir = './tb_log/{}/'.format(self.model_name)
        if not os.path.exists(self.tb_log_dir):
            os.mkdir(self.tb_log_dir)

        self.optimal_metric = 0
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
        if self.cur_metric > self.optimal_metric:
            self.optimal_metric = self.cur_metric
            return True
        return False


    def saveWeights(self, clean_previous=True):
        if clean_previous:
            files = glob(self.checkpoint_dir + '*.pth')
            for f in files:
                os.remove(f)
        torch.save(self.model.state_dict(), '{}model_{}.pth'.format(self.checkpoint_dir, self.epoch))






