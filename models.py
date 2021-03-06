import os
import torch
import torch.nn as nn
import numpy as np
import random
from utils import read_json

class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        in_size, fil_num, drop_rate, out_size = config['in_size'], config['fil_num'], config['drop_rate'], config['out_size']
        self.fil_num = fil_num
        self.out_size = out_size
        self.in_size = in_size
        self.dense1 = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(in_size, fil_num),
            nn.BatchNorm1d(fil_num),
        )
        self.dense2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(fil_num, out_size),
        )

    def forward(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return x


class MLP_exo(nn.Module):
    def __init__(self, config):
        super(MLP_exo, self).__init__()
        in_size, fil_num, drop_rate, out_size = config['in_size'], config['fil_num'], config['drop_rate'], config['out_size']
        self.fil_num = fil_num
        self.out_size = out_size
        self.in_size = in_size
        self.dense1 = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(in_size*3, fil_num),
            nn.BatchNorm1d(fil_num),
        )
        self.dense2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(fil_num, out_size),
        )

    def forward(self, x_i, y_i, z_i):
        x = torch.cat((x_i, y_i, z_i), dim=1)
        x = self.dense1(x)
        x = self.dense2(x)
        return x


class MLP_3D(nn.Module):
    def __init__(self, config):
        super(MLP_3D, self).__init__()
        in_size, fil_num, drop_rate, out_size = config['in_size'], config['fil_num'], config['drop_rate'], 3
        self.fil_num = fil_num
        self.out_size = out_size
        self.in_size = in_size
        self.dense1 = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(in_size*3, fil_num),
            nn.BatchNorm1d(fil_num),
        )
        self.dense2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(fil_num, out_size),
        )

    def forward(self, x_i, y_i, z_i):
        x = torch.cat((x_i, y_i, z_i), dim=1)
        x = self.dense1(x)
        x = self.dense2(x)
        return x


class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        h_size = config['hidden_size']
        layer = config['stack_layer']
        self.lstm = nn.LSTM(3, h_size, layer, batch_first=True)
        self.dense = nn.Linear(h_size, 3)

    def forward(self, x_i, y_i, z_i):
        input = torch.stack((x_i, y_i, z_i), dim=2)
        x, _ = self.lstm(input)
        x = x[:, -1, :]
        x = self.dense(x)
        return x


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        d_model = config['d_model']
        nhead = config['nhead']
        nlayer = config['nlayer']
        w_size = config['w_size'] - 1
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=40)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayer)
        self.emb = nn.Linear(3, d_model)
        self.reg = nn.Linear(w_size * d_model, 3)

    def forward(self, x_i, y_i, z_i):
        input = torch.stack((x_i, y_i, z_i), dim=2)
        input = self.emb(input)
        out = self.transformer_encoder(input)
        out = out.view(out.shape[0], -1)
        out = self.reg(out)
        return out


if __name__ == "__main__":
    # config = read_json('config.json')["MLP"]
    # model = MLP(config)
    # print(model)

    # config = read_json('config.json')["LSTM"]
    # model = LSTM(config)
    # X_i = torch.rand((16, 39))
    # Y_i = torch.rand((16, 39))
    # Z_i = torch.rand((16, 39))
    # out = model(X_i, Y_i, Z_i)
    # print(len(out), out.shape)

    # config = read_json('config.json')["MLP"]
    # model = MLP_exo(config)
    # X_i = torch.rand((16, 59))
    # Y_i = torch.rand((16, 59))
    # Z_i = torch.rand((16, 59))
    # out = model(X_i, Y_i, Z_i)
    # print(out.shape)

    config = read_json('config.json')["Trans"]
    model = Transformer(config)
    print(model)
    X_i = torch.rand((16, 59))
    Y_i = torch.rand((16, 59))
    Z_i = torch.rand((16, 59))
    out = model(X_i, Y_i, Z_i)
    print(out.shape)
