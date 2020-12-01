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


if __name__ == "__main__":
    config = read_json('config.json')["MLP"]
    model = MLP(config)
    print(model)

