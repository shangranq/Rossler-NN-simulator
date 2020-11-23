from dataloader import Rossler, dataset
from torch.utils.data import DataLoader
from visual import show_window

class ModelDev:

    def __init__(self, config):
        self.config = config
        self.prepare_dataloaders(config)


    def prepare_dataloaders(self, config):
        data = Rossler(config)
        train_data = dataset(data.train_X, data.train_Y, data.train_Z, config['w_size'])
        self.train_dataloader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, drop_last=True)
        valid_data = dataset(data.valid_X, data.valid_Y, data.valid_Z, config['w_size'])
        self.valid_dataloader = DataLoader(valid_data, batch_size=config['batch_size'], shuffle=False, drop_last=True)
        test_data  = dataset(data.test_X, data.test_Y, data.test_Z, config['w_size'])
        self.test_dataloader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False, drop_last=True)


    def train(self):
        idx = 0
        for X_i, Y_i, Z_i, X_o, Y_o, Z_o in self.train_dataloader:
            print(X_i.shape, X_o.shape)
            show_window(X_i.data.numpy()[0, :], X_o.data.numpy()[0, :], idx, self.config['w_size'], self.config['stride'])
            idx += 1
            if idx == 3: break
