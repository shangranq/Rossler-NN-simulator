from dataloader import Rossler, dataset
from torch.utils.data import DataLoader

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
        for X_i, Y_i, Z_i, X_o, Y_o, Z_o in self.train_dataloader:
            print(X_i.shape, X_o.shape)
