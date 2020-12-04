from models import *
from dataloader import Dataset
from utils import read_json
from model_wrapper import ModelDev, ModelDev_3D


if __name__ == "__main__":
    config = read_json('config.json')
    print(config)
    # model = ModelDev(config)
    model = ModelDev_3D(config)
    # model.train()
    print('test MSE error ', model.test_MSE())
    model.test_long_window(500)
    model.show_3D_trajectory()

