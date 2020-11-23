from models import *
from dataloader import Dataset
from utils import read_json
from model_wrapper import ModelDev


if __name__ == "__main__":
    config = read_json('config.json')
    print(config)
    model = ModelDev(config)
    model.train()