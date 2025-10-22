import numpy as np
import torch
import pandas as pd
import random
import itertools
import math
import data_generation_functions
from models.onehiddenlayerFCNN import ohlFCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_generation_function(config):
    if config.get("generation_function", {}).get("type", None) == "polynomial":
        return data_generation_functions.rand_poly(config)
    else:
        raise NotImplementedError
    
def get_model(config):
    if config.get("model", {}).get("type", "one_layer_polynomial") == "one_layer_polynomial":
        model = ohlFCNN(
            input_dim = config.get("dataset_params", {}).get("input_dim", None),
            degree = config.get("model", {}).get("degree", 2),
            width = config.get("model", {}).get("width", 1),
        )
    else:
        raise NotImplementedError
    model.to(DEVICE)
    return model

def get_criterion(config):
    if config.get("loss", {}).get("name", "MSE") == "MSE":
        criterion = torch.nn.MSELoss(reduction = config.get("loss", {}).get("reduction", "mean"))
    else:
        raise NotImplementedError
    return criterion

def get_synthetic_data(config):
        input_data = np.random.normal(
            loc = config.get("generation_function", {}).get("input_mean", 0),
            scale = config.get("generation_function", {}).get("input_std", 1),
            size = (config.get("dataset_params", {}).get("input_dim", None), 
                    config.get("generation_function", {}).get("num_data_points", 1))
        )
        return input_data

# print(get_generation_function({"input_dim": 8, "degree": 2}))