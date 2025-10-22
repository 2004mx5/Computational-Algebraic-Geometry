import numpy as np
import torch
import pandas as pd
import random
import itertools
import math
from models.onehiddenlayerFCNN import ohlFCNN

def get_synthetic_data(config):
        input_data = np.random.normal(
            loc = config.get("generation_function", {}).get("input_mean", 0),
            scale = config.get("generation_function", {}).get("input_std", 1),
            size = (config.get("dataset_params", {}).get("input_dim", None), 
                    config.get("generation_function", {}).get("num_data_points", 1))
        )
        return input_data