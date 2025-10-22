import yaml
from torch import cuda, save, device, zeros
import numpy as np
from utils import (
    parse_command_line,
    set_seed,
)
from time import perf_counter
from typing import Dict
from pathlib import Path
from pandas import DataFrame, read_csv, concat
from getters import get_generation_function, get_criterion, get_model
from data_utils import get_synthetic_data
DEVICE = device("cuda" if cuda.is_available() else "cpu")

def train_all_folds(config):
    if config.get("seed", None) is not None:
        set_seed(config.get("seed", None))
    data_acquisition = config.get("dataset_params", {}).get("data_acquisition", "from_csv")
    # data_acquisition = generate or from_csv
    # if data_acquisition == generate, randomly generate some datapoints from a specified function.
    # if data_acquisition = from_csv, take inputs and targets from a csv.
        # Train data and validation data are from the same dataset
    if data_acquisition == "generate":
        generation_function = get_generation_function(config)
        data = get_synthetic_data(config)
        # Now also need to generate synthetic input_data
        # print(generation_function.eval(np.array([1,2,3,4,5,6,7,8,9,10])))
    elif data_acquisition == "from_csv":
        raise NotImplementedError
    else:
        print("Choose generate or from_csv!")
        raise NotImplementedError
    criterion = get_criterion(config)
    model = get_model(config) # gets sent to device in the getter function

if __name__ == "__main__":
    args = parse_command_line()
    with open(args.config, encoding="utf-8") as f:
        read_config = yaml.safe_load(f)  # Loads the config
    # train_all_folds(read_config, args.config)
    train_all_folds(read_config)
