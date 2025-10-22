from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from torch import (  # pylint: disable=no-name-in-module  # type: ignore
    no_grad,
    zeros,
    cuda,
    logical_and,
)
from typing import Optional, Dict, Tuple, List
import torch
import random
import numpy as np

DEVICE = "cuda" if cuda.is_available() else "cpu"

def parse_command_line() -> Namespace:
    """Parse the command-line arguments.

    Returns
    -------
    Namespace
        The command-line arguments.

    """
    parser = ArgumentParser(
        description="Train polynomial neural network",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file for this pipeline.",
    )

    return parser.parse_args()

def set_seed(seed: int) -> None:
    """Set the random seed for torch, torch.cuda, random, np.random.

    Parameters
    ----------
    seed:   integer seed

    """
    print(f"setting seed to {seed}")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

