import numpy as np
import torch
import pandas as pd
import random
import itertools
import math

class rand_poly():
    def __init__(self, config):
        self.coeffs = config.get("generation_function", {}).get("coefficients", ())
        # need to add an option to randomly generate these.
        self.totaldeg = len(self.coeffs) - 1
        if self.totaldeg == -1:
            if config.get("generation_function", {}).get("degree", None) is not None:
                self.input_dim = config.get("dataset_params", {}).get("input_dim", None)
                self.totaldeg = config.get("generation_function", {}).get("degree", None)
                self.mu = config.get("generation_function", {}).get("coeff_mean", 0)
                sigma = config.get("generation_function", {}).get("coeff_std", 1)
                # d by dim x + 1 (constants) tensor
                # d indices tensor where we divide by (n-j)! 
                # where j is the number of indices that are the same. 
                # each tensor has random indices.
                # These are the coefficients.
                # Need to: divide tensor elements by a function of the index in a parallelisable manner.
                # Tensor size dim(x) x dim(x) x dim(x) x ... x dim(x) d times.
                self.sequences = itertools.combinations_with_replacement([r for r in range(self.input_dim)], r = self.totaldeg)
                self.num_terms = math.comb(self.input_dim + self.totaldeg -1, self.totaldeg)
                self.coeffs = np.random.normal(self.mu, sigma, size = self.num_terms)
        else:
            print("Placeholder for manually specified co-efficients.")
    def eval(self, x):
        px = 0
        i = 0
        for seq in self.sequences:
            px += self.coeffs[i]*np.prod(np.array([x[seq[i]] for i in range(len(seq))]))
            i += 1
        return px
    
