import torch
from torch import nn

class ohlFCNN(nn.module):
    def __init__(self, input_dim, degree, width):
        self.s1 = nn.linear(input_dim, width)
        self.degree = degree
    def forward(self, x):
        x = self.s1(x)
        x = torch.pow(x, self.degree)
        return torch.sum(x)
