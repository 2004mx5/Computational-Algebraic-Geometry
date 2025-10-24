import torch
from torch import nn

class ohlFCNN(nn.Module):
    def __init__(self, input_dim, degree, width):
        super().__init__()
        self.s1 = nn.Linear(input_dim, width)
        self.degree = degree
    def forward(self, x):
        x = self.s1(x)
        x = torch.pow(x, self.degree)
        return torch.sum(x)
