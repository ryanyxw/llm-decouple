"""
Binary classifier implemented using Pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryClassifier(torch.nn.Module):
    def __init__(self, input_dim: int):
        super(BinaryClassifier, self).__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(self.input_dim, 2)

    def forward(self, hidden_state):
        return self.linear(hidden_state)