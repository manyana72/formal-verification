from __future__ import annotations

import torch
from torch import nn


class MnistMlp(nn.Module):
    def __init__(self, in_dim: int = 784, h1: int = 128, h2: int = 64, num_classes: int = 10):
        super().__init__()
        self.in_dim = int(in_dim)
        self.h1 = int(h1)
        self.h2 = int(h2)
        self.num_classes = int(num_classes)

        self.fc1 = nn.Linear(self.in_dim, self.h1)
        self.fc2 = nn.Linear(self.h1, self.h2)
        self.fc3 = nn.Linear(self.h2, self.num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def linear_layers(self) -> list[nn.Linear]:
        return [self.fc1, self.fc2, self.fc3]

