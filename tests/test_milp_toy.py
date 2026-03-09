from __future__ import annotations

import itertools

import numpy as np
import torch
from torch import nn

from nnverify.verifiers.milp.backend import get_backend
from nnverify.verifiers.milp.encode_mlp_relu import encode_mlp_on_box
from nnverify.verifiers.milp.verify import _margin_expr


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 2)
        with torch.no_grad():
            self.fc1.weight.copy_(torch.tensor([[1.0, -1.0], [-0.5, 0.5]]))
            self.fc1.bias.zero_()
            self.fc2.weight.copy_(torch.tensor([[1.0, 0.5], [-0.5, 1.0]]))
            self.fc2.bias.zero_()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.fc1(x))
        return self.fc2(h)

    def linear_layers(self):
        return [self.fc1, self.fc2]


def brute_force_margin(model: nn.Module, x0: torch.Tensor, eps: float, y: int, c: int) -> float:
    xs = []
    grid = np.linspace(-eps, eps, 21)
    for dx0, dx1 in itertools.product(grid, grid):
        xs.append(x0 + torch.tensor([dx0, dx1]))
    X = torch.stack(xs)
    with torch.no_grad():
        logits = model(X)
        margins = logits[:, c] - logits[:, y]
    return float(margins.max().item())


def test_milp_matches_bruteforce_margin() -> None:
    torch.manual_seed(0)
    model = TinyNet()
    x0 = torch.tensor([0.2, -0.1])
    eps = 0.1
    y = 0
    c = 1

    backend = get_backend("cbc")
    encoded = encode_mlp_on_box(backend, model=model, x_center=x0, eps=eps)
    logit_vars = encoded["logit_vars"]
    margin_expr = _margin_expr(logit_vars, y=y, c=c)
    backend.set_objective_max(margin_expr)
    status, obj_val, _ = backend.solve(time_limit_s=5.0)

    assert status in {"OPTIMAL", "FEASIBLE"}
    assert obj_val is not None

    brute = brute_force_margin(model, x0, eps, y=y, c=c)
    assert obj_val >= brute - 1e-4
    assert abs(obj_val - brute) <= 1e-2

