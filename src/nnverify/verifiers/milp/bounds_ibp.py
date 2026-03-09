from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor, nn


def affine_bounds(W: Tensor, b: Tensor, h_l: Tensor, h_u: Tensor) -> tuple[Tensor, Tensor]:
    W_plus = torch.clamp(W, min=0.0)
    W_minus = torch.clamp(W, max=0.0)
    a_l = W_plus @ h_l + W_minus @ h_u + b
    a_u = W_plus @ h_u + W_minus @ h_l + b
    return a_l, a_u


def relu_bounds(a_l: Tensor, a_u: Tensor) -> tuple[Tensor, Tensor]:
    return torch.relu(a_l), torch.relu(a_u)


def compute_mlp_ibp_bounds(
    model: nn.Module,
    x_center: Tensor,
    eps: float,
) -> dict:
    x0 = x_center.detach().view(-1)
    x_l = torch.clamp(x0 - float(eps), 0.0, 1.0)
    x_u = torch.clamp(x0 + float(eps), 0.0, 1.0)

    h_l_list: List[Tensor] = [x_l]
    h_u_list: List[Tensor] = [x_u]
    a_l_list: List[Tensor] = []
    a_u_list: List[Tensor] = []

    if not hasattr(model, "linear_layers"):
        raise ValueError("Model must expose linear_layers() for MILP/IBP.")
    linears: List[nn.Linear] = list(model.linear_layers())

    for li, layer in enumerate(linears):
        W = layer.weight.detach()
        b = layer.bias.detach()
        h_l = h_l_list[-1]
        h_u = h_u_list[-1]
        a_l, a_u = affine_bounds(W, b, h_l, h_u)
        a_l_list.append(a_l)
        a_u_list.append(a_u)
        is_last = li == len(linears) - 1
        if not is_last:
            h_l, h_u = relu_bounds(a_l, a_u)
            h_l_list.append(h_l)
            h_u_list.append(h_u)

    return {
        "x_l": x_l,
        "x_u": x_u,
        "a_l_list": a_l_list,
        "a_u_list": a_u_list,
        "h_l_list": h_l_list,
        "h_u_list": h_u_list,
    }

