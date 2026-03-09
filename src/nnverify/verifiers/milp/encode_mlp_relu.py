from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch import nn

from nnverify.verifiers.milp.backend import BaseBackend
from nnverify.verifiers.milp.bounds_ibp import compute_mlp_ibp_bounds
from nnverify.verifiers.milp.types import LinExpr, expr_add, expr_const, expr_mul, expr_var


def _affine_layer(
    backend: BaseBackend,
    W: torch.Tensor,
    b: torch.Tensor,
    prev_h_vars: List[Any],
) -> List[Any]:
    out_dim, in_dim = W.shape
    a_vars: List[Any] = [backend.add_var(lb=-1e9, ub=1e9, binary=False) for _ in range(out_dim)]
    for i in range(out_dim):
        expr_rhs: LinExpr = expr_const(float(b[i].item()))
        for j in range(in_dim):
            coef = float(W[i, j].item())
            if coef == 0.0:
                continue
            expr_rhs = expr_add(expr_rhs, expr_mul(expr_var(prev_h_vars[j]), coef))
        backend.add_eq(expr_var(a_vars[i]), expr_rhs)
    return a_vars


def _relu_layer(
    backend: BaseBackend,
    a_vars: List[Any],
    a_l: torch.Tensor,
    a_u: torch.Tensor,
) -> List[Any]:
    h_vars: List[Any] = []
    for i, a_var in enumerate(a_vars):
        l = float(a_l[i].item())
        u = float(a_u[i].item())
        if u <= 0.0:
            h = backend.add_var(lb=0.0, ub=0.0, binary=False)
            h_vars.append(h)
            continue
        if l >= 0.0:
            h = backend.add_var(lb=l, ub=u, binary=False)
            backend.add_eq(expr_var(h), expr_var(a_var))
            h_vars.append(h)
            continue

        h = backend.add_var(lb=0.0, ub=u, binary=False)
        b_bin = backend.add_binary()

        backend.add_ge(expr_var(h), expr_const(0.0))
        backend.add_ge(expr_var(h), expr_var(a_var))

        one_minus_b = expr_add(expr_const(1.0), expr_mul(expr_var(b_bin), -1.0))
        rhs1 = expr_add(expr_var(a_var), expr_mul(one_minus_b, -l))
        backend.add_le(expr_var(h), rhs1)

        rhs2 = expr_mul(expr_var(b_bin), u)
        backend.add_le(expr_var(h), rhs2)

        backend.add_ge(expr_var(a_var), expr_const(l))
        backend.add_le(expr_var(a_var), expr_const(u))
        h_vars.append(h)
    return h_vars


def encode_mlp_on_box(
    backend: BaseBackend,
    model: nn.Module,
    x_center: torch.Tensor,
    eps: float,
) -> Dict[str, Any]:
    bounds = compute_mlp_ibp_bounds(model, x_center=x_center, eps=eps)
    x_l = bounds["x_l"]
    x_u = bounds["x_u"]
    a_l_list = bounds["a_l_list"]
    a_u_list = bounds["a_u_list"]

    x0_flat = x_center.detach().view(-1)
    dim = x0_flat.numel()
    in_vars: List[Any] = [
        backend.add_var(lb=float(x_l[i].item()), ub=float(x_u[i].item()), binary=False)
        for i in range(dim)
    ]

    if not hasattr(model, "linear_layers"):
        raise ValueError("Model must expose linear_layers() for MILP encoding.")
    linears: List[nn.Linear] = list(model.linear_layers())

    prev_h_vars: List[Any] = in_vars
    logit_vars: List[Any] = []

    for li, layer in enumerate(linears):
        W = layer.weight.detach()
        b = layer.bias.detach()
        a_vars = _affine_layer(backend, W=W, b=b, prev_h_vars=prev_h_vars)
        is_last = li == len(linears) - 1
        if is_last:
            logit_vars = a_vars
        else:
            a_l = a_l_list[li]
            a_u = a_u_list[li]
            prev_h_vars = _relu_layer(backend, a_vars=a_vars, a_l=a_l, a_u=a_u)

    return {
        "input_vars": in_vars,
        "logit_vars": logit_vars,
        "x_l": x_l,
        "x_u": x_u,
    }

