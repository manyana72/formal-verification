from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import torch
from torch import nn

from nnverify.verifiers.milp.backend import BaseBackend, get_backend
from nnverify.verifiers.milp.encode_mlp_relu import encode_mlp_on_box
from nnverify.verifiers.milp.types import (
    LinExpr,
    VerificationResult,
    VerificationStatus,
    expr_add,
    expr_const,
    expr_mul,
    expr_sub,
    expr_var,
)


def _margin_expr(logit_vars: List[Any], y: int, c: int) -> LinExpr:
    return expr_sub(expr_var(logit_vars[c]), expr_var(logit_vars[y]))


def _status_from_backend(status_str: str) -> VerificationStatus:
    if status_str in {"OPTIMAL", "FEASIBLE"}:
        return VerificationStatus.FALSIFIED
    if status_str in {"INFEASIBLE", "UNBOUNDED"}:
        return VerificationStatus.VERIFIED
    if status_str in {"TIME_LIMIT"}:
        return VerificationStatus.TIMEOUT
    return VerificationStatus.ERROR


def verify_point(
    model: nn.Module,
    x: torch.Tensor,
    y: int,
    eps: float,
    backend_name: str = "auto",
    time_limit_s: float | None = None,
) -> VerificationResult:
    device = next(model.parameters()).device
    x = x.detach().to(device)

    num_classes = 10
    classes = [c for c in range(num_classes) if c != int(y)]

    worst_margin = float("-inf")
    best_adv: List[float] | None = None
    solve_time_accum = 0.0
    last_solver_name = ""

    for c in classes:
        backend: BaseBackend = get_backend(backend_name)
        encoded = encode_mlp_on_box(backend, model=model, x_center=x, eps=eps)
        input_vars = encoded["input_vars"]
        logit_vars = encoded["logit_vars"]

        margin = _margin_expr(logit_vars, y=int(y), c=int(c))
        backend.set_objective_max(margin)

        status_str, obj_val, t_s = backend.solve(time_limit_s=time_limit_s)
        last_solver_name = backend.solver_name
        solve_time_accum += float(t_s)
        status = _status_from_backend(status_str)

        if status in {VerificationStatus.ERROR, VerificationStatus.TIMEOUT}:
            return VerificationResult(
                status=status,
                worst_margin=None,
                solve_time=solve_time_accum,
                solver_name=last_solver_name,
                adv_example=None,
            )

        if obj_val is None:
            continue

        if obj_val > worst_margin:
            worst_margin = float(obj_val)
            if status == VerificationStatus.FALSIFIED:
                adv = [backend.get_value(v) for v in input_vars]
                best_adv = adv

    if worst_margin <= 0.0:
        return VerificationResult(
            status=VerificationStatus.VERIFIED,
            worst_margin=worst_margin,
            solve_time=solve_time_accum,
            solver_name=last_solver_name or backend_name,
            adv_example=None,
        )

    return VerificationResult(
        status=VerificationStatus.FALSIFIED,
        worst_margin=worst_margin,
        solve_time=solve_time_accum,
        solver_name=last_solver_name or backend_name,
        adv_example=best_adv,
    )

