from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def _clamp_linf(x: torch.Tensor, x0: torch.Tensor, eps: float) -> torch.Tensor:
    lo = (x0 - float(eps)).clamp(0.0, 1.0)
    hi = (x0 + float(eps)).clamp(0.0, 1.0)
    return x.clamp(lo, hi)


def pgd_linf(
    model,
    x0: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    steps: int,
    step_size: float,
    random_start: bool = False,
) -> torch.Tensor:
    model.eval()
    x0 = x0.detach()
    x = x0.clone()
    if random_start:
        x = x + (2 * torch.rand_like(x) - 1) * float(eps)
        x = _clamp_linf(x, x0, eps)

    for _ in range(int(steps)):
        x.requires_grad_(True)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        grad = torch.autograd.grad(loss, x)[0]
        with torch.no_grad():
            x = x + float(step_size) * grad.sign()
            x = _clamp_linf(x, x0, eps)
        x = x.detach()
    return x

