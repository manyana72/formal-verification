from __future__ import annotations

import numpy as np
import torch

from nnverify.verifiers.milp.bounds_ibp import affine_bounds


def test_affine_bounds_contain_samples() -> None:
    torch.manual_seed(0)
    in_dim = 5
    out_dim = 3
    W = torch.randn(out_dim, in_dim)
    b = torch.randn(out_dim)

    l = torch.randn(in_dim) - 1.0
    u = l + torch.rand(in_dim) * 2.0

    a_l, a_u = affine_bounds(W, b, l, u)

    num_samples = 500
    hs = l + (u - l) * torch.rand(num_samples, in_dim)
    acts = (hs @ W.t()) + b

    assert torch.all(acts >= (a_l - 1e-5))
    assert torch.all(acts <= (a_u + 1e-5))

