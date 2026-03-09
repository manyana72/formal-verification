from __future__ import annotations

import torch
from torch import nn

from nnverify.attacks.pgd import pgd_linf


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def test_pgd_increases_loss_or_flips() -> None:
    torch.manual_seed(0)
    model = ToyModel()
    x0 = torch.tensor([[0.1, -0.2]])
    y = torch.tensor([1])

    model.eval()
    with torch.no_grad():
        logits = model(x0)
        loss_before = nn.functional.cross_entropy(logits, y)
        pred_before = logits.argmax(dim=1)

    x_adv = pgd_linf(model, x0=x0, y=y, eps=0.3, steps=20, step_size=0.05, random_start=True)

    model.eval()
    with torch.no_grad():
        logits_adv = model(x_adv)
        loss_after = nn.functional.cross_entropy(logits_adv, y)
        pred_after = logits_adv.argmax(dim=1)

    assert (loss_after > loss_before) or bool((pred_after != pred_before).any())

