from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from nnverify.attacks.pgd import pgd_linf
from nnverify.data.mnist import get_mnist_datasets
from nnverify.data.splits import load_split
from nnverify.models.io import load_checkpoint


@torch.no_grad()
def _batch_metrics(model, x, y):
    logits = model(x)
    loss = F.cross_entropy(logits, y, reduction="none")
    pred = logits.argmax(dim=1)
    return pred, loss


def run(
    ckpt: str | Path,
    eps: float,
    subset: str | Path,
    steps: int = 40,
    step_size: float = 0.01,
    random_start: bool = True,
    batch_size: int = 64,
    device: str = "cpu",
    data_dir: str | Path = "data",
) -> Path:
    model, meta, _ = load_checkpoint(ckpt, map_location=device)
    model.to(device)
    model.eval()

    _, test_ds = get_mnist_datasets(data_dir)
    split = load_split(subset)
    sub_ds = Subset(test_ds, split.indices)
    loader = DataLoader(sub_ds, batch_size=int(batch_size), shuffle=False, num_workers=0)

    rows = []
    start = time.time()
    for batch_idx, (x0, y) in enumerate(loader):
        x0 = x0.to(device)
        y = y.to(device)

        clean_pred, clean_loss = _batch_metrics(model, x0, y)
        x_adv = pgd_linf(
            model,
            x0=x0,
            y=y,
            eps=float(eps),
            steps=int(steps),
            step_size=float(step_size),
            random_start=bool(random_start),
        )
        adv_pred, adv_loss = _batch_metrics(model, x_adv, y)

        base = batch_idx * int(batch_size)
        for i in range(y.shape[0]):
            idx = int(split.indices[base + i])
            rows.append(
                {
                    "index": idx,
                    "y": int(y[i].item()),
                    "clean_pred": int(clean_pred[i].item()),
                    "adv_pred": int(adv_pred[i].item()),
                    "success": int(adv_pred[i].item()) != int(y[i].item()),
                    "clean_loss": float(clean_loss[i].item()),
                    "adv_loss": float(adv_loss[i].item()),
                    "eps": float(eps),
                    "steps": int(steps),
                    "step_size": float(step_size),
                    "random_start": bool(random_start),
                    "run_name": meta.run_name,
                }
            )

    elapsed = time.time() - start
    df = pd.DataFrame(rows)
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pgd_{meta.run_name}_eps{eps:.4f}.csv"
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(df)} rows) in {elapsed:.2f}s")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eps", type=float, required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--step-size", type=float, default=0.01)
    ap.add_argument("--random-start", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()
    run(
        ckpt=args.ckpt,
        eps=args.eps,
        subset=args.subset,
        steps=args.steps,
        step_size=args.step_size,
        random_start=args.random_start,
        batch_size=args.batch_size,
        device=args.device,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()

