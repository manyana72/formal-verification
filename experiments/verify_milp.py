from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from nnverify.data.mnist import get_mnist_datasets
from nnverify.data.splits import load_split
from nnverify.models.io import load_checkpoint
from nnverify.verifiers.milp.types import VerificationStatus
from nnverify.verifiers.milp.verify import verify_point


def run(
    ckpt: str | Path,
    eps: float,
    subset: str | Path,
    solver: str = "auto",
    time_limit: float = 30.0,
    max_samples: int = 5,
    batch_size: int = 1,
    device: str = "cpu",
    data_dir: str | Path = "data",
) -> Path:
    model, meta, _ = load_checkpoint(ckpt, map_location=device)
    model.to(device)
    model.eval()

    _, test_ds = get_mnist_datasets(data_dir)
    split = load_split(subset)
    indices = split.indices[: max_samples]
    sub_ds = Subset(test_ds, indices)
    loader = DataLoader(sub_ds, batch_size=int(batch_size), shuffle=False, num_workers=0)

    rows = []
    adv_dir = Path("results") / "adversarial"
    adv_dir.mkdir(parents=True, exist_ok=True)

    for local_idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        idx = int(indices[local_idx])
        label = int(y[0].item())
        res = verify_point(
            model=model,
            x=x[0],
            y=label,
            eps=eps,
            backend_name=solver,
            time_limit_s=time_limit,
        )

        adv_path_str: str | None = None
        if res.status == VerificationStatus.FALSIFIED and res.adv_example is not None:
            adv = np.array(res.adv_example, dtype=np.float32).reshape(1, 28, 28)
            adv_path = adv_dir / f"mnist_idx{idx}_eps{eps:.4f}.npz"
            np.savez_compressed(
                adv_path,
                x_adv=adv,
                x0=x.cpu().numpy(),
                eps=float(eps),
                y=int(label),
            )
            adv_path_str = str(adv_path)

        rows.append(
            {
                "index": idx,
                "y": label,
                "status": res.status.value,
                "worst_margin": res.worst_margin,
                "time_s": res.solve_time,
                "solver": res.solver_name,
                "adv_path": adv_path_str,
            }
        )

    df = pd.DataFrame(rows)
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"milp_{meta.run_name}_eps{eps:.4f}.csv"
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(df)} rows)")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eps", type=float, required=True)
    ap.add_argument("--solver", default="auto")
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--max-samples", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()

    run(
        ckpt=args.ckpt,
        eps=args.eps,
        subset=args.subset,
        solver=args.solver,
        time_limit=args.time_limit,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        device=args.device,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()

