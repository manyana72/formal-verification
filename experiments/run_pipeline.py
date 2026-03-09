from __future__ import annotations

import argparse
from pathlib import Path

from nnverify.data.splits import ensure_mnist_eval_split
from nnverify.models.io import load_checkpoint

from .attack_pgd import run as run_pgd
from .train_mnist import train_from_config
from .verify_milp import run as run_milp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Baseline YAML config (configs/baseline.yaml)")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    # Baseline config is small; parse lazily using yaml in train script if needed.
    import yaml

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    subset_path = Path(cfg.get("subset_path", "assets/splits/mnist_eval_100.json"))
    ensure_mnist_eval_split(subset_path)

    mlp_cfg = Path(cfg["mlp_config"])
    mlp_ckpt = Path("runs") / cfg.get("mlp_run_name", "mlp_mnist") / "model.pt"
    if not mlp_ckpt.exists():
        mlp_ckpt = train_from_config(mlp_cfg)

    cnn_cfg = Path(cfg["cnn_config"])
    cnn_ckpt = Path("runs") / cfg.get("cnn_run_name", "cnn_small_mnist") / "model.pt"
    if not cnn_ckpt.exists():
        cnn_ckpt = train_from_config(cnn_cfg)

    pgd_cfg = cfg.get("pgd", {})
    eps = float(pgd_cfg.get("eps", 0.03))
    steps = int(pgd_cfg.get("steps", 40))
    step_size = float(pgd_cfg.get("step_size", 0.01))
    random_start = bool(pgd_cfg.get("random_start", True))

    run_pgd(
        ckpt=mlp_ckpt,
        eps=eps,
        subset=subset_path,
        steps=steps,
        step_size=step_size,
        random_start=random_start,
    )

    milp_cfg = cfg.get("milp", {})
    milp_eps = float(milp_cfg.get("eps", 0.03))
    solver = str(milp_cfg.get("solver", "auto"))
    time_limit = float(milp_cfg.get("time_limit", 30.0))
    max_samples = int(milp_cfg.get("max_samples", 5))

    run_milp(
        ckpt=mlp_ckpt,
        eps=milp_eps,
        subset=subset_path,
        solver=solver,
        time_limit=time_limit,
        max_samples=max_samples,
    )


if __name__ == "__main__":
    main()

