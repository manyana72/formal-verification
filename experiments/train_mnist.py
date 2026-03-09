from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch import nn, optim
from tqdm import tqdm

from nnverify.data.mnist import get_mnist_datasets, make_loader
from nnverify.models.io import CheckpointMeta, build_model, save_checkpoint
from nnverify.utils.seed import set_seed


def _load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


@torch.no_grad()
def _eval_accuracy(model: nn.Module, loader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x).argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return correct / max(total, 1)


def train_from_config(cfg_path: str | Path) -> Path:
    cfg = _load_yaml(cfg_path)
    seed = int(cfg.get("seed", 1234))
    set_seed(seed)

    run_name = str(cfg.get("run_name", "run"))
    data_dir = str(cfg.get("data_dir", "data"))
    batch_size = int(cfg.get("batch_size", 128))
    epochs = int(cfg.get("epochs", 3))
    lr = float(cfg.get("lr", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    num_workers = int(cfg.get("num_workers", 0))
    device = str(cfg.get("device", "cpu"))

    model_cfg = dict(cfg.get("model", {}))
    model_type = str(model_cfg.get("type", "mlp"))
    model_kwargs: dict[str, Any] = {}
    if model_type == "mlp":
        model_kwargs = {"in_dim": 784, "h1": 128, "h2": 64, "num_classes": 10}
    elif model_type == "cnn_small":
        model_kwargs = {"num_classes": 10}
    else:
        raise ValueError(f"Unknown model type {model_type!r}")

    train_ds, test_ds = get_mnist_datasets(data_dir)
    train_loader = make_loader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, seed=seed
    )
    test_loader = make_loader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, seed=seed
    )

    model = build_model(model_type, model_kwargs).to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    metrics: dict[str, Any] = {"config": cfg, "epochs": []}
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        start = time.time()
        for x, y in tqdm(train_loader, desc=f"train epoch {epoch}/{epochs}", leave=False):
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()

            epoch_loss += float(loss.item()) * int(y.numel())
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())

        train_loss = epoch_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        test_acc = _eval_accuracy(model, test_loader, device=device)
        metrics["epochs"].append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "epoch_time_s": time.time() - start,
            }
        )
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f}"
        )

    metrics["total_time_s"] = time.time() - t0

    out_dir = Path("runs") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "model.pt"
    meta = CheckpointMeta(model_type=model_type, model_kwargs=model_kwargs, run_name=run_name)
    save_checkpoint(ckpt_path, model, meta=meta, metrics=metrics)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return ckpt_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config (e.g., configs/mlp.yaml)")
    args = ap.parse_args()
    ckpt = train_from_config(args.config)
    print(f"saved: {ckpt}")


if __name__ == "__main__":
    main()

