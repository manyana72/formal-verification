from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from nnverify.models.cnn_small import CnnSmall
from nnverify.models.mlp import MnistMlp


@dataclass(frozen=True)
class CheckpointMeta:
    model_type: str
    model_kwargs: dict[str, Any]
    run_name: str


def build_model(model_type: str, model_kwargs: dict[str, Any]) -> nn.Module:
    if model_type == "mlp":
        return MnistMlp(**model_kwargs)
    if model_type == "cnn_small":
        return CnnSmall(**model_kwargs)
    raise ValueError(f"Unknown model_type={model_type!r}")


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    meta: CheckpointMeta,
    metrics: dict[str, Any] | None = None,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "meta": asdict(meta),
        "model_state_dict": model.state_dict(),
        "metrics": metrics or {},
    }
    torch.save(payload, str(p))
    meta_path = p.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(payload["meta"], indent=2) + "\n", encoding="utf-8")


def load_checkpoint(path: str | Path, map_location: str = "cpu"):
    p = Path(path)
    payload = torch.load(str(p), map_location=map_location)
    meta_raw = payload["meta"]
    meta = CheckpointMeta(
        model_type=str(meta_raw["model_type"]),
        model_kwargs=dict(meta_raw.get("model_kwargs", {})),
        run_name=str(meta_raw.get("run_name", "run")),
    )
    model = build_model(meta.model_type, meta.model_kwargs)
    model.load_state_dict(payload["model_state_dict"])
    return model, meta, payload.get("metrics", {})

