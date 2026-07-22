from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metrics: dict | None = None,
) -> None:
    """모델 종류를 포함해 checkpoint를 저장한다."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": getattr(model, "checkpoint_schema", model.__class__.__name__),
            "epoch": int(epoch),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "metrics": metrics or {},
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    """현재 모델 schema와 같은 checkpoint만 복원한다.

    예전 pose-conditioned Refiner checkpoint를 새 completion 모델에 잘못 넣으면
    state-dict key 오류 대신 명확한 비호환 메시지를 낸다.
    """

    payload = torch.load(path, map_location=map_location, weights_only=False)
    expected = getattr(model, "checkpoint_schema", model.__class__.__name__)
    actual = payload.get("schema")
    if actual != expected:
        raise RuntimeError(
            f"호환되지 않는 checkpoint schema입니다: expected={expected!r}, actual={actual!r}. "
            "Convex Quad completion 모델을 다시 학습하세요."
        )
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return payload
