from __future__ import annotations

import pytest
import torch

from wide_fov_supervision_v2.config import CompletionConfig
from wide_fov_supervision_v2.modules.quad_completion.model import QuadRayCompletionModel
from wide_fov_supervision_v2.train.checkpoints import load_checkpoint, save_checkpoint


def test_checkpoint_save_load_identity(tmp_path) -> None:
    model = QuadRayCompletionModel(CompletionConfig(hidden_dim=32, attention_heads=4, attention_blocks=1))
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, optimizer, epoch=3, metrics={"x": 1.0})
    restored = QuadRayCompletionModel(CompletionConfig(hidden_dim=32, attention_heads=4, attention_blocks=1))
    payload = load_checkpoint(path, restored)
    assert payload["epoch"] == 3
    for expected, actual in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(expected, actual)


def test_old_refiner_checkpoint_reports_schema_mismatch(tmp_path) -> None:
    path = tmp_path / "old.pt"
    torch.save({"model": {}, "schema": "pose_refiner_v1"}, path)
    model = QuadRayCompletionModel(CompletionConfig(hidden_dim=32, attention_heads=4, attention_blocks=1))
    with pytest.raises(RuntimeError, match="호환되지 않는 checkpoint schema"):
        load_checkpoint(path, model)
