from __future__ import annotations

import torch

from wide_fov_supervision_v2.config import RefinerConfig
from wide_fov_supervision_v2.modules.refiner import RayAwareQueryRefiner
from wide_fov_supervision_v2.train.checkpoints import load_checkpoint, save_checkpoint


def test_checkpoint_save_load_identity(tmp_path) -> None:
    model = RayAwareQueryRefiner(RefinerConfig(base_channels=16, hidden_dim=32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, optimizer, epoch=3, metrics={"x": 1.0})
    restored = RayAwareQueryRefiner(RefinerConfig(base_channels=16, hidden_dim=32))
    payload = load_checkpoint(path, restored)
    assert payload["epoch"] == 3
    for a, b in zip(model.parameters(), restored.parameters()):
        assert torch.allclose(a, b)
