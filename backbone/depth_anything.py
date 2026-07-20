from __future__ import annotations

import importlib
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

from wide_fov_supervision_v2.config import BackboneConfig, PathConfig


class DepthAnythingMetricWrapper:
    """Depth Anything V2 metric-depth local wrapper.

    입력은 RGB `uint8 (H,W,3)`이고, 출력은 `float32 (H,W)` meter 단위 source
    z-depth로 취급한다. fisheye direct branch에서는 pretrained pinhole 가정을 벗어난
    baseline이므로 결과 해석에 그 한계를 기록한다.
    """

    MODEL_CONFIGS = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }

    def __init__(self, paths: PathConfig, config: BackboneConfig, device: str | None = None) -> None:
        self.paths = paths
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._model: torch.nn.Module | None = None

    def _checkpoint(self) -> Path:
        if self.config.da_encoder == "vitl":
            return self.paths.depth_anything_vitl_ckpt
        if self.config.da_encoder == "vits":
            return self.paths.depth_anything_vits_ckpt
        raise ValueError(f"No local checkpoint configured for encoder={self.config.da_encoder}")

    def load(self) -> torch.nn.Module:
        if self._model is not None:
            return self._model
        root = self.paths.depth_anything_root
        ckpt = self._checkpoint()
        if not root.exists():
            raise FileNotFoundError(f"Depth Anything V2 metric_depth root not found: {root}")
        if not ckpt.exists():
            raise FileNotFoundError(f"Depth Anything V2 checkpoint not found: {ckpt}")

        root_str = str(root.resolve())
        if root_str in sys.path:
            sys.path.remove(root_str)
        sys.path.insert(0, root_str)
        importlib.invalidate_caches()
        from depth_anything_v2.dpt import DepthAnythingV2

        model = DepthAnythingV2(
            **{
                **self.MODEL_CONFIGS[self.config.da_encoder],
                "max_depth": float(self.config.da_max_depth_m),
            }
        )
        state = torch.load(str(ckpt), map_location="cpu")
        model.load_state_dict(state, strict=True)
        self._model = model.to(self.device).eval()
        return self._model

    @torch.inference_mode()
    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """RGB image에서 z-depth map을 추정한다."""

        model = self.load()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        depth = model.infer_image(bgr, int(self.config.da_input_size))
        return np.asarray(depth, dtype=np.float32)
