from __future__ import annotations

import importlib
import importlib.machinery
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from wide_fov_supervision_v2.config import BackboneConfig, FisheyeCameraConfig, PathConfig


class DSINEWrapper:
    """DSINE normal estimator local wrapper.

    반환 normal은 OpenCV camera frame 기준 `(H,W,3)`이다. direct fisheye branch에서는
    DSINE에 pinhole 근사 intrinsics를 전달하므로 baseline 성격이 강하다.
    """

    def __init__(self, paths: PathConfig, config: BackboneConfig, device: str | None = None) -> None:
        self.paths = paths
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._model: torch.nn.Module | None = None

    def load(self) -> torch.nn.Module:
        if self._model is not None:
            return self._model
        if not self.paths.dsine_root.exists():
            raise FileNotFoundError(f"DSINE root not found: {self.paths.dsine_root}")
        if not self.paths.dsine_ckpt.exists():
            raise FileNotFoundError(f"DSINE checkpoint not found: {self.paths.dsine_ckpt}")
        self._prepare_import_path(self.paths.dsine_root)
        from models.dsine.v02 import DSINE_v02

        model = DSINE_v02(_dsine_v02_args())
        state = torch.load(str(self.paths.dsine_ckpt), map_location="cpu")["model"]
        state = {key.replace("module.", ""): value for key, value in state.items()}
        model.load_state_dict(state, strict=True)
        self._model = model.to(self.device).eval()
        return self._model

    @torch.inference_mode()
    def predict(self, rgb: np.ndarray, camera: FisheyeCameraConfig | None = None, intrinsics: np.ndarray | None = None) -> np.ndarray:
        """RGB image에서 camera-frame normal map을 추정한다."""

        model = self.load()
        image = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
        _, _, orig_h, orig_w = image.shape
        left, right, top, bottom = _padding_to_multiple_of_32(orig_h, orig_w)
        image = F.pad(image, (left, right, top, bottom), mode="constant", value=0.0)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        image = (image - mean) / std

        if intrinsics is None:
            if camera is None:
                fx = fy = 0.5 * orig_w / np.tan(np.deg2rad(self.config.dsine_pinhole_fov_degrees) * 0.5)
                cx = (orig_w - 1) * 0.5
                cy = (orig_h - 1) * 0.5
            else:
                fx, fy, cx, cy = camera.fx, camera.fy, camera.cx, camera.cy
            intrinsics = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        intrins = torch.from_numpy(intrinsics.astype(np.float32)).unsqueeze(0).to(self.device)
        intrins[:, 0, 2] += left
        intrins[:, 1, 2] += top
        pred = model(image, intrins=intrins, mode="test")[-1]
        pred = pred[:, :, top : top + orig_h, left : left + orig_w][0]
        normal = pred.permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(normal, axis=-1, keepdims=True)
        normal = normal / np.clip(norm, 1.0e-6, None)
        return normal.astype(np.float32)

    @staticmethod
    def _prepare_import_path(root: Path) -> None:
        root_str = str(root.resolve())
        if root_str in sys.path:
            sys.path.remove(root_str)
        sys.path.insert(0, root_str)
        for module_name in list(sys.modules):
            if module_name == "models" or module_name.startswith("models.") or module_name == "utils" or module_name.startswith("utils."):
                del sys.modules[module_name]
        for package_name in ("models", "utils"):
            package_dir = root / package_name
            package = ModuleType(package_name)
            package.__file__ = str(package_dir / "__init__.py")
            package.__path__ = [str(package_dir)]
            package.__package__ = package_name
            package.__spec__ = importlib.machinery.ModuleSpec(package_name, loader=None, is_package=True)
            package.__spec__.submodule_search_locations = [str(package_dir)]
            sys.modules[package_name] = package
        importlib.invalidate_caches()


def _dsine_v02_args() -> SimpleNamespace:
    return SimpleNamespace(
        NNET_encoder_B=5,
        NNET_decoder_NF=2048,
        NNET_decoder_BN=False,
        NNET_decoder_down=8,
        NNET_learned_upsampling=True,
        NNET_output_dim=3,
        NNET_feature_dim=64,
        NNET_hidden_dim=64,
        NRN_prop_ps=5,
        NRN_num_iter_train=5,
        NRN_num_iter_test=5,
        NRN_ray_relu=True,
    )


def _padding_to_multiple_of_32(height: int, width: int) -> tuple[int, int, int, int]:
    new_width = width if width % 32 == 0 else 32 * ((width // 32) + 1)
    new_height = height if height % 32 == 0 else 32 * ((height // 32) + 1)
    left = (new_width - width) // 2
    right = (new_width - width) - left
    top = (new_height - height) // 2
    bottom = (new_height - height) - top
    return left, right, top, bottom
