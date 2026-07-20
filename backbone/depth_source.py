from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib

import numpy as np


@dataclass(frozen=True)
class ExternalDepthPrediction:
    """외부 NPY에서 로드한 source-camera z-depth.

    Attributes:
        depth_z: ``(H,W) float32``. 단위는 metre이며 invalid 픽셀은 NaN이다.
        valid: ``(H,W) bool``. finite이고 0보다 큰 depth인 픽셀만 True이다.
        metadata: 재현성을 위한 절대 경로와 SHA-256 해시.

    이 로더는 radial depth를 z-depth로 변환하지 않는다. 입력 NPY는
    반드시 현재 RGB와 pixel-aligned된 source-camera z-depth여야 한다.
    """

    depth_z: np.ndarray
    valid: np.ndarray
    metadata: dict[str, str | int | list[int]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_external_z_depth(path: Path, expected_hw: tuple[int, int]) -> ExternalDepthPrediction:
    """외부 ``.npy`` z-depth를 검증하고 표준 depth prediction으로 변환한다.

    해상도가 다른 depth를 resize하면 카메라 ray와 픽셀 대응이 깨지므로
    자동 resize하지 않고 명시적으로 실패시킨다.
    """

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"External z-depth NPY not found: {path}")
    raw = np.load(path, allow_pickle=False)
    if not isinstance(raw, np.ndarray) or raw.ndim != 2:
        raise ValueError(f"External z-depth must be a 2D NPY array, got {getattr(raw, 'shape', None)}")
    if tuple(raw.shape) != tuple(expected_hw):
        raise ValueError(
            f"External z-depth shape {tuple(raw.shape)} does not match RGB/camera shape {tuple(expected_hw)}. "
            "Automatic resize is disabled because it breaks ray alignment."
        )

    depth_z = np.asarray(raw, dtype=np.float32).copy()
    valid = np.isfinite(depth_z) & (depth_z > 0.0)
    depth_z[~valid] = np.nan
    return ExternalDepthPrediction(
        depth_z=depth_z,
        valid=valid,
        metadata={
            "depth_source": "external_npy",
            "external_depth_path": str(path),
            "external_depth_sha256": _sha256(path),
            "external_depth_shape": [int(raw.shape[0]), int(raw.shape[1])],
            "external_depth_valid_pixels": int(valid.sum()),
            "external_depth_semantics": "source-camera z-depth in metres; invalid values are non-finite or <= 0",
        },
    )
