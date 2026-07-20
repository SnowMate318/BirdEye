from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb.astype(np.uint8)).save(path)


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8)).save(path)


def save_heatmap(
    path: Path,
    value: np.ndarray,
    *,
    cmap_name: str = "magma",
    invalid_color: tuple[int, int, int] = (0, 0, 0),
    value_min: float | None = None,
    value_max: float | None = None,
) -> None:
    """float map을 colormap PNG로 저장한다.

    before/after 지도를 직접 비교할 때는 두 호출에 같은 ``value_min``과
    ``value_max``를 전달한다. 값을 생략하면 각 배열의 1/99 percentile을 사용한다.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(value, dtype=np.float32)
    finite = np.isfinite(arr)
    out = np.zeros((*arr.shape, 3), dtype=np.uint8)
    if np.any(finite):
        lo = float(np.nanpercentile(arr[finite], 1.0)) if value_min is None else float(value_min)
        hi = float(np.nanpercentile(arr[finite], 99.0)) if value_max is None else float(value_max)
        norm = (arr - lo) / max(hi - lo, 1.0e-8)
        norm = np.clip(norm, 0.0, 1.0)
        cmap = matplotlib.colormaps.get_cmap(cmap_name)
        out = (cmap(norm)[..., :3] * 255.0).astype(np.uint8)
    out[~finite] = invalid_color
    Image.fromarray(out).save(path)


def save_depth(path: Path, depth: np.ndarray) -> None:
    save_heatmap(path, depth, cmap_name="Spectral")


def save_normal(path: Path, normal: np.ndarray) -> None:
    """[-1,1] normal을 RGB visualization으로 저장한다."""

    path.parent.mkdir(parents=True, exist_ok=True)
    n = np.asarray(normal, dtype=np.float32)
    out = ((np.nan_to_num(n, nan=0.0) + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(out).save(path)


def save_coverage(path: Path, coverage: np.ndarray) -> None:
    """front hemisphere coverage: 0=없음, 1=observed, 2=unknown."""

    path.parent.mkdir(parents=True, exist_ok=True)
    colors = np.zeros((*coverage.shape, 3), dtype=np.uint8)
    colors[coverage == 1] = np.array([40, 190, 90], dtype=np.uint8)
    colors[coverage == 2] = np.array([220, 80, 80], dtype=np.uint8)
    Image.fromarray(colors).save(path)
