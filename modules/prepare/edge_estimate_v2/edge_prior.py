"""RGB에서 2D edge 보조 신호를 만드는 작은 유틸리티."""

from __future__ import annotations

import cv2
import numpy as np

from .config import EdgePriorConfig


def _normalize_map(value: np.ndarray) -> np.ndarray:
    """강한 outlier가 전체 contrast를 먹지 않도록 percentile 기준으로 0..1 정규화한다."""

    arr = np.asarray(value, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros(arr.shape, dtype=np.float32)
    high = float(np.percentile(arr[finite], 99.0))
    if high <= 1.0e-6:
        return np.zeros(arr.shape, dtype=np.float32)
    return np.clip(arr / high, 0.0, 1.0).astype(np.float32)


def estimate_2d_edge_prior(
    rgb: np.ndarray,
    valid: np.ndarray | None = None,
    config: EdgePriorConfig | None = None,
) -> np.ndarray:
    """Canny 단독보다 안정적인 RGB multi-cue 2D edge prior를 만든다.

    외부 checkpoint 없이 재현 가능하도록 Lab 색공간 Scharr gradient, Laplacian,
    adaptive Canny를 섞는다. 출력은 확률 보정된 값은 아니고, 모델 입력과 후보 cell
    점수에 쓰는 0..1 보조 신호다.
    """

    cfg = config or EdgePriorConfig()
    image = np.asarray(rgb)
    if image.dtype != np.uint8:
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"rgb must be HxWx3, got {image.shape}")

    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=float(cfg.blur_sigma), sigmaY=float(cfg.blur_sigma))
    lab = cv2.cvtColor(blurred, cv2.COLOR_RGB2LAB).astype(np.float32)
    gradient = np.zeros(image.shape[:2], dtype=np.float32)
    for channel in range(3):
        gx = cv2.Scharr(lab[..., channel], cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(lab[..., channel], cv2.CV_32F, 0, 1)
        gradient += gx * gx + gy * gy
    gradient = _normalize_map(np.sqrt(gradient))

    gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
    laplacian = _normalize_map(np.abs(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=3)))
    median = float(np.median(gray))
    lower = int(max(0, (1.0 - cfg.canny_sigma) * median))
    upper = int(min(255, (1.0 + cfg.canny_sigma) * median))
    canny = (cv2.Canny(gray, lower, max(lower + 1, upper)).astype(np.float32) / 255.0)

    edge = (
        float(cfg.gradient_weight) * gradient
        + float(cfg.laplacian_weight) * laplacian
        + float(cfg.canny_weight) * canny
    )
    edge = _normalize_map(cv2.GaussianBlur(edge, (0, 0), sigmaX=0.75, sigmaY=0.75))
    if valid is not None:
        edge = np.where(np.asarray(valid, dtype=bool), edge, 0.0)
    return edge.astype(np.float32)


def cell_edge_prior(edge_prior: np.ndarray) -> np.ndarray:
    """Pixel prior를 2x2 source cell 단위 prior로 변환한다."""

    edge = np.asarray(edge_prior, dtype=np.float32)
    return np.maximum.reduce([edge[:-1, :-1], edge[:-1, 1:], edge[1:, 1:], edge[1:, :-1]])
