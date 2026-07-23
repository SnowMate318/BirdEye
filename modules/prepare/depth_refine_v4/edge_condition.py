from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def depth_edge_condition(
    depth_z: np.ndarray,
    valid: np.ndarray,
    *,
    threshold: float = 0.10,
    band_radius: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a leakage-free training edge condition from target depth geometry.

    This is used only for cached NYU supervision. In inference the condition is
    read from the frozen V2 edge model outputs.
    """

    depth = np.asarray(depth_z, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(depth) & (depth > 0.0)
    log_depth = np.log(np.where(mask, depth, 1.0))
    gx = np.zeros_like(log_depth, dtype=np.float32)
    gy = np.zeros_like(log_depth, dtype=np.float32)
    gx[:, 1:] = np.abs(log_depth[:, 1:] - log_depth[:, :-1])
    gy[1:, :] = np.abs(log_depth[1:, :] - log_depth[:-1, :])
    edge = ((gx > threshold) | (gy > threshold)) & mask
    kernel = np.ones((2 * int(band_radius) + 1, 2 * int(band_radius) + 1), np.uint8)
    band = cv2.dilate(edge.astype(np.uint8), kernel, iterations=1) > 0
    probability = cv2.GaussianBlur(edge.astype(np.float32), (0, 0), sigmaX=max(float(band_radius), 1.0))
    probability = np.clip(probability / max(float(np.nanmax(probability)), 1.0e-6), 0.0, 1.0)
    confidence = np.where(mask, 1.0, 0.0).astype(np.float32)
    distance = cv2.distanceTransform((~edge & mask).astype(np.uint8), cv2.DIST_L2, 3)
    distance = np.exp(-distance / max(float(band_radius), 1.0)).astype(np.float32)
    condition = np.stack(
        [
            probability,
            confidence,
            distance,
            np.where(edge, 1.0, 0.0).astype(np.float32),
            np.zeros_like(probability, dtype=np.float32),
            np.zeros_like(probability, dtype=np.float32),
        ],
        axis=0,
    )
    return condition.astype(np.float32), band


def load_v2_edge_condition(
    variant_dir: Path,
    target_shape: tuple[int, int],
    depth0_z: np.ndarray,
) -> np.ndarray:
    """Rasterize frozen V2 edge outputs into dense condition channels.

    Channels: edge probability, confidence, distance-to-edge, near/D0 log ratio,
    far/D0 log ratio, occlusion indicator.
    """

    h, w = target_shape
    probability = _load_or_zero(variant_dir / "edge_probability.npy", (h, w))
    confidence = _load_or_zero(variant_dir / "edge_confidence.npy", (h, w))
    near = _load_or_nan(variant_dir / "edge_depth_near_z.npy", (h, w))
    far = _load_or_nan(variant_dir / "edge_depth_far_z.npy", (h, w))
    if probability.shape != (h, w):
        probability = cv2.resize(probability, (w, h), interpolation=cv2.INTER_LINEAR)
    if confidence.shape != (h, w):
        confidence = cv2.resize(confidence, (w, h), interpolation=cv2.INTER_LINEAR)
    if near.shape != (h, w):
        near = cv2.resize(near, (w, h), interpolation=cv2.INTER_NEAREST)
    if far.shape != (h, w):
        far = cv2.resize(far, (w, h), interpolation=cv2.INTER_NEAREST)

    edge_mask = (probability >= 0.5) & (confidence >= 0.3)
    distance = cv2.distanceTransform((~edge_mask).astype(np.uint8), cv2.DIST_L2, 3)
    distance = np.exp(-distance / 3.0).astype(np.float32)
    d0 = np.asarray(depth0_z, dtype=np.float32)
    near_ratio = _safe_log_ratio(near, d0)
    far_ratio = _safe_log_ratio(far, d0)
    occlusion = np.where(np.isfinite(far) & (far > near), 1.0, 0.0).astype(np.float32)
    return np.stack(
        [
            np.nan_to_num(probability, nan=0.0),
            np.nan_to_num(confidence, nan=0.0),
            distance,
            near_ratio,
            far_ratio,
            occlusion,
        ],
        axis=0,
    ).astype(np.float32)


def condition_preview(condition: np.ndarray) -> np.ndarray:
    edge = np.clip(condition[0], 0.0, 1.0)
    confidence = np.clip(condition[1], 0.0, 1.0)
    distance = np.clip(condition[2], 0.0, 1.0)
    return (np.stack([edge, confidence, distance], axis=-1) * 255.0).astype(np.uint8)


def _load_or_zero(path: Path, shape: tuple[int, int]) -> np.ndarray:
    if path.exists():
        return np.asarray(np.load(path), dtype=np.float32)
    return np.zeros(shape, dtype=np.float32)


def _load_or_nan(path: Path, shape: tuple[int, int]) -> np.ndarray:
    if path.exists():
        return np.asarray(np.load(path), dtype=np.float32)
    return np.full(shape, np.nan, dtype=np.float32)


def _safe_log_ratio(value: np.ndarray, reference: np.ndarray) -> np.ndarray:
    valid = np.isfinite(value) & (value > 0.0) & np.isfinite(reference) & (reference > 0.0)
    result = np.zeros_like(reference, dtype=np.float32)
    result[valid] = np.log(value[valid] / reference[valid])
    return np.clip(result, -1.0, 1.0).astype(np.float32)

