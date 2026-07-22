from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from wide_fov_supervision_v2.config import PipelineConfig
from wide_fov_supervision_v2.datasets.nyu.dataset import NYU_CX, NYU_CY, NYU_FX, NYU_FY, NYURawDataset
from wide_fov_supervision_v2.datasets.nyu.splits import read_nyu_split
from wide_fov_supervision_v2.modules.quad_completion.geometry import (
    CORNER_RELATIVE_UV,
    bilinear_quad_map,
    pinhole_rays_from_xy,
    quad_is_valid,
)


def quad_manifest_hash(config: PipelineConfig) -> str:
    """Manifest 좌표 분포를 바꾸는 설정만 해시한다."""

    payload = {
        "schema": config.train.quad_manifest_schema_version,
        "seed": config.train.seed,
        "train_quads": config.train.train_quads_per_frame,
        "test_quads": config.train.test_quads_per_frame,
        "side": [config.train.quad_min_side_px, config.train.quad_max_side_px],
        "rotation": config.train.quad_max_rotation_degrees,
        "jitter": config.train.quad_corner_jitter_fraction,
        "guided_fraction": config.train.guided_quad_fraction,
        "continuous_fraction": config.train.continuous_quad_fraction,
        "target_surface_gap_m": config.ray.target_surface_gap_m,
        "depth_edge_threshold": config.loss.depth_discontinuity_log_threshold,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def quad_manifest_root(config: PipelineConfig) -> Path:
    return config.paths.outputs / "cache" / "nyu_quad_manifest" / quad_manifest_hash(config)


def _sample_bilinear(image: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Array-index 좌표에서 bilinear sampling하고 네 이웃 유효성을 반환한다."""

    value = np.asarray(image)
    coords = np.asarray(xy, dtype=np.float32)
    h, w = value.shape[:2]
    x, y = coords[..., 0], coords[..., 1]
    inside = (x >= 0.0) & (x <= w - 1.0) & (y >= 0.0) & (y <= h - 1.0)
    x0 = np.floor(np.clip(x, 0.0, w - 1.0)).astype(np.int64)
    y0 = np.floor(np.clip(y, 0.0, h - 1.0)).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    wx = x - x0
    wy = y - y0
    v00, v10 = value[y0, x0], value[y0, x1]
    v01, v11 = value[y1, x0], value[y1, x1]
    if value.ndim == 3:
        wx = wx[..., None]
        wy = wy[..., None]
    sampled = (
        (1.0 - wx) * (1.0 - wy) * v00
        + wx * (1.0 - wy) * v10
        + (1.0 - wx) * wy * v01
        + wx * wy * v11
    )
    if value.ndim == 2:
        neighbors_valid = (
            np.isfinite(v00) & (v00 > 0.0)
            & np.isfinite(v10) & (v10 > 0.0)
            & np.isfinite(v01) & (v01 > 0.0)
            & np.isfinite(v11) & (v11 > 0.0)
        )
    else:
        neighbors_valid = np.isfinite(v00).all(axis=-1) & np.isfinite(v10).all(axis=-1)
        neighbors_valid &= np.isfinite(v01).all(axis=-1) & np.isfinite(v11).all(axis=-1)
    return sampled.astype(np.float32), (inside & neighbors_valid)


def _random_quad(
    rng: np.random.Generator,
    config: PipelineConfig,
    width: int,
    height: int,
    *,
    center_hint: np.ndarray | None = None,
) -> np.ndarray:
    """회전 rectangle에 독립 corner jitter를 더한 일반 convex-quad 후보를 만든다."""

    min_side = float(config.train.quad_min_side_px)
    max_side = float(config.train.quad_max_side_px)
    side_u = float(np.exp(rng.uniform(np.log(min_side), np.log(max_side))))
    side_v = float(np.exp(rng.uniform(np.log(min_side), np.log(max_side))))
    margin = 0.75 * max(side_u, side_v) + 2.0
    if margin * 2.0 >= min(width, height):
        margin = 2.0
    if center_hint is None:
        center = np.array(
            [rng.uniform(margin, width - 1.0 - margin), rng.uniform(margin, height - 1.0 - margin)],
            dtype=np.float32,
        )
    else:
        center = np.asarray(center_hint, dtype=np.float32) + rng.uniform(-2.0, 2.0, size=2)
        center = np.clip(center, [margin, margin], [width - 1.0 - margin, height - 1.0 - margin]).astype(np.float32)
    local = np.array(
        [
            [-0.5 * side_u, -0.5 * side_v],
            [0.5 * side_u, -0.5 * side_v],
            [0.5 * side_u, 0.5 * side_v],
            [-0.5 * side_u, 0.5 * side_v],
        ],
        dtype=np.float32,
    )
    theta = np.deg2rad(rng.uniform(-config.train.quad_max_rotation_degrees, config.train.quad_max_rotation_degrees))
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32)
    jitter_scale = float(config.train.quad_corner_jitter_fraction) * min(side_u, side_v)
    jitter = rng.uniform(-jitter_scale, jitter_scale, size=(4, 2)).astype(np.float32)
    return (center + local @ rotation.T + jitter).astype(np.float32)


def _quad_properties(depth: np.ndarray, corners: np.ndarray, config: PipelineConfig) -> tuple[bool, bool, float]:
    """지원점 유효성, 연속 표면 여부, 3D support edge 최대 간격을 계산한다."""

    corner_depth, corner_valid = _sample_bilinear(depth, corners)
    if not np.all(corner_valid):
        return False, False, float("nan")
    rays = pinhole_rays_from_xy(corners, fx=NYU_FX, fy=NYU_FY, cx=NYU_CX, cy=NYU_CY)
    points = corner_depth[:, None] / rays[:, 2:3] * rays
    edge_pairs = ((0, 1), (1, 2), (2, 3), (3, 0))
    max_gap = max(float(np.linalg.norm(points[b] - points[a])) for a, b in edge_pairs)

    axis = np.linspace(0.0, 1.0, 7, dtype=np.float32)
    u, v = np.meshgrid(axis, axis)
    grid_uv = np.stack([u, v], axis=-1).reshape(1, -1, 2)
    grid_xy = bilinear_quad_map(corners[None], grid_uv)[0].reshape(7, 7, 2)
    grid_depth, grid_valid = _sample_bilinear(depth, grid_xy)
    if not np.all(grid_valid):
        return False, False, max_gap
    log_depth = np.log(grid_depth.clip(min=1.0e-6))
    max_local_jump = max(
        float(np.max(np.abs(np.diff(log_depth, axis=0)))),
        float(np.max(np.abs(np.diff(log_depth, axis=1)))),
    )
    continuous = max_local_jump <= float(config.loss.depth_discontinuity_log_threshold)
    return True, continuous, max_gap


def generate_frame_quad_manifest(
    depth_z: np.ndarray,
    *,
    frame_index: int,
    split: Literal["train", "test"],
    config: PipelineConfig,
) -> dict[str, np.ndarray]:
    """한 NYU frame에서 결정적인 convex-quad 좌표 manifest를 생성한다."""

    h, w = depth_z.shape
    count = config.train.train_quads_per_frame if split == "train" else config.train.test_quads_per_frame
    rng = np.random.default_rng(np.random.SeedSequence([config.train.seed, frame_index, 0 if split == "train" else 1]))
    corners_out: list[np.ndarray] = []
    continuous_out: list[bool] = []
    guided_out: list[bool] = []
    gap_out: list[float] = []
    target_gap = float(config.ray.target_surface_gap_m)
    safe_log_depth = np.log(np.where(np.isfinite(depth_z) & (depth_z > 0.0), depth_z, 1.0))
    edge_threshold = float(config.loss.depth_discontinuity_log_threshold)
    edge_map = np.zeros_like(depth_z, dtype=bool)
    edge_map[:, 1:] |= np.abs(np.diff(safe_log_depth, axis=1)) > edge_threshold
    edge_map[1:, :] |= np.abs(np.diff(safe_log_depth, axis=0)) > edge_threshold
    edge_y, edge_x = np.nonzero(edge_map)

    for sample_index in range(int(count)):
        want_guided = sample_index < int(round(count * config.train.guided_quad_fraction))
        want_continuous = (sample_index % 10) < int(round(10 * config.train.continuous_quad_fraction))
        fallback: tuple[np.ndarray, bool, float] | None = None
        for _ in range(int(config.train.manifest_max_attempts_per_quad)):
            center_hint = None
            if not want_continuous and len(edge_x) > 0:
                edge_index = int(rng.integers(0, len(edge_x)))
                center_hint = np.array([edge_x[edge_index], edge_y[edge_index]], dtype=np.float32)
            corners = _random_quad(rng, config, w, h, center_hint=center_hint)
            if not quad_is_valid(corners, image_width=w, image_height=h):
                continue
            valid, continuous, gap = _quad_properties(depth_z, corners, config)
            if not valid:
                continue
            fallback = (corners, continuous, gap)
            guided = 2.0 * target_gap <= gap <= 8.0 * target_gap
            if continuous == want_continuous and (not want_guided or guided):
                break
        if fallback is None:
            raise RuntimeError(f"NYU frame {frame_index}에서 유효 convex quad를 생성하지 못했습니다.")
        corners, continuous, gap = fallback
        corners_out.append(corners)
        continuous_out.append(continuous)
        guided_out.append(2.0 * target_gap <= gap <= 8.0 * target_gap)
        gap_out.append(gap)

    return {
        "corners_xy": np.stack(corners_out).astype(np.float32),
        "source_continuous": np.asarray(continuous_out, dtype=bool),
        "guided_gap_match": np.asarray(guided_out, dtype=bool),
        "support_edge_gap_m": np.asarray(gap_out, dtype=np.float32),
    }


def build_nyu_quad_manifest(config: PipelineConfig, split: Literal["train", "test"]) -> Path:
    """NYU RGB-D를 중복 저장하지 않고 frame별 quad 좌표만 cache한다."""

    max_items = config.train.max_train_items if split == "train" else config.train.max_eval_items
    dataset = NYURawDataset(config.paths, split=split, max_items=max_items)
    out_dir = quad_manifest_root(config) / split
    out_dir.mkdir(parents=True, exist_ok=True)
    for frame in tqdm(dataset, desc=f"cache NYU convex quads {split}"):
        path = out_dir / f"{frame.index:06d}.npz"
        if path.exists():
            try:
                with np.load(path) as current:
                    if int(current["schema_version"]) == config.train.quad_manifest_schema_version:
                        continue
            except (OSError, KeyError, ValueError):
                pass
        payload = generate_frame_quad_manifest(frame.depth_z, frame_index=frame.index, split=split, config=config)
        np.savez_compressed(
            path,
            **payload,
            frame_index=np.array(frame.index, dtype=np.int32),
            schema_version=np.array(config.train.quad_manifest_schema_version, dtype=np.int16),
            manifest_hash=np.array(quad_manifest_hash(config)),
        )
    return out_dir


def _confidence_targets(depth: np.ndarray, query_xy: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.float32)
    samples = []
    valid = []
    for offset in offsets:
        value, mask = _sample_bilinear(depth, query_xy + offset)
        samples.append(value)
        valid.append(mask)
    values = np.stack(samples, axis=-1)
    all_valid = np.stack(valid, axis=-1).all(axis=-1)
    safe = np.where(all_valid[..., None], values, 1.0)
    log_range = np.max(np.log(safe.clip(min=1.0e-6)), axis=-1) - np.min(
        np.log(safe.clip(min=1.0e-6)), axis=-1
    )
    confidence = all_valid & (log_range <= float(threshold))
    return valid[0].astype(bool), confidence.astype(bool)


class NYUQuadCompletionDataset(Dataset):
    """Coordinate manifest와 원본 HDF5를 결합하는 fixed-shape 학습 dataset.

    각 item은 random 내부 query 64개와 corner reconstruction query 4개를 갖는다.
    HDF5 RGB-D는 실행 중 bilinear sampling하며 manifest에는 좌표만 저장한다.
    """

    def __init__(self, config: PipelineConfig, split: Literal["train", "test"]) -> None:
        self.config = config
        self.split = split
        max_items = config.train.max_train_items if split == "train" else config.train.max_eval_items
        indices = read_nyu_split(config.paths.nyu_split_train if split == "train" else config.paths.nyu_split_test)
        if max_items is not None:
            indices = indices[: int(max_items)]
        self.entries: list[tuple[int, int, Path]] = []
        for frame_index in indices:
            path = quad_manifest_root(config) / split / f"{int(frame_index):06d}.npz"
            if not path.exists():
                raise FileNotFoundError(f"Convex quad manifest가 없습니다: {path}. `python run.py --mode cache`를 먼저 실행하세요.")
            with np.load(path) as manifest:
                if int(manifest["schema_version"]) != config.train.quad_manifest_schema_version:
                    raise RuntimeError(f"Quad manifest schema가 현재 설정과 다릅니다: {path}")
                quad_count = len(manifest["corners_xy"])
            self.entries.extend((int(frame_index), quad_index, path) for quad_index in range(quad_count))
        self._h5: h5py.File | None = None

    def __len__(self) -> int:
        return len(self.entries)

    def _file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.config.paths.nyu_mat, "r")
        return self._h5

    def __getitem__(self, item: int) -> dict[str, np.ndarray]:
        frame_index, quad_index, manifest_path = self.entries[item]
        h5 = self._file()
        rgb = np.asarray(h5["images"][frame_index], dtype=np.uint8).transpose(2, 1, 0).astype(np.float32) / 255.0
        depth = np.asarray(h5["depths"][frame_index], dtype=np.float32).T
        with np.load(manifest_path) as manifest:
            corners = manifest["corners_xy"][quad_index].astype(np.float32)
            source_continuous = bool(manifest["source_continuous"][quad_index])

        rng = np.random.default_rng(np.random.SeedSequence([self.config.train.seed, frame_index, quad_index, 17]))
        random_uv = rng.uniform(0.0, 1.0, size=(self.config.train.queries_per_quad, 2)).astype(np.float32)
        relative_uv = np.concatenate([random_uv, CORNER_RELATIVE_UV], axis=0)
        query_xy = bilinear_quad_map(corners[None], relative_uv[None])[0]

        support_rgb, support_rgb_valid = _sample_bilinear(rgb, corners)
        support_depth, support_depth_valid = _sample_bilinear(depth, corners)
        support_valid = support_rgb_valid & support_depth_valid
        query_rgb, query_rgb_valid = _sample_bilinear(rgb, query_xy)
        query_depth, query_valid = _sample_bilinear(depth, query_xy)
        query_valid &= query_rgb_valid
        confidence_valid, confidence = _confidence_targets(
            depth, query_xy, self.config.loss.depth_discontinuity_log_threshold
        )
        confidence &= query_valid & confidence_valid

        step = float(self.config.ray.normal_stencil_relative_step)
        offsets = np.array([[-step, 0.0], [step, 0.0], [0.0, -step], [0.0, step]], dtype=np.float32)
        stencil_uv = np.clip(relative_uv[:, None, :] + offsets[None, :, :], 0.0, 1.0)
        stencil_xy = bilinear_quad_map(corners[None], stencil_uv.reshape(1, -1, 2))[0].reshape(-1, 4, 2)
        stencil_depth, stencil_valid = _sample_bilinear(depth, stencil_xy)

        scale = float(np.exp(rng.uniform(np.log(self.config.train.common_depth_scale_min), np.log(self.config.train.common_depth_scale_max))))
        support_depth = support_depth * scale
        query_depth = query_depth * scale
        stencil_depth = stencil_depth * scale
        if self.split == "train" and self.config.train.support_log_depth_noise_std > 0.0:
            noise = rng.normal(0.0, self.config.train.support_log_depth_noise_std, size=4).astype(np.float32)
            support_depth = support_depth * np.exp(noise)

        # 마지막 네 query는 corner reconstruction용이므로 입력 support 값을 그대로 target으로 둔다.
        query_rgb[-4:] = support_rgb
        query_depth[-4:] = support_depth
        query_valid[-4:] = support_valid
        confidence[-4:] = support_valid

        query_rays = pinhole_rays_from_xy(query_xy, fx=NYU_FX, fy=NYU_FY, cx=NYU_CX, cy=NYU_CY)
        stencil_rays = pinhole_rays_from_xy(stencil_xy, fx=NYU_FX, fy=NYU_FY, cx=NYU_CX, cy=NYU_CY)
        return {
            "support_ray_dir": pinhole_rays_from_xy(corners, fx=NYU_FX, fy=NYU_FY, cx=NYU_CX, cy=NYU_CY),
            "support_rgb": support_rgb.astype(np.float32),
            "support_depth_z": np.where(support_valid, support_depth, 0.0).astype(np.float32),
            "support_valid": support_valid.astype(bool),
            "query_ray_dir": query_rays,
            "query_relative_uv": relative_uv.astype(np.float32),
            "query_mask": np.ones(len(relative_uv), dtype=bool),
            "target_rgb": np.nan_to_num(query_rgb).astype(np.float32),
            "target_depth_z": np.where(query_valid, query_depth, 0.0).astype(np.float32),
            "target_valid": query_valid.astype(bool),
            "target_confidence": confidence.astype(bool),
            "corner_query_mask": np.arange(len(relative_uv)) >= len(relative_uv) - 4,
            "stencil_ray_dir": stencil_rays.astype(np.float32),
            "stencil_relative_uv": stencil_uv.astype(np.float32),
            "stencil_depth_z": np.where(stencil_valid, stencil_depth, 0.0).astype(np.float32),
            "stencil_valid": stencil_valid.astype(bool),
            "source_continuous": np.array(source_continuous, dtype=bool),
            "frame_index": np.array(frame_index, dtype=np.int32),
        }
