"""NYU RGB-D를 virtual fisheye 5×5 lattice 학습 patch로 변환하고 cache하는 모듈."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Literal

import cv2
import h5py
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from wide_fov_supervision_v2.backbone.depth_anything import DepthAnythingMetricWrapper
from wide_fov_supervision_v2.config import FisheyeCameraConfig
from wide_fov_supervision_v2.datasets.nyu.dataset import (
    NYUFrame,
    orientation_for_frame,
    sample_nyu_depth_at_fisheye_rays,
    virtual_fisheye_from_nyu,
)
from wide_fov_supervision_v2.datasets.nyu.splits import read_nyu_split
from wide_fov_supervision_v2.modules.camera_geometry import build_fisheye_rays, project_fisheye_rays

from .config import EdgeEstimateConfig, Variant
from .edge_prior import estimate_2d_edge_prior
from .pseudo_labels import build_pseudo_edges


def _remap_flat(
    array: np.ndarray,
    points: np.ndarray,
    *,
    interpolation: int,
    border_value: float | int,
) -> np.ndarray:
    """OpenCV의 한 축 32,767 제한을 피하며 임의 개수 query를 sampling한다."""

    flat = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    source = np.asarray(array)
    tail = () if source.ndim == 2 else (source.shape[-1],)
    if len(flat) == 0:
        return np.empty((0,) + tail, dtype=source.dtype)
    chunks: list[np.ndarray] = []
    for start in range(0, len(flat), 30_000):
        chunk = flat[start : start + 30_000]
        sampled = cv2.remap(
            source,
            (chunk[:, 0] - 0.5).reshape(1, -1),
            (chunk[:, 1] - 0.5).reshape(1, -1),
            interpolation=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value,
        )
        chunks.append(sampled.reshape((-1,) + tail))
    return np.concatenate(chunks, axis=0)


def _sample_bilinear(array: np.ndarray, uv: np.ndarray, *, border_value: float = 0.0) -> np.ndarray:
    """pixel-center 좌표 ``uv``에서 OpenCV bilinear sampling을 수행한다."""

    points = np.asarray(uv, dtype=np.float32)
    shape = points.shape[:-1]
    sampled = _remap_flat(
        array,
        points,
        interpolation=cv2.INTER_LINEAR,
        border_value=border_value,
    )
    tail = () if array.ndim == 2 else (array.shape[-1],)
    return sampled.reshape(shape + tail)


def _sample_nearest(array: np.ndarray, uv: np.ndarray, *, border_value: int = 0) -> np.ndarray:
    points = np.asarray(uv, dtype=np.float32)
    shape = points.shape[:-1]
    sampled = _remap_flat(
        array,
        points,
        interpolation=cv2.INTER_NEAREST,
        border_value=border_value,
    )
    tail = () if array.ndim == 2 else (array.shape[-1],)
    return sampled.reshape(shape + tail)


def _sample_bilinear_masked(value: np.ndarray, valid: np.ndarray, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NaN/invalid 이웃이 0으로 섞이지 않도록 유효 bilinear weight로 다시 정규화한다."""

    mask = np.asarray(valid, dtype=np.float32)
    numerator = _sample_bilinear(np.where(valid, value, 0.0).astype(np.float32), uv)
    denominator = _sample_bilinear(mask, uv)
    sampled = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float32),
        where=denominator > 1.0e-6,
    )
    return sampled.astype(np.float32), denominator.astype(np.float32)


def spherical_bilerp(corner_rays: np.ndarray, relative_uv: np.ndarray) -> np.ndarray:
    """네 corner ray를 bilinear 혼합한 뒤 단위화해 subpixel query ray를 만든다.

    corner 순서는 ``p00, p10, p11, p01``이다. 실제 fisheye 투영식의 역함수를
    매 query마다 호출하지 않아도 되며 기존 completion과 같은 결정적 정의를 쓴다.
    """

    corners = np.asarray(corner_rays, dtype=np.float32)
    uv = np.asarray(relative_uv, dtype=np.float32)
    u, v = uv[..., 0], uv[..., 1]
    weights = np.stack([(1 - u) * (1 - v), u * (1 - v), u * v, (1 - u) * v], axis=-1)
    rays = np.sum(weights[..., None] * corners[..., None, :, :], axis=-2)
    norm = np.linalg.norm(rays, axis=-1, keepdims=True)
    return np.divide(rays, norm, out=np.zeros_like(rays), where=norm > 1.0e-8).astype(np.float32)


def _query_grid(grid_size: int) -> np.ndarray:
    if grid_size < 2:
        raise ValueError("query_grid_size는 공유 cell 경계를 포함하기 위해 2 이상이어야 합니다.")
    # 0과 1을 포함해야 인접 cell이 공유하는 변의 query 위치와 ray가 정확히 같다.
    axis = np.linspace(0.0, 1.0, grid_size, dtype=np.float32)
    u, v = np.meshgrid(axis, axis)
    return np.stack([u, v], axis=-1).reshape(-1, 2)


def _lattice_uv(origin_x: int, origin_y: int, span: int, points: int = 5) -> np.ndarray:
    x = origin_x + np.arange(points, dtype=np.float32) * span + 0.5
    y = origin_y + np.arange(points, dtype=np.float32) * span + 0.5
    u, v = np.meshgrid(x, y)
    return np.stack([u, v], axis=-1)


def _cell_queries(
    lattice_uv: np.ndarray,
    lattice_rays: np.ndarray,
    grid_size: int,
    camera: FisheyeCameraConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if lattice_uv.shape != (5, 5, 2) or lattice_rays.shape != (5, 5, 3):
        raise ValueError("lattice_uv/lattice_rays는 (5,5,2)/(5,5,3)이어야 합니다.")
    relative = _query_grid(grid_size)
    q = len(relative)
    query_xy = np.empty((4, 4, q, 2), dtype=np.float32)
    query_rays = np.empty((4, 4, q, 3), dtype=np.float32)
    relative_grid = np.broadcast_to(relative, (4, 4, q, 2)).copy()
    for cy in range(4):
        for cx in range(4):
            corners_ray = np.stack(
                [
                    lattice_rays[cy, cx],
                    lattice_rays[cy, cx + 1],
                    lattice_rays[cy + 1, cx + 1],
                    lattice_rays[cy + 1, cx],
                ]
            )
            query_rays[cy, cx] = spherical_bilerp(corners_ray[None], relative[None])[0]
            projected, _ = project_fisheye_rays(query_rays[cy, cx], camera)
            query_xy[cy, cx] = projected.astype(np.float32)
    return query_xy, query_rays, relative_grid


def _cell_type_from_queries(edge_target: np.ndarray, type_target: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    result = np.zeros((4, 4), dtype=np.int64)
    for y in range(4):
        for x in range(4):
            mask = (edge_target[y, x] >= 0.5) & (confidence[y, x] >= 0.5) & (type_target[y, x] > 0)
            if np.any(mask):
                values = type_target[y, x][mask]
                counts = np.bincount(values.astype(np.int64), minlength=4)
                result[y, x] = int(np.argmax(counts[1:]) + 1)
    return result


def _make_patch(
    *,
    rgb: np.ndarray,
    depth: np.ndarray,
    source_valid: np.ndarray,
    rays: np.ndarray,
    labels,
    da_relative: np.ndarray | None,
    edge_prior_2d: np.ndarray | None,
    origin_x: int,
    origin_y: int,
    span: int,
    grid_size: int,
    depth_prior_blur_sigma_px: float,
    frame_index: int,
    camera: FisheyeCameraConfig,
) -> dict[str, np.ndarray]:
    lattice = _lattice_uv(origin_x, origin_y, span)
    ix = np.rint(lattice[..., 0] - 0.5).astype(np.int64)
    iy = np.rint(lattice[..., 1] - 0.5).astype(np.int64)
    support_rgb = rgb[iy, ix].astype(np.float32) / 255.0
    support_rays = rays[iy, ix].astype(np.float32)
    support_valid = source_valid[iy, ix].astype(bool)
    if edge_prior_2d is None:
        support_edge = np.zeros((5, 5), dtype=np.float32)
    else:
        support_edge = edge_prior_2d[iy, ix].astype(np.float32)
    query_xy, query_rays, query_relative = _cell_queries(lattice, support_rays, grid_size, camera)
    query_edge = _sample_bilinear(labels.edge_soft, query_xy).astype(np.float32)
    query_type = _sample_nearest(labels.edge_type, query_xy).astype(np.int64)
    query_confidence = _sample_bilinear(labels.confidence, query_xy).astype(np.float32)
    query_ignore = _sample_nearest(labels.ignore.astype(np.uint8), query_xy).astype(bool)
    query_near, near_weight = _sample_bilinear_masked(
        labels.near_depth_z, np.isfinite(labels.near_depth_z), query_xy
    )
    query_far, far_weight = _sample_bilinear_masked(
        labels.far_depth_z, np.isfinite(labels.far_depth_z), query_xy
    )
    # crease는 연속 표면 위의 query이므로 dense metric depth를 직접 사용한다.
    query_dense_depth = _sample_bilinear(depth, query_xy).astype(np.float32)
    prior_source = cv2.GaussianBlur(
        depth.astype(np.float32),
        (0, 0),
        sigmaX=float(depth_prior_blur_sigma_px),
        sigmaY=float(depth_prior_blur_sigma_px),
    )
    query_prior_depth, prior_weight = _sample_bilinear_masked(
        prior_source, np.isfinite(depth) & (depth > 0.0), query_xy
    )
    query_near = np.where(query_type == 1, query_dense_depth, query_near)
    query_valid = (~query_ignore) & np.isfinite(query_xy).all(axis=-1)
    query_valid &= np.isfinite(query_rays).all(axis=-1) & (query_rays[..., 2] > 1.0e-3)
    cell_edge = np.max(np.where(query_valid, query_edge, 0.0), axis=-1) >= 0.5
    cell_type = _cell_type_from_queries(query_edge, query_type, query_confidence)
    cell_valid = (
        support_valid[:-1, :-1]
        & support_valid[:-1, 1:]
        & support_valid[1:, 1:]
        & support_valid[1:, :-1]
    )
    query_valid &= cell_valid[..., None]
    high_confidence = query_confidence >= 0.5
    near_valid = query_valid & high_confidence & (query_edge >= 0.5) & (query_type > 0) & (query_near > 0.0)
    near_valid &= ((query_type == 1) | (near_weight > 1.0e-6))
    far_valid = query_valid & high_confidence & (query_type == 2) & (query_far > 0.0) & (far_weight > 1.0e-6)
    query_confidence = np.where(query_valid, query_confidence, 0.0)
    if da_relative is None:
        da_support = np.zeros((5, 5), dtype=np.float32)
        da_valid = np.zeros((5, 5), dtype=bool)
    else:
        da_support = da_relative[iy, ix].astype(np.float32)
        da_valid = support_valid & np.isfinite(da_support)
        da_support = np.nan_to_num(da_support)
    return {
        "support_rgb": support_rgb,
        "support_ray_dir": support_rays,
        "support_valid": support_valid,
        "support_edge_2d": support_edge,
        "da_relative_log_depth": da_support,
        "da_valid": da_valid,
        "query_ray_dir": query_rays,
        "query_relative_uv": query_relative,
        "query_prior_depth_z": np.nan_to_num(query_prior_depth, nan=0.0).astype(np.float32),
        "query_prior_valid": (prior_weight > 1.0e-6) & np.isfinite(query_prior_depth) & (query_prior_depth > 0.0),
        "query_source_uv": query_xy,
        "query_mask": query_valid,
        "target_cell_edge": cell_edge.astype(np.float32),
        "target_cell_type": cell_type.astype(np.int64),
        "target_query_edge": query_edge,
        "target_query_type": query_type,
        "target_near_depth_z": np.where(near_valid, query_near, 0.0).astype(np.float32),
        "target_far_depth_z": np.where(far_valid, query_far, 0.0).astype(np.float32),
        "target_near_valid": near_valid,
        "target_far_valid": far_valid,
        "target_confidence": query_confidence,
        "cell_valid": cell_valid,
        "frame_index": np.array(frame_index, dtype=np.int32),
        "cell_span_px": np.array(span, dtype=np.int16),
        "origin_xy": np.array([origin_x, origin_y], dtype=np.int32),
    }


def _choose_center(
    rng: np.random.Generator,
    labels,
    source_valid: np.ndarray,
    want_edge: bool,
    positive_kind: int,
) -> tuple[int, int] | None:
    if want_edge:
        mask = labels.edge & (labels.edge_type == positive_kind) & (labels.confidence >= 0.5)
        if not np.any(mask):
            mask = labels.edge & (labels.confidence >= 0.5)
    else:
        dilated = cv2.dilate(labels.edge.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        mask = source_valid & ~dilated & ~labels.ignore
    y, x = np.nonzero(mask)
    if len(x) == 0:
        return None
    index = int(rng.integers(0, len(x)))
    return int(x[index]), int(y[index])


def _normalize_da(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.full(depth.shape, np.nan, dtype=np.float32)
    good = np.asarray(valid, dtype=bool) & np.isfinite(depth) & (depth > 0.0)
    if np.any(good):
        log_depth = np.log(depth[good])
        result[good] = log_depth - float(np.median(log_depth))
    return result


def build_edge_cache(
    config: EdgeEstimateConfig,
    split: Literal["train", "test"],
    *,
    include_da: bool | None = None,
) -> Path:
    """NYU frame에서 virtual-fisheye lattice patch cache를 생성한다."""

    include_da = config.data.cache_da_v2 if include_da is None else bool(include_da)
    split_path = config.base.paths.nyu_split_train if split == "train" else config.base.paths.nyu_split_test
    indices = read_nyu_split(split_path)
    maximum = config.data.max_train_frames if split == "train" else config.data.max_test_frames
    if maximum is not None:
        indices = indices[: int(maximum)]
    split_root = config.cache_root / split
    patch_root = split_root / "patches"
    patch_root.mkdir(parents=True, exist_ok=True)
    rays = build_fisheye_rays(config.base.camera)
    patches_per_frame = config.data.train_patches_per_frame if split == "train" else config.data.test_patches_per_frame
    da_runner = DepthAnythingMetricWrapper(config.base.paths, config.base.backbone) if include_da else None
    records: list[str] = []
    log_depth_samples: list[np.ndarray] = []
    edge_patch_count = 0
    with h5py.File(config.base.paths.nyu_mat, "r") as h5:
        for frame_index in tqdm(indices, desc=f"edge cache {split}", disable=not sys.stderr.isatty()):
            frame_index = int(frame_index)
            expected_outputs = [
                patch_root / f"{frame_index:06d}_{patch_index:03d}.npz"
                for patch_index in range(int(patches_per_frame))
            ]
            da_file = split_root / "da_v2" / f"{frame_index:06d}.npy"
            frame_cache_complete = all(output.exists() for output in expected_outputs)
            frame_cache_complete &= da_runner is None or da_file.exists()
            if frame_cache_complete:
                for output in expected_outputs:
                    records.append(str(output.relative_to(config.cache_root)).replace("\\", "/"))
                    with np.load(output) as cached:
                        cached_mask = cached["target_near_valid"]
                        cached_depth = cached["target_near_depth_z"]
                        edge_patch_count += int(np.any(cached["target_cell_edge"] >= 0.5))
                        if np.any(cached_mask):
                            log_depth_samples.append(np.log(cached_depth[cached_mask]))
                continue
            _, orientation = orientation_for_frame(frame_index, config.data.orientations_degrees)
            rgb = np.asarray(h5["images"][frame_index], dtype=np.uint8).transpose(2, 1, 0)
            dense = np.asarray(h5["depths"][frame_index], dtype=np.float32).T
            raw = np.asarray(h5["rawDepths"][frame_index], dtype=np.float32).T
            frame = NYUFrame(rgb=rgb, depth_z=dense, index=frame_index)
            virtual = virtual_fisheye_from_nyu(frame, config.base.camera, rays.rays_cv, orientation)
            raw_samples = sample_nyu_depth_at_fisheye_rays(
                raw, rays.rays_cv, orientation, geometry_z_eps=config.base.camera.geometry_z_eps
            )
            raw_valid = raw_samples.gt_valid
            source_valid = rays.valid & virtual.source_observed & virtual.gt_valid
            labels = build_pseudo_edges(virtual.depth_gt_z, raw_valid, rays.rays_cv, config.data)
            edge_prior_2d = (
                estimate_2d_edge_prior(virtual.rgb, source_valid, config.edge_prior)
                if config.edge_prior.enabled
                else None
            )
            da_relative = None
            if da_runner is not None:
                if da_file.exists():
                    da_relative = np.load(da_file).astype(np.float32)
                else:
                    da_depth = da_runner.predict(virtual.rgb)
                    da_relative = _normalize_da(da_depth, source_valid)
                    da_file.parent.mkdir(parents=True, exist_ok=True)
                    np.save(da_file, da_relative.astype(np.float16))

            rng = np.random.default_rng(np.random.SeedSequence([config.train.seed, frame_index, 0 if split == "train" else 1]))
            for patch_index in range(int(patches_per_frame)):
                output = patch_root / f"{frame_index:06d}_{patch_index:03d}.npz"
                if output.exists():
                    records.append(str(output.relative_to(config.cache_root)).replace("\\", "/"))
                    with np.load(output) as cached:
                        cached_mask = cached["target_near_valid"]
                        cached_depth = cached["target_near_depth_z"]
                        edge_patch_count += int(np.any(cached["target_cell_edge"] >= 0.5))
                        if np.any(cached_mask):
                            log_depth_samples.append(np.log(cached_depth[cached_mask]))
                    continue
                want_edge = patch_index < int(round(patches_per_frame * config.data.edge_center_fraction))
                positive_kind = 1 + (patch_index % 2)
                sample = None
                # 일부 orientation에는 reliable edge 또는 hard negative가 전혀 없을 수 있다.
                # 먼저 요청한 종류를 찾고, 불가능할 때만 반대 종류로 결정적으로 대체한다.
                for required_edge in (want_edge, not want_edge):
                    for _ in range(int(config.data.max_attempts_per_patch)):
                        span = int(config.data.cell_spans_px[int(rng.integers(0, len(config.data.cell_spans_px)))])
                        center = _choose_center(rng, labels, source_valid, required_edge, positive_kind)
                        if center is None:
                            break
                        origin_x = int(round(center[0] - 2 * span))
                        origin_y = int(round(center[1] - 2 * span))
                        if origin_x < 0 or origin_y < 0:
                            continue
                        if origin_x + 4 * span >= config.base.camera.width or origin_y + 4 * span >= config.base.camera.height:
                            continue
                        lattice = _lattice_uv(origin_x, origin_y, span)
                        ix = np.rint(lattice[..., 0] - 0.5).astype(np.int64)
                        iy = np.rint(lattice[..., 1] - 0.5).astype(np.int64)
                        if not np.all(source_valid[iy, ix]):
                            continue
                        sample = _make_patch(
                            rgb=virtual.rgb,
                            depth=virtual.depth_gt_z,
                            source_valid=source_valid,
                            rays=rays.rays_cv,
                            labels=labels,
                            da_relative=da_relative,
                            edge_prior_2d=edge_prior_2d,
                            origin_x=origin_x,
                            origin_y=origin_y,
                            span=span,
                            grid_size=config.data.query_grid_size,
                            depth_prior_blur_sigma_px=config.data.depth_prior_blur_sigma_px,
                            frame_index=frame_index,
                            camera=config.base.camera,
                        )
                        has_edge = bool(np.any(sample["target_cell_edge"]))
                        if has_edge == required_edge:
                            break
                        sample = None
                    if sample is not None:
                        break
                if sample is None:
                    raise RuntimeError(f"NYU frame {frame_index}에서 edge patch {patch_index}를 만들지 못했습니다.")
                np.savez_compressed(output, **sample)
                records.append(str(output.relative_to(config.cache_root)).replace("\\", "/"))
                edge_patch_count += int(np.any(sample["target_cell_edge"] >= 0.5))
                mask = sample["target_near_valid"]
                if np.any(mask):
                    log_depth_samples.append(np.log(sample["target_near_depth_z"][mask]))

    metadata = {
        "schema_version": config.data.schema_version,
        "split": split,
        "records": records,
        "include_da": bool(include_da),
        "edge_patch_count": int(edge_patch_count),
        "edge_patch_fraction": float(edge_patch_count / max(len(records), 1)),
        "log_depth_mean": float(np.mean(np.concatenate(log_depth_samples))) if log_depth_samples else config.model.log_depth_mean,
        "log_depth_std": float(np.std(np.concatenate(log_depth_samples))) if log_depth_samples else config.model.log_depth_std,
    }
    (split_root / "index.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return split_root


class EdgePatchDataset(Dataset):
    """cache된 5x5 support lattice patch를 읽는 dataset."""

    def __init__(self, config: EdgeEstimateConfig, split: Literal["train", "test"], variant: Variant) -> None:
        self.config = config
        self.split = split
        self.variant = variant
        index_path = config.cache_root / split / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Edge cache가 없습니다: {index_path}. --mode cache를 먼저 실행하세요.")
        self.metadata = json.loads(index_path.read_text(encoding="utf-8"))
        if int(self.metadata["schema_version"]) != int(config.data.schema_version):
            raise RuntimeError("Edge cache schema가 현재 코드와 다릅니다. cache를 다시 생성하세요.")
        if variant == "rgb_da_context" and not bool(self.metadata.get("include_da")):
            raise RuntimeError("rgb_da_context 학습에는 DA-V2 cache가 필요합니다.")
        self.paths = [config.cache_root / item for item in self.metadata["records"]]

    @property
    def log_depth_mean(self) -> float:
        train_meta = self.config.cache_root / "train" / "index.json"
        if train_meta.exists():
            return float(json.loads(train_meta.read_text(encoding="utf-8"))["log_depth_mean"])
        return float(self.metadata["log_depth_mean"])

    @property
    def log_depth_std(self) -> float:
        train_meta = self.config.cache_root / "train" / "index.json"
        if train_meta.exists():
            return max(float(json.loads(train_meta.read_text(encoding="utf-8"))["log_depth_std"]), 1.0e-3)
        return max(float(self.metadata["log_depth_std"]), 1.0e-3)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, item: int) -> dict[str, np.ndarray]:
        with np.load(self.paths[item]) as data:
            result = {key: data[key].copy() for key in data.files}
        if self.variant != "rgb_da_context":
            result["da_relative_log_depth"].fill(0.0)
            result["da_valid"].fill(False)
        return result
