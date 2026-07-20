from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from wide_fov_supervision_v2.config import PipelineConfig
from wide_fov_supervision_v2.datasets.nyu.dataset import (
    NYUFrame,
    NYURawDataset,
    VirtualFisheyeOrientation,
    orientation_for_frame,
    sample_nyu_depth_at_fisheye_rays,
)


QueryBuilder = Callable[[Mapping[str, np.ndarray], PipelineConfig, np.random.Generator], Any]


def _canonical_jsonable(value: Any) -> Any:
    """dataclass/NumPy 값을 hash용 JSON 값으로 정규화한다."""

    if is_dataclass(value):
        return _canonical_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _canonical_jsonable(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def query_sampler_config_hash(config: PipelineConfig) -> str:
    """query 위치와 GT sidecar 내용에 영향을 주는 설정의 SHA-256 hash를 만든다.

    teacher cache는 sampler 설정이 바뀌어도 재사용할 수 있다. 반면 query sidecar는
    아래 설정이 달라지면 별도 디렉터리를 사용해 이전 query와 섞이지 않게 한다.
    """

    payload = {
        "schema_version": int(config.train.query_cache_schema_version),
        "ray_sampler": config.ray,
        "camera": config.camera,
        "bev": config.bev,
        "orientations_degrees": config.train.nyu_virtual_orientations_degrees,
        "seed": int(config.train.seed),
    }
    encoded = json.dumps(_canonical_jsonable(payload), ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def query_sidecar_root(config: PipelineConfig) -> Path:
    """현재 sampler hash에 대응하는 query sidecar root를 반환한다."""

    return config.paths.outputs / "cache" / "nyu_queries" / query_sampler_config_hash(config)


def _query_set_to_mapping(query_set: Any) -> dict[str, np.ndarray]:
    """Mapping 또는 dataclass query set을 저장 가능한 array dict로 바꾼다."""

    if hasattr(query_set, "queries"):
        query_set = query_set.queries
    if isinstance(query_set, Mapping):
        items = query_set.items()
    elif is_dataclass(query_set):
        items = ((name, getattr(query_set, name)) for name in query_set.__dataclass_fields__)
    else:
        names = (
            "ray_dir",
            "source_uv",
            "parent_cell",
            "relative_uv",
            "angular_gap_before",
            "surface_gap_before_m",
            "bev_gap_before_cells",
            "sampling_score",
            "subdivision_u",
            "subdivision_v",
            "sampling_features",
            "observed",
            "added",
            "unknown",
        )
        items = ((name, getattr(query_set, name)) for name in names if hasattr(query_set, name))
    payload = {str(name): np.asarray(value) for name, value in items}
    if "ray_dir" not in payload:
        raise ValueError("Query builder result must contain ray_dir.")
    ray_dir = payload["ray_dir"]
    if ray_dir.ndim != 2 or ray_dir.shape[1] != 3:
        raise ValueError(f"query ray_dir must be (Q,3), got {ray_dir.shape}")
    count = len(ray_dir)
    for name, value in payload.items():
        if value.ndim > 0 and len(value) != count:
            raise ValueError(f"Query field {name!r} first dimension {len(value)} does not match Q={count}.")
    return payload


def _uniform_subpixel_queries(sample: Mapping[str, np.ndarray], budget: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """GT가 있는 source cell에서 결정적 seed의 균일 subpixel query를 만든다."""

    from wide_fov_supervision_v2.modules.adaptive_ray import spherical_bilerp

    rays = np.asarray(sample["source_rays"], dtype=np.float32)
    valid = (
        np.asarray(sample["source_observed"], dtype=bool)
        & np.asarray(sample["teacher_valid"], dtype=bool)
        & np.asarray(sample["gt_valid"], dtype=bool)
    )
    cell_valid = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    cells = np.column_stack(np.nonzero(cell_valid)).astype(np.int32)
    if budget <= 0 or len(cells) == 0:
        count = 0
        selected = np.zeros((0, 2), dtype=np.int32)
        relative_uv = np.zeros((0, 2), dtype=np.float32)
    else:
        chosen = rng.choice(len(cells), size=int(budget), replace=len(cells) < int(budget))
        selected = cells[chosen]
        relative_uv = rng.random((int(budget), 2), dtype=np.float32)
        count = int(budget)

    if count:
        y = selected[:, 0]
        x = selected[:, 1]
        rel_u = relative_uv[:, 0]
        rel_v = relative_uv[:, 1]
        ray_dir = spherical_bilerp(rays[y, x], rays[y, x + 1], rays[y + 1, x], rays[y + 1, x + 1], rel_u, rel_v)
        source_uv = np.column_stack([x.astype(np.float32) + rel_u + 0.5, y.astype(np.float32) + rel_v + 0.5])
    else:
        ray_dir = np.zeros((0, 3), dtype=np.float32)
        source_uv = np.zeros((0, 2), dtype=np.float32)

    return {
        "ray_dir": ray_dir.astype(np.float32),
        "source_uv": source_uv.astype(np.float32),
        "parent_cell": selected.astype(np.int32),
        "relative_uv": relative_uv.astype(np.float32),
        "angular_gap_before": np.zeros(count, dtype=np.float32),
        "surface_gap_before_m": np.zeros(count, dtype=np.float32),
        "bev_gap_before_cells": np.zeros(count, dtype=np.float32),
        "sampling_score": np.zeros(count, dtype=np.float32),
        "subdivision_u": np.ones(count, dtype=np.int16),
        "subdivision_v": np.ones(count, dtype=np.int16),
        "sampling_features": np.zeros((count, 3), dtype=np.float32),
        "observed": np.ones(count, dtype=bool),
        "added": np.ones(count, dtype=bool),
        "unknown": np.zeros(count, dtype=bool),
        "guided": np.zeros(count, dtype=bool),
        "uniform": np.ones(count, dtype=bool),
    }


def _concatenate_queries(guided: dict[str, np.ndarray], uniform: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """guided/uniform payload를 같은 schema로 맞춰 concatenate한다."""

    guided_count = len(guided["ray_dir"])
    uniform_count = len(uniform["ray_dir"])
    guided = dict(guided)
    guided["guided"] = np.ones(guided_count, dtype=bool)
    guided["uniform"] = np.zeros(guided_count, dtype=bool)
    all_names = set(guided) | set(uniform)
    merged: dict[str, np.ndarray] = {}
    for name in sorted(all_names):
        left = guided.get(name)
        right = uniform.get(name)
        if left is None and right is not None:
            left = np.zeros((guided_count,) + right.shape[1:], dtype=right.dtype)
        if right is None and left is not None:
            right = np.zeros((uniform_count,) + left.shape[1:], dtype=left.dtype)
        assert left is not None and right is not None
        if left.shape[1:] != right.shape[1:]:
            raise ValueError(f"Cannot merge query field {name!r}: {left.shape} vs {right.shape}")
        merged[name] = np.concatenate([left, right], axis=0)
    return merged


def default_nyu_query_builder(sample: Mapping[str, np.ndarray], config: PipelineConfig, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """3D-guided 75%와 균일 subpixel 25%를 만드는 기본 builder.

    guided sampler import는 함수 호출 시점까지 미룬다. 따라서 teacher cache 및 이
    모듈의 hash/GT 유틸리티는 sampler 구현이 교체되어도 독립적으로 사용할 수 있다.
    """

    from wide_fov_supervision_v2.modules.adaptive_ray import generate_guided_observed_queries

    total_budget = int(config.ray.max_added_queries_train)
    guided_budget = int(round(total_budget * float(config.ray.guided_train_fraction)))
    source_valid = (
        np.asarray(sample["source_observed"], dtype=bool)
        & np.asarray(sample["teacher_valid"], dtype=bool)
        & np.isfinite(np.asarray(sample["depth0_z"], dtype=np.float32))
        & (np.asarray(sample["depth0_z"], dtype=np.float32) > 0.0)
    )
    result = generate_guided_observed_queries(
        np.asarray(sample["source_rays"], dtype=np.float32),
        source_valid,
        np.asarray(sample["depth0_z"], dtype=np.float32),
        config.camera,
        config.bev,
        config.ray,
        mode="surface",
    )
    guided = _query_set_to_mapping(result.queries)
    if len(guided["ray_dir"]) > guided_budget:
        # sampler 자체도 같은 예산을 적용하지만 custom 구현에 대비해 결정적으로 자른다.
        guided = {name: values[:guided_budget] for name, values in guided.items()}
    # 회전된 NYU view에서 guided 후보가 부족하더라도 batch마다 총 query 예산은
    # 유지한다. guided는 최대 75%이고 남는 자리는 deterministic uniform query로 채운다.
    uniform_budget = max(0, total_budget - len(guided["ray_dir"]))
    uniform = _uniform_subpixel_queries(sample, uniform_budget, rng)
    return _concatenate_queries(guided, uniform)


def _stencil_rays(ray_dir: np.ndarray, step_rad: float) -> np.ndarray:
    """기존 differentiable geometry와 동일한 left/right/up/down stencil을 만든다."""

    import torch

    from wide_fov_supervision_v2.modules.query_geometry import query_stencil_rays

    rays = torch.from_numpy(np.asarray(ray_dir, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        return query_stencil_rays(rays, float(step_rad))[0].cpu().numpy().astype(np.float32)


def attach_exact_nyu_query_gt(
    frame: NYUFrame,
    orientation: VirtualFisheyeOrientation,
    query_payload: Mapping[str, np.ndarray],
    config: PipelineConfig,
) -> dict[str, np.ndarray]:
    """query와 네 stencil ray에 정확한 target-fisheye z-depth GT를 붙인다."""

    payload = {name: np.asarray(value) for name, value in query_payload.items()}
    query_rays = np.asarray(payload["ray_dir"], dtype=np.float32)
    center = sample_nyu_depth_at_fisheye_rays(
        frame.depth_z,
        query_rays,
        orientation,
        geometry_z_eps=config.camera.geometry_z_eps,
    )
    if "stencil_ray_dir" in payload:
        stencil_rays = np.asarray(payload["stencil_ray_dir"], dtype=np.float32)
    else:
        stencil_rays = _stencil_rays(query_rays, config.ray.stencil_step_rad)
    if stencil_rays.shape != (len(query_rays), 4, 3):
        raise ValueError(f"stencil_ray_dir must be (Q,4,3), got {stencil_rays.shape}")
    stencil = sample_nyu_depth_at_fisheye_rays(
        frame.depth_z,
        stencil_rays,
        orientation,
        geometry_z_eps=config.camera.geometry_z_eps,
    )
    payload.update(
        {
            "query_depth_gt_z": center.depth_z.astype(np.float32),
            "query_source_observed": center.source_observed.astype(bool),
            "query_gt_valid": center.gt_valid.astype(bool),
            "stencil_ray_dir": stencil_rays,
            "stencil_depth_gt_z": stencil.depth_z.astype(np.float32),
            "stencil_source_observed": stencil.source_observed.astype(bool),
            "stencil_gt_valid": stencil.gt_valid.astype(bool),
        }
    )
    return payload


def build_nyu_query_sidecar_cache(
    config: PipelineConfig,
    split: str = "train",
    *,
    query_builder: QueryBuilder | None = None,
) -> Path:
    """teacher cache와 분리된 sampler-hash query sidecar를 생성한다.

    custom sampler 실험은 ``query_builder(sample, config, rng)``를 전달하면 된다.
    반환값은 최소 ``ray_dir``을 가진 Mapping 또는 dataclass여야 한다. exact center 및
    stencil GT는 builder가 근사하지 않고 이 함수가 원본 NYU depth에서 다시 계산한다.
    """

    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    if not config.train.enable_query_sidecar_cache:
        return query_sidecar_root(config)
    builder = default_nyu_query_builder if query_builder is None else query_builder
    sampler_hash = query_sampler_config_hash(config)
    out_root = query_sidecar_root(config)
    out_dir = out_root / split
    out_dir.mkdir(parents=True, exist_ok=True)
    teacher_dir = config.paths.outputs / "cache" / "nyu" / split
    max_items = config.train.max_train_items if split == "train" else config.train.max_eval_items
    dataset = NYURawDataset(config.paths, split=split, max_items=max_items)

    for frame in tqdm(dataset, desc=f"cache nyu queries {split}"):
        teacher_path = teacher_dir / f"{frame.index:06d}.npz"
        if not teacher_path.exists():
            raise FileNotFoundError(f"Teacher cache must be built before query sidecar: {teacher_path}")
        orientation_index, orientation = orientation_for_frame(frame.index, config.train.nyu_virtual_orientations_degrees)
        sidecar_path = out_dir / f"{frame.index:06d}.npz"
        if sidecar_path.exists():
            try:
                with np.load(sidecar_path) as existing:
                    if (
                        int(existing["query_cache_schema_version"]) == int(config.train.query_cache_schema_version)
                        and str(existing["sampler_config_hash"]) == sampler_hash
                        and int(existing["orientation_index"]) == orientation_index
                    ):
                        continue
            except (OSError, ValueError, KeyError):
                pass

        with np.load(teacher_path) as teacher:
            sample = {name: teacher[name].copy() for name in teacher.files}
        cached_rotation = np.asarray(sample.get("rotation_nyu_from_fisheye"), dtype=np.float32)
        if cached_rotation.shape != (3, 3) or not np.allclose(cached_rotation, orientation.rotation_nyu_from_fisheye, atol=1.0e-6):
            raise RuntimeError(f"Teacher cache orientation does not match current config: {teacher_path}")

        rng = np.random.default_rng(np.random.SeedSequence([int(config.train.seed), int(frame.index)]))
        query_payload = _query_set_to_mapping(builder(sample, config, rng))
        query_payload = attach_exact_nyu_query_gt(frame, orientation, query_payload, config)
        query_payload.update(
            {
                "index": np.array(frame.index, dtype=np.int32),
                "orientation_index": np.array(orientation_index, dtype=np.int16),
                "query_cache_schema_version": np.array(config.train.query_cache_schema_version, dtype=np.int16),
                "sampler_config_hash": np.array(sampler_hash),
            }
        )
        np.savez_compressed(sidecar_path, **query_payload)
    return out_root
