from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from wide_fov_supervision_v2.backbone.runner import BackboneRunner
from wide_fov_supervision_v2.config import PipelineConfig
from wide_fov_supervision_v2.datasets.nyu.dataset import (
    NYURawDataset,
    VirtualFisheyeOrientation,
    orientation_for_frame,
    virtual_fisheye_from_nyu,
)
from wide_fov_supervision_v2.modules.camera_geometry import build_fisheye_rays


TEACHER_CACHE_SCHEMA_VERSION = 2


def _teacher_cache_is_current(
    path: Path,
    orientation_index: int,
    orientation: VirtualFisheyeOrientation,
    expected_ray_shape: tuple[int, int, int],
) -> bool:
    """기존 파일이 현재 mask/orientation schema인지 가볍게 확인한다."""

    if not path.exists():
        return False
    try:
        with np.load(path) as data:
            required = {"source_observed", "teacher_valid", "gt_valid", "orientation_index"}
            return bool(
                required.issubset(data.files)
                and int(data["cache_schema_version"]) == TEACHER_CACHE_SCHEMA_VERSION
                and int(data["orientation_index"]) == int(orientation_index)
                and data["source_rays"].shape == expected_ray_shape
                and np.allclose(
                    data["rotation_nyu_from_fisheye"],
                    orientation.rotation_nyu_from_fisheye,
                    atol=1.0e-6,
                )
            )
    except (OSError, ValueError, KeyError):
        return False


def build_nyu_teacher_cache(config: PipelineConfig, split: str = "train") -> Path:
    """orientation이 적용된 NYU RGB-D와 frozen teacher 결과를 cache한다.

    저장 위치는 ``outputs/cache/nyu/{split}/{index:06d}.npz``이다. teacher cache는
    adaptive sampler 설정과 무관하며, query는 별도의 sampler-hash sidecar에 둔다.

    mask 의미:
        ``source_observed``: NYU RGB가 가상 fisheye pixel에 투영되는지 여부.
        ``teacher_valid``: DA-V2 depth와 DSINE normal이 finite인지 여부.
        ``gt_valid``: 정확히 회전 변환된 target z-depth가 유효한지 여부.
        ``source_valid``: 구 trainer 호환용 ``teacher_valid & gt_valid`` alias.
    """

    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    cache_root = config.paths.outputs / "cache" / "nyu"
    out_dir = cache_root / split
    out_dir.mkdir(parents=True, exist_ok=True)
    max_items = config.train.max_train_items if split == "train" else config.train.max_eval_items
    dataset = NYURawDataset(config.paths, split=split, max_items=max_items)
    camera_rays = build_fisheye_rays(config.camera)
    rays = camera_rays.rays_cv
    runner = BackboneRunner(config.paths, config.backbone, config.camera)
    use_tangent = config.toggles.enable_tangent_backbone
    cache_float_dtype = np.float16 if config.train.cache_dtype == "float16" else np.float32

    for frame in tqdm(dataset, desc=f"cache nyu {split}"):
        orientation_index, orientation = orientation_for_frame(
            frame.index,
            config.train.nyu_virtual_orientations_degrees,
        )
        path = out_dir / f"{frame.index:06d}.npz"
        if _teacher_cache_is_current(path, orientation_index, orientation, rays.shape):
            continue

        warped = virtual_fisheye_from_nyu(frame, config.camera, rays, orientation)
        pred = runner.run_tangent(warped.rgb, rays) if use_tangent else runner.run_direct(warped.rgb)
        teacher_valid = (
            pred.valid.astype(bool)
            & np.isfinite(pred.depth0_z)
            & (pred.depth0_z > 0.0)
            & np.isfinite(pred.normal0).all(axis=-1)
        )
        gt_valid = warped.gt_valid.astype(bool)
        # 이전 trainer/evaluate가 새 cache를 읽을 수 있도록 유지하는 alias다.
        source_valid_compat = teacher_valid & gt_valid
        np.savez_compressed(
            path,
            rgb=warped.rgb,
            depth_gt_z=warped.depth_gt_z.astype(np.float32),
            depth0_z=pred.depth0_z.astype(cache_float_dtype),
            normal0=pred.normal0.astype(cache_float_dtype),
            source_rays=rays.astype(np.float32),
            source_observed=warped.source_observed.astype(bool),
            teacher_valid=teacher_valid.astype(bool),
            gt_valid=gt_valid,
            source_valid=source_valid_compat.astype(bool),
            lens_valid=camera_rays.valid.astype(bool),
            index=np.array(frame.index, dtype=np.int32),
            branch=np.array(pred.branch),
            cache_schema_version=np.array(TEACHER_CACHE_SCHEMA_VERSION, dtype=np.int16),
            orientation_index=np.array(orientation_index, dtype=np.int16),
            orientation_name=np.array(orientation.name),
            orientation_yaw_degrees=np.array(orientation.yaw_degrees, dtype=np.float32),
            orientation_pitch_degrees=np.array(orientation.pitch_degrees, dtype=np.float32),
            rotation_nyu_from_fisheye=orientation.rotation_nyu_from_fisheye.astype(np.float32),
        )
    return cache_root
