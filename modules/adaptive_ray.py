from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

from wide_fov_supervision_v2.config import BevConfig, FisheyeCameraConfig, RaySamplerConfig
from wide_fov_supervision_v2.modules.camera_geometry import (
    angular_distance,
    camera_to_world_points,
    cell_angular_gap,
    central_median_gap,
    points_from_z_depth,
    project_fisheye_rays,
)


SamplingMode = Literal["surface", "surface_bev", "angular"]


@dataclass
class RayQuerySet:
    """임의 개수 query ray와 sampler 특징을 담는 numpy container.

    Shape 규칙:
        ray_dir: ``(Q, 3)`` OpenCV camera-frame 단위 ray.
        source_uv: ``(Q, 2)`` source 영상의 pixel-center 좌표계에서 bilinear
            sampling할 위치.
        parent_cell: ``(Q, 2)`` ``(cell_y, cell_x)``. source cell에 속하지 않는
            front-hemisphere query는 ``(-1, -1)``이다.
        relative_uv: ``(Q, 2)`` parent cell 안의 상대 좌표 ``(rel_u, rel_v)``.
        *_gap_before: query를 만든 cell에서 측정한 angular/3D/BEV 최대 간격.
        sampling_score: 해당 cell의 subdivision 우선순위.
        subdivision_u/v: 수평/수직 방향의 독립적인 subdivision 수.
        sampling_features: ``(Q, 3)`` 순서로 angular, surface, BEV gap ratio.
        observed/added/unknown: ``(Q,)`` bool. unknown은 loss/3D/BEV에서 제외한다.
    """

    ray_dir: np.ndarray
    source_uv: np.ndarray
    parent_cell: np.ndarray
    relative_uv: np.ndarray
    angular_gap_before: np.ndarray
    surface_gap_before_m: np.ndarray
    bev_gap_before_cells: np.ndarray
    sampling_score: np.ndarray
    subdivision_u: np.ndarray
    subdivision_v: np.ndarray
    sampling_features: np.ndarray
    observed: np.ndarray
    added: np.ndarray
    unknown: np.ndarray

    def __len__(self) -> int:
        return int(self.ray_dir.shape[0])

    @property
    def subdivision(self) -> np.ndarray:
        """구형 artifact와 호출부를 위한 최대 subdivision 호환 값이다."""

        return np.maximum(self.subdivision_u, self.subdivision_v)

    def save_npz(self, path, **extra_arrays) -> None:
        """query와 후처리 결과를 ``ray_queries.npz`` 형식으로 저장한다."""

        payload = {
            "ray_dir": self.ray_dir,
            "source_uv": self.source_uv,
            "parent_cell": self.parent_cell,
            "relative_uv": self.relative_uv,
            "angular_gap_before": self.angular_gap_before,
            "surface_gap_before_m": self.surface_gap_before_m,
            "bev_gap_before_cells": self.bev_gap_before_cells,
            "sampling_score": self.sampling_score,
            "subdivision_u": self.subdivision_u,
            "subdivision_v": self.subdivision_v,
            "subdivision": self.subdivision,
            "sampling_features": self.sampling_features,
            "observed": self.observed,
            "added": self.added,
            "unknown": self.unknown,
        }
        payload.update(extra_arrays)
        np.savez_compressed(path, **payload)

    def subset(self, mask: np.ndarray) -> "RayQuerySet":
        """bool 또는 integer mask로 query subset을 만든다."""

        return RayQuerySet(
            ray_dir=self.ray_dir[mask],
            source_uv=self.source_uv[mask],
            parent_cell=self.parent_cell[mask],
            relative_uv=self.relative_uv[mask],
            angular_gap_before=self.angular_gap_before[mask],
            surface_gap_before_m=self.surface_gap_before_m[mask],
            bev_gap_before_cells=self.bev_gap_before_cells[mask],
            sampling_score=self.sampling_score[mask],
            subdivision_u=self.subdivision_u[mask],
            subdivision_v=self.subdivision_v[mask],
            sampling_features=self.sampling_features[mask],
            observed=self.observed[mask],
            added=self.added[mask],
            unknown=self.unknown[mask],
        )


@dataclass(frozen=True)
class AdaptiveRayResult:
    """3D·BEV guided sampler의 query와 cell 단위 진단 결과.

    ``*_before`` map은 source cell의 실제 간격이고, ``*_planned_after`` map은
    anisotropic subdivision을 모두 적용했을 때의 예상 최대 간격이다. query 예산으로
    일부 cell만 선택되더라도 planned map은 sampler 자체의 subdivision 계획을 보존한다.
    invalid cell은 gap/priority map에서 ``NaN``, subdivision map에서 0이다.
    """

    queries: RayQuerySet
    angular_gap_before: np.ndarray
    angular_gap_planned_after: np.ndarray
    surface_gap_u_m: np.ndarray
    surface_gap_v_m: np.ndarray
    surface_gap_before_m: np.ndarray
    surface_gap_planned_after_m: np.ndarray
    bev_gap_u_cells: np.ndarray
    bev_gap_v_cells: np.ndarray
    bev_gap_before_cells: np.ndarray
    bev_gap_planned_after_cells: np.ndarray
    sampling_priority: np.ndarray
    eligible_mask: np.ndarray
    subdivision_u: np.ndarray
    subdivision_v: np.ndarray
    added_density: np.ndarray
    target_angular_gap_rad: float
    query_budget: int
    candidate_query_count: int
    budget_truncated: bool


def _empty_query_set() -> RayQuerySet:
    return RayQuerySet(
        ray_dir=np.zeros((0, 3), dtype=np.float32),
        source_uv=np.zeros((0, 2), dtype=np.float32),
        parent_cell=np.zeros((0, 2), dtype=np.int32),
        relative_uv=np.zeros((0, 2), dtype=np.float32),
        angular_gap_before=np.zeros((0,), dtype=np.float32),
        surface_gap_before_m=np.zeros((0,), dtype=np.float32),
        bev_gap_before_cells=np.zeros((0,), dtype=np.float32),
        sampling_score=np.zeros((0,), dtype=np.float32),
        subdivision_u=np.zeros((0,), dtype=np.int16),
        subdivision_v=np.zeros((0,), dtype=np.int16),
        sampling_features=np.zeros((0, 3), dtype=np.float32),
        observed=np.zeros((0,), dtype=bool),
        added=np.zeros((0,), dtype=bool),
        unknown=np.zeros((0,), dtype=bool),
    )


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(norm, 1.0e-12, None)


def slerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    """두 단위 ray 사이 spherical interpolation을 계산한다."""

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)[..., None]
    dot = np.clip(np.sum(a * b, axis=-1, keepdims=True), -1.0, 1.0)
    omega = np.arccos(dot)
    sin_omega = np.sin(omega)
    linear = (1.0 - t) * a + t * b
    spherical = (
        np.sin((1.0 - t) * omega) / np.clip(sin_omega, 1.0e-8, None) * a
        + np.sin(t * omega) / np.clip(sin_omega, 1.0e-8, None) * b
    )
    use_linear = np.abs(sin_omega) < 1.0e-6
    return _normalize(np.where(use_linear, linear, spherical))


def spherical_bilerp(
    r00: np.ndarray,
    r10: np.ndarray,
    r01: np.ndarray,
    r11: np.ndarray,
    rel_u: np.ndarray,
    rel_v: np.ndarray,
) -> np.ndarray:
    """네 corner ray 사이를 spherical interpolation으로 보간한다."""

    top = slerp(r00, r10, rel_u)
    bottom = slerp(r01, r11, rel_u)
    return slerp(top, bottom, rel_v).astype(np.float32)


def compute_target_gap(
    gap_before: np.ndarray,
    cell_valid: np.ndarray,
    config: RaySamplerConfig,
) -> float:
    """config와 중앙 median 기준으로 진단용 목표 angular gap(rad)을 결정한다."""

    if config.target_gap_rad is not None:
        return float(config.target_gap_rad)
    try:
        base = central_median_gap(gap_before, cell_valid, config.central_fraction)
    except RuntimeError:
        samples = gap_before[cell_valid & np.isfinite(gap_before)]
        if len(samples) == 0:
            raise RuntimeError("Cannot compute target angular gap; valid cells are empty.")
        base = float(np.median(samples))
    return base * float(config.target_gap_multiplier)


def _directional_angular_gap(rays_cv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """cell의 u/v 방향 두 edge 중 큰 angular gap을 각각 반환한다."""

    r00, r10 = rays_cv[:-1, :-1], rays_cv[:-1, 1:]
    r01, r11 = rays_cv[1:, :-1], rays_cv[1:, 1:]
    gap_u = np.maximum(angular_distance(r00, r10), angular_distance(r01, r11))
    gap_v = np.maximum(angular_distance(r00, r01), angular_distance(r10, r11))
    return gap_u.astype(np.float32), gap_v.astype(np.float32)


def _directional_point_gap(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """3D 또는 2D point grid의 cell별 u/v 최대 edge 길이를 반환한다."""

    p00, p10 = points[:-1, :-1], points[:-1, 1:]
    p01, p11 = points[1:, :-1], points[1:, 1:]
    gap_u = np.maximum(np.linalg.norm(p10 - p00, axis=-1), np.linalg.norm(p11 - p01, axis=-1))
    gap_v = np.maximum(np.linalg.norm(p01 - p00, axis=-1), np.linalg.norm(p11 - p10, axis=-1))
    return gap_u.astype(np.float32), gap_v.astype(np.float32)


def _query_budget(config: RaySamplerConfig, mode: SamplingMode) -> int:
    """학습(surface)과 inference(surface_bev/angular)의 추가 query 예산을 고른다."""

    if mode == "surface":
        budget = int(round(config.max_added_queries_train * config.guided_train_fraction))
    else:
        budget = int(config.max_added_queries_inference)
    return max(0, budget)


def _empty_guided_result(
    shape: tuple[int, int],
    *,
    angular_gap: np.ndarray,
    target_angular_gap: float,
    budget: int,
) -> AdaptiveRayResult:
    nan_map = np.full(shape, np.nan, dtype=np.float32)
    zero_i16 = np.zeros(shape, dtype=np.int16)
    return AdaptiveRayResult(
        queries=_empty_query_set(),
        angular_gap_before=angular_gap.astype(np.float32),
        angular_gap_planned_after=angular_gap.astype(np.float32),
        surface_gap_u_m=nan_map.copy(),
        surface_gap_v_m=nan_map.copy(),
        surface_gap_before_m=nan_map.copy(),
        surface_gap_planned_after_m=nan_map.copy(),
        bev_gap_u_cells=nan_map.copy(),
        bev_gap_v_cells=nan_map.copy(),
        bev_gap_before_cells=nan_map.copy(),
        bev_gap_planned_after_cells=nan_map.copy(),
        sampling_priority=nan_map.copy(),
        eligible_mask=np.zeros(shape, dtype=bool),
        subdivision_u=zero_i16.copy(),
        subdivision_v=zero_i16.copy(),
        added_density=np.zeros(shape, dtype=np.float32),
        target_angular_gap_rad=float(target_angular_gap),
        query_budget=budget,
        candidate_query_count=0,
        budget_truncated=False,
    )


def generate_source_queries(rays_cv: np.ndarray, valid: np.ndarray) -> RayQuerySet:
    """원본 observed source pixel을 확장된 query schema로 만든다."""

    ys, xs = np.nonzero(valid)
    if len(xs) == 0:
        return _empty_query_set()
    count = len(xs)
    uv = np.column_stack([xs.astype(np.float32) + 0.5, ys.astype(np.float32) + 0.5])
    return RayQuerySet(
        ray_dir=rays_cv[ys, xs].astype(np.float32),
        source_uv=uv.astype(np.float32),
        parent_cell=np.column_stack([ys, xs]).astype(np.int32),
        relative_uv=np.zeros((count, 2), dtype=np.float32),
        angular_gap_before=np.zeros(count, dtype=np.float32),
        surface_gap_before_m=np.zeros(count, dtype=np.float32),
        bev_gap_before_cells=np.zeros(count, dtype=np.float32),
        sampling_score=np.zeros(count, dtype=np.float32),
        subdivision_u=np.ones(count, dtype=np.int16),
        subdivision_v=np.ones(count, dtype=np.int16),
        sampling_features=np.zeros((count, 3), dtype=np.float32),
        observed=np.ones(count, dtype=bool),
        added=np.zeros(count, dtype=bool),
        unknown=np.zeros(count, dtype=bool),
    )


def generate_guided_observed_queries(
    rays_cv: np.ndarray,
    source_valid: np.ndarray,
    depth0_z: np.ndarray,
    camera: FisheyeCameraConfig,
    bev_config: BevConfig,
    sampler_config: RaySamplerConfig,
    mode: SamplingMode,
) -> AdaptiveRayResult:
    """3D surface와 BEV 희소도를 기준으로 추가 observed query ray를 생성한다.

    Args:
        rays_cv: ``(H,W,3)`` OpenCV camera-frame source 단위 ray.
        source_valid: ``(H,W)`` lens/source correspondence 유효 mask.
        depth0_z: ``(H,W)`` DA-V2 teacher의 camera +z 방향 metric depth.
        camera: z-depth 복원과 camera-to-world 변환에 필요한 카메라 설정.
        bev_config: ``surface_bev`` 모드의 world-XY 범위와 BEV cell 크기.
        sampler_config: 목표 간격, discontinuity threshold, subdivision 및 예산.
        mode: ``surface``는 NYU 학습용 3D 기준, ``surface_bev``는 Isaac inference용
            3D·BEV 혼합 기준, ``angular``는 비교용 기존 ray-angle 기준이다.

    ``surface``와 ``surface_bev``에서는 네 corner depth의 최대 log 차이가 threshold를
    넘는 cell을 제외한다. 이는 물체 경계를 가로질러 존재하지 않는 중간 3D surface를
    만드는 것을 방지한다. ray 위치는 학습하지 않고 spherical bilerp로 결정한다.
    """

    if mode not in ("surface", "surface_bev", "angular"):
        raise ValueError(f"Unsupported adaptive sampling mode: {mode}")
    rays = np.asarray(rays_cv, dtype=np.float32)
    valid = np.asarray(source_valid, dtype=bool)
    depth = np.asarray(depth0_z, dtype=np.float32)
    if rays.ndim != 3 or rays.shape[-1] != 3:
        raise ValueError(f"rays_cv must have shape (H,W,3), got {rays.shape}")
    if valid.shape != rays.shape[:2] or depth.shape != rays.shape[:2]:
        raise ValueError("source_valid and depth0_z must match rays_cv spatial shape")
    if rays.shape[0] < 2 or rays.shape[1] < 2:
        raise ValueError("At least a 2x2 ray grid is required")

    angular_gap, angular_cell_valid = cell_angular_gap(rays, valid)
    angular_u, angular_v = _directional_angular_gap(rays)
    if np.any(angular_cell_valid):
        target_angular = compute_target_gap(angular_gap, angular_cell_valid, sampler_config)
    else:
        target_angular = float(sampler_config.target_gap_rad or 1.0e-3)
    budget = _query_budget(sampler_config, mode)
    cell_shape = angular_gap.shape
    if not np.any(angular_cell_valid):
        return _empty_guided_result(
            cell_shape,
            angular_gap=angular_gap,
            target_angular_gap=target_angular,
            budget=budget,
        )

    points_cv, _, point_valid = points_from_z_depth(depth, rays, z_eps=camera.geometry_z_eps)
    point_valid &= valid
    p00_valid, p10_valid = point_valid[:-1, :-1], point_valid[:-1, 1:]
    p01_valid, p11_valid = point_valid[1:, :-1], point_valid[1:, 1:]
    depth_cell_valid = p00_valid & p10_valid & p01_valid & p11_valid

    safe_log_depth = np.full(depth.shape, np.nan, dtype=np.float32)
    safe_log_depth[point_valid] = np.log(depth[point_valid])
    l00, l10 = safe_log_depth[:-1, :-1], safe_log_depth[:-1, 1:]
    l01, l11 = safe_log_depth[1:, :-1], safe_log_depth[1:, 1:]
    log_min = np.minimum(np.minimum(l00, l10), np.minimum(l01, l11))
    log_max = np.maximum(np.maximum(l00, l10), np.maximum(l01, l11))
    depth_continuous = depth_cell_valid & (
        (log_max - log_min) <= float(sampler_config.depth_discontinuity_log_threshold)
    )

    surface_u, surface_v = _directional_point_gap(points_cv)
    surface_before = np.maximum(surface_u, surface_v).astype(np.float32)

    points_world = camera_to_world_points(points_cv, camera).astype(np.float32)
    bev_u_m, bev_v_m = _directional_point_gap(points_world[..., :2])
    meters_per_pixel = max(float(bev_config.meters_per_pixel), 1.0e-8)
    bev_u_cells = (bev_u_m / meters_per_pixel).astype(np.float32)
    bev_v_cells = (bev_v_m / meters_per_pixel).astype(np.float32)
    bev_before_cells = np.maximum(bev_u_cells, bev_v_cells).astype(np.float32)

    half = float(bev_config.size_m) * 0.5
    center_x, center_y = bev_config.center_xy
    world_point_in_bev = (
        np.isfinite(points_world[..., :2]).all(axis=-1)
        & (points_world[..., 0] >= center_x - half)
        & (points_world[..., 0] < center_x + half)
        & (points_world[..., 1] >= center_y - half)
        & (points_world[..., 1] < center_y + half)
    )
    bev_cell_valid = (
        world_point_in_bev[:-1, :-1]
        & world_point_in_bev[:-1, 1:]
        & world_point_in_bev[1:, :-1]
        & world_point_in_bev[1:, 1:]
    )

    angular_ratio_u = angular_u / max(float(target_angular), 1.0e-8)
    angular_ratio_v = angular_v / max(float(target_angular), 1.0e-8)
    surface_ratio_u = surface_u / max(float(sampler_config.target_surface_gap_m), 1.0e-8)
    surface_ratio_v = surface_v / max(float(sampler_config.target_surface_gap_m), 1.0e-8)
    bev_ratio_u = bev_u_cells / max(float(sampler_config.target_bev_gap_cells), 1.0e-8)
    bev_ratio_v = bev_v_cells / max(float(sampler_config.target_bev_gap_cells), 1.0e-8)

    if mode == "angular":
        eligible = angular_cell_valid
        ratio_u, ratio_v = angular_ratio_u, angular_ratio_v
    elif mode == "surface":
        eligible = angular_cell_valid & depth_continuous
        ratio_u, ratio_v = surface_ratio_u, surface_ratio_v
    else:
        eligible = angular_cell_valid & depth_continuous & bev_cell_valid
        ratio_u = np.maximum(surface_ratio_u, bev_ratio_u)
        ratio_v = np.maximum(surface_ratio_v, bev_ratio_v)

    finite_ratio = np.isfinite(ratio_u) & np.isfinite(ratio_v)
    eligible &= finite_ratio
    subdivision_u = np.zeros(cell_shape, dtype=np.int16)
    subdivision_v = np.zeros(cell_shape, dtype=np.int16)
    clipped_u = np.clip(np.ceil(np.where(eligible, ratio_u, 1.0)), 1, sampler_config.max_subdivision)
    clipped_v = np.clip(np.ceil(np.where(eligible, ratio_v, 1.0)), 1, sampler_config.max_subdivision)
    subdivision_u[eligible] = clipped_u[eligible].astype(np.int16)
    subdivision_v[eligible] = clipped_v[eligible].astype(np.int16)

    score = np.maximum(ratio_u, ratio_v).astype(np.float32)
    priority = np.full(cell_shape, np.nan, dtype=np.float32)
    priority[eligible] = score[eligible]
    add_mask = (
        eligible
        & (score > float(sampler_config.min_gap_multiplier_to_add))
        & ((subdivision_u > 1) | (subdivision_v > 1))
    )

    safe_sub_u = np.maximum(subdivision_u, 1)
    safe_sub_v = np.maximum(subdivision_v, 1)
    angular_after = np.full(cell_shape, np.nan, dtype=np.float32)
    surface_after = np.full(cell_shape, np.nan, dtype=np.float32)
    bev_after = np.full(cell_shape, np.nan, dtype=np.float32)
    angular_after[eligible] = np.maximum(
        angular_u[eligible] / safe_sub_u[eligible], angular_v[eligible] / safe_sub_v[eligible]
    )
    surface_after[eligible] = np.maximum(
        surface_u[eligible] / safe_sub_u[eligible], surface_v[eligible] / safe_sub_v[eligible]
    )
    bev_after[eligible] = np.maximum(
        bev_u_cells[eligible] / safe_sub_u[eligible], bev_v_cells[eligible] / safe_sub_v[eligible]
    )

    ys, xs = np.nonzero(add_mask)
    per_cell_candidates = (
        (subdivision_u[ys, xs].astype(np.int64) + 1)
        * (subdivision_v[ys, xs].astype(np.int64) + 1)
        - 4
    )
    candidate_count = int(per_cell_candidates.sum(dtype=np.int64))
    if len(ys) == 0 or budget == 0:
        queries = _empty_query_set()
        added_density = np.zeros(cell_shape, dtype=np.float32)
    else:
        # score 내림차순, 같은 score에서는 y/x 오름차순으로 항상 같은 query를 고른다.
        order = np.lexsort((xs, ys, -score[ys, xs]))
        scale = 10 ** int(sampler_config.dedupe_uv_decimals)
        seen_uv: set[tuple[int, int]] = set()
        records: list[tuple] = []
        stop = False
        for index in order.tolist():
            y, x = int(ys[index]), int(xs[index])
            sub_u, sub_v = int(subdivision_u[y, x]), int(subdivision_v[y, x])
            r00, r10 = rays[y, x], rays[y, x + 1]
            r01, r11 = rays[y + 1, x], rays[y + 1, x + 1]
            features = tuple(
                float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))
                for value in (
                    max(angular_ratio_u[y, x], angular_ratio_v[y, x]),
                    max(surface_ratio_u[y, x], surface_ratio_v[y, x]),
                    max(bev_ratio_u[y, x], bev_ratio_v[y, x]),
                )
            )
            for iy in range(sub_v + 1):
                rel_v = iy / sub_v
                for ix in range(sub_u + 1):
                    rel_u = ix / sub_u
                    if ix in (0, sub_u) and iy in (0, sub_v):
                        continue
                    source_uv = np.array([x + rel_u + 0.5, y + rel_v + 0.5], dtype=np.float32)
                    key = tuple(np.rint(source_uv * scale).astype(np.int64).tolist())
                    if key in seen_uv:
                        continue
                    seen_uv.add(key)
                    ray = spherical_bilerp(r00, r10, r01, r11, np.array(rel_u), np.array(rel_v))
                    records.append(
                        (
                            ray.reshape(3),
                            source_uv,
                            (y, x),
                            (rel_u, rel_v),
                            float(angular_gap[y, x]),
                            float(surface_before[y, x]),
                            float(bev_before_cells[y, x]),
                            float(score[y, x]),
                            sub_u,
                            sub_v,
                            features,
                        )
                    )
                    if len(records) >= budget:
                        stop = True
                        break
                if stop:
                    break
            if stop:
                break

        if records:
            count = len(records)
            parent_cell = np.asarray([item[2] for item in records], dtype=np.int32)
            queries = RayQuerySet(
                ray_dir=np.stack([item[0] for item in records]).astype(np.float32),
                source_uv=np.stack([item[1] for item in records]).astype(np.float32),
                parent_cell=parent_cell,
                relative_uv=np.asarray([item[3] for item in records], dtype=np.float32),
                angular_gap_before=np.asarray([item[4] for item in records], dtype=np.float32),
                surface_gap_before_m=np.asarray([item[5] for item in records], dtype=np.float32),
                bev_gap_before_cells=np.asarray([item[6] for item in records], dtype=np.float32),
                sampling_score=np.asarray([item[7] for item in records], dtype=np.float32),
                subdivision_u=np.asarray([item[8] for item in records], dtype=np.int16),
                subdivision_v=np.asarray([item[9] for item in records], dtype=np.int16),
                sampling_features=np.asarray([item[10] for item in records], dtype=np.float32),
                observed=np.ones(count, dtype=bool),
                added=np.ones(count, dtype=bool),
                unknown=np.zeros(count, dtype=bool),
            )
            added_density = np.zeros(cell_shape, dtype=np.float32)
            np.add.at(added_density, (parent_cell[:, 0], parent_cell[:, 1]), 1.0)
        else:
            queries = _empty_query_set()
            added_density = np.zeros(cell_shape, dtype=np.float32)

    surface_u = np.where(depth_cell_valid, surface_u, np.nan).astype(np.float32)
    surface_v = np.where(depth_cell_valid, surface_v, np.nan).astype(np.float32)
    surface_before = np.where(depth_cell_valid, surface_before, np.nan).astype(np.float32)
    bev_u_cells = np.where(depth_cell_valid, bev_u_cells, np.nan).astype(np.float32)
    bev_v_cells = np.where(depth_cell_valid, bev_v_cells, np.nan).astype(np.float32)
    bev_before_cells = np.where(depth_cell_valid, bev_before_cells, np.nan).astype(np.float32)
    return AdaptiveRayResult(
        queries=queries,
        angular_gap_before=angular_gap.astype(np.float32),
        angular_gap_planned_after=angular_after,
        surface_gap_u_m=surface_u,
        surface_gap_v_m=surface_v,
        surface_gap_before_m=surface_before,
        surface_gap_planned_after_m=surface_after,
        bev_gap_u_cells=bev_u_cells,
        bev_gap_v_cells=bev_v_cells,
        bev_gap_before_cells=bev_before_cells,
        bev_gap_planned_after_cells=bev_after,
        sampling_priority=priority,
        eligible_mask=eligible,
        subdivision_u=subdivision_u,
        subdivision_v=subdivision_v,
        added_density=added_density,
        target_angular_gap_rad=float(target_angular),
        query_budget=budget,
        candidate_query_count=candidate_count,
        budget_truncated=(budget == 0 and candidate_count > 0)
        or (len(queries) >= budget and candidate_count > len(queries)),
    )


def generate_adaptive_observed_queries(
    rays_cv: np.ndarray,
    valid: np.ndarray,
    config: RaySamplerConfig,
) -> tuple[RayQuerySet, np.ndarray, np.ndarray, np.ndarray, float]:
    """기존 angular-gap baseline sampler를 호환용으로 유지한다.

    새 inference/training 코드는 ``generate_guided_observed_queries``를 사용해야 한다.
    이 함수는 이전 공개 반환 형식을 보존해 기존 진단과 회귀 테스트를 깨지 않는다.
    """

    gap_before, cell_valid = cell_angular_gap(rays_cv, valid)
    target_gap = compute_target_gap(gap_before, cell_valid, config)
    finite_gap = np.where(cell_valid & np.isfinite(gap_before), gap_before, 0.0)
    subdivision = np.ceil(finite_gap / max(target_gap, 1.0e-8)).astype(np.int16)
    subdivision = np.clip(subdivision, 1, int(config.max_subdivision))
    add_mask = cell_valid & (finite_gap > target_gap * config.min_gap_multiplier_to_add) & (subdivision > 1)

    records: list[tuple] = []
    ys, xs = np.nonzero(add_mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        sub = int(subdivision[y, x])
        for iy in range(sub + 1):
            rel_v = iy / sub
            for ix in range(sub + 1):
                rel_u = ix / sub
                if ix in (0, sub) and iy in (0, sub):
                    continue
                ray = spherical_bilerp(
                    rays_cv[y, x], rays_cv[y, x + 1], rays_cv[y + 1, x], rays_cv[y + 1, x + 1],
                    np.array(rel_u), np.array(rel_v),
                )
                records.append(
                    (ray.reshape(3), (x + rel_u + 0.5, y + rel_v + 0.5), (y, x), (rel_u, rel_v), float(gap_before[y, x]), sub)
                )
    if not records:
        gap_after = np.where(cell_valid, gap_before, np.nan).astype(np.float32)
        return _empty_query_set(), gap_before, gap_after, np.zeros_like(gap_before, dtype=np.float32), target_gap

    source_uv = np.asarray([item[1] for item in records], dtype=np.float32)
    rounded = np.round(source_uv, int(config.dedupe_uv_decimals))
    _, keep = np.unique(rounded, axis=0, return_index=True)
    keep = np.sort(keep)
    count = len(keep)
    parent_cell = np.asarray([records[index][2] for index in keep], dtype=np.int32)
    angular = np.asarray([records[index][4] for index in keep], dtype=np.float32)
    sub = np.asarray([records[index][5] for index in keep], dtype=np.int16)
    queries = RayQuerySet(
        ray_dir=np.stack([records[index][0] for index in keep]).astype(np.float32),
        source_uv=source_uv[keep],
        parent_cell=parent_cell,
        relative_uv=np.asarray([records[index][3] for index in keep], dtype=np.float32),
        angular_gap_before=angular,
        surface_gap_before_m=np.full(count, np.nan, dtype=np.float32),
        bev_gap_before_cells=np.full(count, np.nan, dtype=np.float32),
        sampling_score=angular / max(target_gap, 1.0e-8),
        subdivision_u=sub,
        subdivision_v=sub,
        sampling_features=np.column_stack(
            [angular / max(target_gap, 1.0e-8), np.zeros(count), np.zeros(count)]
        ).astype(np.float32),
        observed=np.ones(count, dtype=bool),
        added=np.ones(count, dtype=bool),
        unknown=np.zeros(count, dtype=bool),
    )
    density = np.zeros_like(gap_before, dtype=np.float32)
    np.add.at(density, (parent_cell[:, 0], parent_cell[:, 1]), 1.0)
    gap_after = np.where(cell_valid, gap_before / subdivision.clip(min=1), np.nan).astype(np.float32)
    return queries, gap_before, gap_after, density, target_gap


def generate_front_hemisphere_queries(
    camera: FisheyeCameraConfig,
    lens_valid: np.ndarray,
    target_gap_rad: float | None,
    config: RaySamplerConfig,
) -> tuple[RayQuerySet, np.ndarray]:
    """전방 180도 ray를 만들고 원본 대응 여부만 기록한다.

    ``target_gap_rad=None``이면 확정된 기본 간격 0.5도를 사용한다. unknown ray는
    coverage artifact용이며 Refiner, loss, 3D point, BEV에는 전달하지 않는다.
    """

    base_step = np.deg2rad(config.hemisphere_step_degrees) if target_gap_rad is None else target_gap_rad
    step = max(float(base_step) * float(config.hemisphere_gap_multiplier), 1.0e-3)
    rays = [np.array([0.0, 0.0, 1.0], dtype=np.float32)]
    for theta in np.arange(step, np.pi * 0.5, step, dtype=np.float64):
        ring_count = max(8, int(np.ceil(2.0 * np.pi * np.sin(theta) / step)))
        phi = np.linspace(0.0, 2.0 * np.pi, ring_count, endpoint=False)
        sin_theta = np.sin(theta)
        rays.extend(
            np.stack(
                [sin_theta * np.cos(phi), sin_theta * np.sin(phi), np.full_like(phi, np.cos(theta))],
                axis=-1,
            ).astype(np.float32)
        )
    ray_dir = _normalize(np.asarray(rays, dtype=np.float32)).astype(np.float32)
    uv, project_valid = project_fisheye_rays(ray_dir, camera)
    inside = (
        project_valid
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < camera.width)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < camera.height)
        & (ray_dir[:, 2] > camera.geometry_z_eps)
    )
    px = np.floor(uv[:, 0]).astype(np.int64, copy=False)
    py = np.floor(uv[:, 1]).astype(np.int64, copy=False)
    lens = np.zeros(len(ray_dir), dtype=bool)
    if np.any(inside):
        lens[inside] = lens_valid[py[inside], px[inside]]
    observed = inside & lens
    unknown = ~observed

    coverage = np.zeros((camera.height, camera.width), dtype=np.uint8)
    coverage[py[observed], px[observed]] = 1
    unknown_in_image = (
        unknown
        & project_valid
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < camera.width)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < camera.height)
    )
    coverage[py[unknown_in_image], px[unknown_in_image]] = 2
    source_uv = uv.astype(np.float32)
    source_uv[unknown] = np.nan
    count = len(ray_dir)
    queries = RayQuerySet(
        ray_dir=ray_dir,
        source_uv=source_uv,
        parent_cell=np.full((count, 2), -1, dtype=np.int32),
        relative_uv=np.full((count, 2), np.nan, dtype=np.float32),
        angular_gap_before=np.full(count, step, dtype=np.float32),
        surface_gap_before_m=np.full(count, np.nan, dtype=np.float32),
        bev_gap_before_cells=np.full(count, np.nan, dtype=np.float32),
        sampling_score=np.zeros(count, dtype=np.float32),
        subdivision_u=np.ones(count, dtype=np.int16),
        subdivision_v=np.ones(count, dtype=np.int16),
        sampling_features=np.zeros((count, 3), dtype=np.float32),
        observed=observed,
        added=np.zeros(count, dtype=bool),
        unknown=unknown,
    )
    return queries, coverage


def merge_query_sets(query_sets: Iterable[RayQuerySet], config: RaySamplerConfig) -> RayQuerySet:
    """여러 query set을 합치고 동일 ray 방향을 결정적으로 중복 제거한다."""

    sets = [query for query in query_sets if len(query) > 0]
    if not sets:
        return _empty_query_set()
    merged = RayQuerySet(
        ray_dir=np.concatenate([query.ray_dir for query in sets], axis=0),
        source_uv=np.concatenate([query.source_uv for query in sets], axis=0),
        parent_cell=np.concatenate([query.parent_cell for query in sets], axis=0),
        relative_uv=np.concatenate([query.relative_uv for query in sets], axis=0),
        angular_gap_before=np.concatenate([query.angular_gap_before for query in sets], axis=0),
        surface_gap_before_m=np.concatenate([query.surface_gap_before_m for query in sets], axis=0),
        bev_gap_before_cells=np.concatenate([query.bev_gap_before_cells for query in sets], axis=0),
        sampling_score=np.concatenate([query.sampling_score for query in sets], axis=0),
        subdivision_u=np.concatenate([query.subdivision_u for query in sets], axis=0),
        subdivision_v=np.concatenate([query.subdivision_v for query in sets], axis=0),
        sampling_features=np.concatenate([query.sampling_features for query in sets], axis=0),
        observed=np.concatenate([query.observed for query in sets], axis=0),
        added=np.concatenate([query.added for query in sets], axis=0),
        unknown=np.concatenate([query.unknown for query in sets], axis=0),
    )
    rounded_rays = np.round(merged.ray_dir, 6)
    _, keep = np.unique(rounded_rays, axis=0, return_index=True)
    keep = np.sort(keep)
    if config.max_queries_per_inference is not None and len(keep) > config.max_queries_per_inference:
        priority = merged.observed[keep].astype(np.int32) * 2 + merged.added[keep].astype(np.int32)
        order = np.argsort(-priority, kind="stable")
        keep = np.sort(keep[order[: int(config.max_queries_per_inference)]])
    return merged.subset(keep)
