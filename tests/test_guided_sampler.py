from __future__ import annotations

import numpy as np

from wide_fov_supervision_v2.config import BevConfig, FisheyeCameraConfig, RaySamplerConfig
from wide_fov_supervision_v2.modules.adaptive_ray import generate_guided_observed_queries
from wide_fov_supervision_v2.modules.camera_geometry import build_fisheye_rays


def _identity_camera(width: int, height: int) -> FisheyeCameraConfig:
    """합성 ray가 사용하는 단순 camera-to-world 설정을 만든다."""

    return FisheyeCameraConfig(
        width=width,
        height=height,
        fx=20.0,
        fy=20.0,
        cx=width * 0.5,
        cy=height * 0.5,
        distortion=(0.0, 0.0, 0.0, 0.0),
        world_from_camera=((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
        camera_position_world=(0.0, 0.0, 0.0),
    )


def _anisotropic_rays(height: int, width: int, step_u: float, step_v: float) -> np.ndarray:
    """z=1 평면에서 u/v 간격이 다른 pinhole형 단위 ray grid를 만든다."""

    x = (np.arange(width, dtype=np.float32) - (width - 1) * 0.5) * step_u
    y = (np.arange(height, dtype=np.float32) - (height - 1) * 0.5) * step_v
    xx, yy = np.meshgrid(x, y)
    rays = np.stack([xx, yy, np.ones_like(xx)], axis=-1)
    return (rays / np.linalg.norm(rays, axis=-1, keepdims=True)).astype(np.float32)


def test_surface_sampler_concentrates_queries_at_wide_fov_edge() -> None:
    """평면의 3D sampling 간격이 커지는 외곽에 query가 집중되어야 한다."""

    camera = _identity_camera(65, 65)
    camera.fx = camera.fy = 24.0
    camera.cx = camera.cy = 32.5
    source = build_fisheye_rays(camera)
    depth = np.full((65, 65), 2.0, dtype=np.float32)
    config = RaySamplerConfig(
        target_surface_gap_m=0.08,
        max_added_queries_train=2_000,
        guided_train_fraction=1.0,
    )
    result = generate_guided_observed_queries(
        source.rays_cv,
        source.valid,
        depth,
        camera,
        BevConfig(center_xy=(0.0, 0.0), size_m=100.0),
        config,
        mode="surface",
    )

    density = result.added_density
    center_mean = float(density[20:44, 20:44].mean())
    outer = np.concatenate(
        [density[:12].ravel(), density[-12:].ravel(), density[:, :12].ravel(), density[:, -12:].ravel()]
    )
    assert len(result.queries) == result.query_budget == 2_000
    assert float(outer.mean()) > center_mean
    assert result.budget_truncated


def test_depth_discontinuity_cells_are_not_subdivided() -> None:
    """큰 depth jump를 가로지르는 cell에는 가짜 중간 surface query를 만들지 않는다."""

    height, width = 6, 7
    rays = _anisotropic_rays(height, width, step_u=0.08, step_v=0.04)
    valid = np.ones((height, width), dtype=bool)
    depth = np.ones((height, width), dtype=np.float32)
    depth[:, 3:] = 2.0
    result = generate_guided_observed_queries(
        rays,
        valid,
        depth,
        _identity_camera(width, height),
        BevConfig(center_xy=(0.0, 0.0), size_m=100.0),
        RaySamplerConfig(
            target_surface_gap_m=0.02,
            depth_discontinuity_log_threshold=0.20,
            max_added_queries_train=10_000,
            guided_train_fraction=1.0,
        ),
        mode="surface",
    )

    assert not np.any(result.eligible_mask[:, 2])
    assert np.all(result.subdivision_u[:, 2] == 0)
    assert not np.any(result.queries.parent_cell[:, 1] == 2)
    assert np.any(result.eligible_mask[:, :2])


def test_anisotropic_subdivision_and_source_uv_deduplication() -> None:
    """u 간격만 큰 평면은 u만 세분화하고 공유 edge query를 한 번만 남겨야 한다."""

    height, width = 5, 6
    rays = _anisotropic_rays(height, width, step_u=0.10, step_v=0.01)
    result = generate_guided_observed_queries(
        rays,
        np.ones((height, width), dtype=bool),
        np.ones((height, width), dtype=np.float32),
        _identity_camera(width, height),
        BevConfig(center_xy=(0.0, 0.0), size_m=100.0),
        RaySamplerConfig(
            target_surface_gap_m=0.04,
            max_added_queries_train=10_000,
            guided_train_fraction=1.0,
            dedupe_uv_decimals=4,
        ),
        mode="surface",
    )

    assert np.all(result.subdivision_u[result.eligible_mask] == 3)
    assert np.all(result.subdivision_v[result.eligible_mask] == 1)
    rounded_uv = np.round(result.queries.source_uv, 4)
    assert len(np.unique(rounded_uv, axis=0)) == len(result.queries)
    assert np.all(result.queries.subdivision_u == 3)
    assert np.all(result.queries.subdivision_v == 1)
    assert result.queries.sampling_features.shape == (len(result.queries), 3)


def test_guided_sampler_is_deterministic_and_honors_budget() -> None:
    """같은 입력은 같은 우선순위 query를 만들며 추가 query 예산을 넘지 않는다."""

    height = width = 9
    rays = _anisotropic_rays(height, width, step_u=0.12, step_v=0.08)
    config = RaySamplerConfig(
        target_surface_gap_m=0.02,
        max_added_queries_train=23,
        guided_train_fraction=1.0,
    )
    args = (
        rays,
        np.ones((height, width), dtype=bool),
        np.ones((height, width), dtype=np.float32),
        _identity_camera(width, height),
        BevConfig(center_xy=(0.0, 0.0), size_m=100.0),
        config,
        "surface",
    )
    first = generate_guided_observed_queries(*args)
    second = generate_guided_observed_queries(*args)

    assert len(first.queries) == 23
    assert len(first.queries) <= first.query_budget
    assert first.budget_truncated
    np.testing.assert_array_equal(first.queries.source_uv, second.queries.source_uv)
    np.testing.assert_array_equal(first.queries.ray_dir, second.queries.ray_dir)
    np.testing.assert_array_equal(first.added_density, second.added_density)
