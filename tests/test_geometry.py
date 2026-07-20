from __future__ import annotations

import numpy as np

from wide_fov_supervision_v2.config import FisheyeCameraConfig, RaySamplerConfig
from wide_fov_supervision_v2.modules.adaptive_ray import generate_adaptive_observed_queries, generate_front_hemisphere_queries
from wide_fov_supervision_v2.modules.camera_geometry import build_fisheye_rays, project_fisheye_rays


def small_camera() -> FisheyeCameraConfig:
    return FisheyeCameraConfig(width=96, height=96, fx=42.0, fy=42.0, cx=47.5, cy=47.5)


def test_fisheye_projection_unprojection_roundtrip() -> None:
    camera = small_camera()
    rays = build_fisheye_rays(camera)
    uv, valid = project_fisheye_rays(rays.rays_cv[rays.valid], camera)
    original_y, original_x = np.nonzero(rays.valid)
    original_uv = np.column_stack([original_x + 0.5, original_y + 0.5])
    err = np.linalg.norm(uv[valid] - original_uv[valid], axis=1)
    assert float(err.max()) < 1.0e-3


def test_adaptive_subdivision_reduces_max_gap() -> None:
    camera = small_camera()
    rays = build_fisheye_rays(camera)
    cfg = RaySamplerConfig(max_subdivision=6, target_gap_rad=None, central_fraction=0.25)
    queries, gap_before, gap_after, density, target = generate_adaptive_observed_queries(rays.rays_cv, rays.valid, cfg)
    assert target > 0.0
    assert len(queries) > 0
    assert float(np.nanmax(gap_after)) <= float(np.nanmax(gap_before))
    assert float(np.nanmax(gap_after)) <= target * 1.05
    added_cells = density > 0
    assert np.all(gap_before[added_cells] >= target)
    assert float(np.nanmean(gap_before[added_cells])) >= float(np.nanmean(gap_before[~added_cells]))


def test_front_hemisphere_unknown_for_boundary_or_unobserved() -> None:
    camera = small_camera()
    rays = build_fisheye_rays(camera)
    cfg = RaySamplerConfig()
    queries, coverage = generate_front_hemisphere_queries(camera, rays.valid, 0.08, cfg)
    assert len(queries) > 0
    assert np.any(queries.unknown)
    assert np.all(queries.unknown[queries.ray_dir[:, 2] <= camera.geometry_z_eps])
    assert set(np.unique(coverage)).issubset({0, 1, 2})
