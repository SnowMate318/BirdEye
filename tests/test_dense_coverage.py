from __future__ import annotations

import numpy as np

from wide_fov_supervision_v2.config import BevConfig, FisheyeCameraConfig, RaySamplerConfig
from wide_fov_supervision_v2.modules.dense_coverage import build_dense_coverage_bev


def _simple_camera() -> FisheyeCameraConfig:
    return FisheyeCameraConfig(
        width=2,
        height=2,
        world_from_camera=((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
        camera_position_world=(0.0, 0.0, 0.0),
        geometry_z_eps=1.0e-6,
    )


def _corner_rays() -> np.ndarray:
    rays = np.array(
        [
            [[-0.2, -0.2, 1.0], [0.2, -0.2, 1.0]],
            [[-0.2, 0.2, 1.0], [0.2, 0.2, 1.0]],
        ],
        dtype=np.float32,
    )
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def test_dense_source_cell_rays_fill_empty_bev_cells() -> None:
    config = BevConfig(center_xy=(0.0, 0.0), size_m=2.0, meters_per_pixel=0.25)
    base_rgb = np.zeros((config.resolution, config.resolution, 4), dtype=np.uint8)
    base_valid = np.zeros((config.resolution, config.resolution), dtype=np.uint8)

    result = build_dense_coverage_bev(
        rgb=np.full((2, 2, 3), 180, dtype=np.uint8),
        depth0_z=np.full((2, 2), 2.0, dtype=np.float32),
        rays_cv=_corner_rays(),
        source_valid=np.ones((2, 2), dtype=bool),
        camera=_simple_camera(),
        bev_config=config,
        sampler_config=RaySamplerConfig(dense_coverage_subdivision=3),
        base_bev_rgb=base_rgb,
        base_bev_valid=base_valid,
        base_top_occupancy=np.zeros_like(base_valid),
        base_top_probability=np.zeros_like(base_valid),
        floor_z=0.0,
    )

    assert result.metrics["dense_coverage_source_cells"] == 1
    assert result.metrics["dense_coverage_predicted_queries"] == 4
    assert result.metrics["dense_coverage_newly_covered_cells"] > 0
    assert result.metrics["dense_coverage_top_added_cells"] > 0
    assert np.count_nonzero(result.bev_valid) == result.metrics["dense_coverage_newly_covered_cells"]
    assert np.count_nonzero(result.observed_support_occupancy) == np.count_nonzero(result.bev_valid)
    assert result.observed_top_occupancy is not None
    assert np.count_nonzero(result.observed_top_occupancy) > 0


def test_dense_source_cell_skips_depth_discontinuity() -> None:
    config = BevConfig(center_xy=(0.0, 0.0), size_m=2.0, meters_per_pixel=0.25)
    base_rgb = np.zeros((config.resolution, config.resolution, 4), dtype=np.uint8)
    base_valid = np.zeros((config.resolution, config.resolution), dtype=np.uint8)

    result = build_dense_coverage_bev(
        rgb=np.full((2, 2, 3), 180, dtype=np.uint8),
        depth0_z=np.array([[1.0, 10.0], [1.0, 10.0]], dtype=np.float32),
        rays_cv=_corner_rays(),
        source_valid=np.ones((2, 2), dtype=bool),
        camera=_simple_camera(),
        bev_config=config,
        sampler_config=RaySamplerConfig(dense_coverage_subdivision=3),
        base_bev_rgb=base_rgb,
        base_bev_valid=base_valid,
    )

    assert result.metrics["dense_coverage_source_cells"] == 0
    assert result.metrics["dense_coverage_predicted_queries"] == 0
    assert result.metrics["dense_coverage_depth_discontinuity_skipped_cells"] == 1
    assert not np.any(result.bev_valid)
