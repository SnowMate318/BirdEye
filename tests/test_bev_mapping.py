from __future__ import annotations

import numpy as np

from wide_fov_supervision_v2.config import BevConfig
from wide_fov_supervision_v2.modules.bev_mapping import build_bev_outputs, build_bev_valid


def test_rgb_bev_keeps_depth_point_when_normal_is_invalid() -> None:
    config = BevConfig(center_xy=(0.0, 0.0), size_m=2.0, meters_per_pixel=1.0)
    rgb = np.array([[120, 80, 40]], dtype=np.uint8)
    points = np.array([[0.25, 0.25, 1.0]], dtype=np.float32)
    normals = np.full((1, 3), np.nan, dtype=np.float32)
    point_valid = np.array([True])

    result = build_bev_outputs(
        rgb,
        points,
        normals,
        point_valid,
        config,
        normal_valid_mask=np.array([False]),
    )

    assert np.count_nonzero(result.bev_valid) == 1
    assert np.count_nonzero(result.observed_top_occupancy) == 0


def test_empty_added_point_set_covers_no_new_bev_cells() -> None:
    config = BevConfig(center_xy=(0.0, 0.0), size_m=2.0, meters_per_pixel=1.0)
    points = np.array([[0.25, 0.25, 1.0]], dtype=np.float32)
    before = build_bev_valid(points, np.array([True]), config)
    added = build_bev_valid(points, np.array([False]), config)
    after = np.maximum(before, added)
    newly_covered = (added > 0) & (before == 0)

    assert np.array_equal(after, before)
    assert not np.any(newly_covered)
