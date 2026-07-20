from __future__ import annotations

import numpy as np

from wide_fov_supervision_v2.config import make_default_config
from wide_fov_supervision_v2.datasets.nyu.dataset import (
    NYUFrame,
    make_virtual_fisheye_orientation,
    orientation_for_frame,
    sample_nyu_depth_at_fisheye_rays,
)
from wide_fov_supervision_v2.train.query_cache import attach_exact_nyu_query_gt, query_sampler_config_hash


ORIENTATIONS = (
    (0.0, 0.0),
    (55.0, 0.0),
    (-55.0, 0.0),
    (0.0, 55.0),
    (0.0, -55.0),
)


def test_orientation_cycles_by_raw_frame_index() -> None:
    selected = [orientation_for_frame(index, ORIENTATIONS) for index in range(7)]
    assert [item[0] for item in selected] == [0, 1, 2, 3, 4, 0, 1]
    assert [(item[1].yaw_degrees, item[1].pitch_degrees) for item in selected[:5]] == list(ORIENTATIONS)


def test_rotated_nyu_depth_is_converted_to_target_fisheye_z() -> None:
    yaw_degrees = 55.0
    orientation = make_virtual_fisheye_orientation(yaw_degrees, 0.0)
    # 이 fish ray는 +55도 회전 뒤 NYU optical axis가 된다.
    yaw = np.deg2rad(yaw_degrees)
    ray_fisheye = np.array([[-np.sin(yaw), 0.0, np.cos(yaw)]], dtype=np.float32)
    depth_nyu = np.full((480, 640), 2.0, dtype=np.float32)

    result = sample_nyu_depth_at_fisheye_rays(depth_nyu, ray_fisheye, orientation)

    assert result.source_observed.tolist() == [True]
    assert result.gt_valid.tolist() == [True]
    assert np.isclose(result.radial_t[0], 2.0, atol=1.0e-5)
    assert np.isclose(result.depth_z[0], 2.0 * np.cos(yaw), atol=1.0e-5)
    assert not np.isclose(result.depth_z[0], 2.0, atol=1.0e-3)


def test_observed_and_gt_valid_are_independent_masks() -> None:
    orientation = make_virtual_fisheye_orientation(0.0, 0.0)
    ray = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    invalid_depth = np.zeros((480, 640), dtype=np.float32)

    result = sample_nyu_depth_at_fisheye_rays(invalid_depth, ray, orientation)

    assert result.source_observed.tolist() == [True]
    assert result.gt_valid.tolist() == [False]
    assert np.isnan(result.depth_z[0])


def test_exact_center_and_stencil_gt_are_saved_in_fisheye_z() -> None:
    config = make_default_config()
    orientation = make_virtual_fisheye_orientation(0.0, 0.0, name="center")
    frame = NYUFrame(
        rgb=np.zeros((480, 640, 3), dtype=np.uint8),
        depth_z=np.full((480, 640), 2.0, dtype=np.float32),
        index=0,
    )
    payload = {"ray_dir": np.array([[0.0, 0.0, 1.0]], dtype=np.float32)}

    result = attach_exact_nyu_query_gt(frame, orientation, payload, config)

    assert result["stencil_ray_dir"].shape == (1, 4, 3)
    assert result["query_gt_valid"].tolist() == [True]
    assert result["stencil_gt_valid"].tolist() == [[True, True, True, True]]
    assert np.allclose(result["query_depth_gt_z"], 2.0, atol=1.0e-5)
    assert np.allclose(result["stencil_depth_gt_z"], 2.0, atol=1.0e-5)


def test_query_cache_hash_changes_with_sampler_configuration() -> None:
    config = make_default_config()
    before = query_sampler_config_hash(config)
    config.ray.target_surface_gap_m *= 2.0
    after = query_sampler_config_hash(config)

    assert len(before) == 16
    assert before != after
