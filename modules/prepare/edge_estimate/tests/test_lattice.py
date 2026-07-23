from __future__ import annotations

import numpy as np

from wide_fov_supervision_v2.config import FisheyeCameraConfig
from wide_fov_supervision_v2.modules.camera_geometry import unproject_fisheye_pixels
from wide_fov_supervision_v2.modules.prepare.edge_estimate.dataset import (
    _cell_queries,
    _lattice_uv,
    _sample_bilinear,
    _sample_nearest,
    spherical_bilerp,
)
from wide_fov_supervision_v2.modules.prepare.edge_estimate.pipeline import _subpixel_edge_nms


CAMERA = FisheyeCameraConfig(width=64, height=64, fx=30.0, fy=30.0, cx=31.5, cy=31.5)


def test_lattice_contains_sixteen_overlapping_cells() -> None:
    lattice = _lattice_uv(10, 20, 4)
    query_xy, query_rays, relative = _cell_queries(lattice, unproject_fisheye_pixels(lattice, CAMERA), 8, CAMERA)
    assert query_xy.shape == (4, 4, 64, 2)
    assert query_rays.shape == (4, 4, 64, 3)
    assert relative.shape == (4, 4, 64, 2)
    np.testing.assert_allclose(np.linalg.norm(query_rays, axis=-1), 1.0, atol=1.0e-6)


def test_shared_cell_boundary_has_identical_position_and_ray() -> None:
    lattice = _lattice_uv(10, 20, 3)
    query_xy, query_rays, _ = _cell_queries(lattice, unproject_fisheye_pixels(lattice, CAMERA), 8, CAMERA)
    left_xy = query_xy[1, 1].reshape(8, 8, 2)[:, -1]
    right_xy = query_xy[1, 2].reshape(8, 8, 2)[:, 0]
    left_ray = query_rays[1, 1].reshape(8, 8, 3)[:, -1]
    right_ray = query_rays[1, 2].reshape(8, 8, 3)[:, 0]
    np.testing.assert_allclose(left_xy, right_xy, atol=1.0e-6)
    np.testing.assert_allclose(left_ray, right_ray, atol=1.0e-6)


def test_spherical_bilerp_is_deterministic_and_preserves_corners() -> None:
    corners = unproject_fisheye_pixels(
        np.array([[20.5, 20.5], [21.5, 20.5], [21.5, 21.5], [20.5, 21.5]], dtype=np.float32),
        CAMERA,
    )
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.3, 0.7]], dtype=np.float32)
    first = spherical_bilerp(corners[None], uv[None])[0]
    second = spherical_bilerp(corners[None], uv[None])[0]
    np.testing.assert_allclose(first[:4], corners, atol=1.0e-6)
    np.testing.assert_array_equal(first, second)


def test_subpixel_nms_keeps_ridge_peak_and_suppresses_neighbor() -> None:
    edge_map = np.tile(np.array([0.0, 0.1, 0.4, 1.0, 0.6, 0.2, 0.1, 0.0], dtype=np.float32), (8, 1))
    keep = _subpixel_edge_nms(edge_map.reshape(1, 64)).reshape(8, 8)
    assert np.all(keep[:, 3])
    assert not np.any(keep[1:-1, 2])


def test_sampling_more_than_opencv_short_axis_limit_is_chunked() -> None:
    image = np.arange(16 * 16, dtype=np.float32).reshape(16, 16)
    index = np.arange(40_001)
    x = index % 16
    y = (index // 16) % 16
    uv = np.stack([x + 0.5, y + 0.5], axis=-1).astype(np.float32)
    expected = image[y, x]
    np.testing.assert_array_equal(_sample_nearest(image, uv), expected)
    np.testing.assert_allclose(_sample_bilinear(image, uv), expected, atol=1.0e-6)
