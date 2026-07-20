from __future__ import annotations

import numpy as np
import pytest

from wide_fov_supervision_v2.backbone.depth_source import load_external_z_depth
from wide_fov_supervision_v2.backbone.runner import BackboneRunner
from wide_fov_supervision_v2.config import BackboneConfig, FisheyeCameraConfig, PathConfig
from wide_fov_supervision_v2.modules.camera_geometry import build_fisheye_rays


class _FakeNormalModel:
    """foundation checkpoint 없이 depth-source wiring만 검증하는 DSINE test double."""

    def predict(self, rgb, camera=None, intrinsics=None):
        normal = np.zeros((*rgb.shape[:2], 3), dtype=np.float32)
        normal[..., 2] = -1.0
        return normal


def test_external_depth_loader_masks_invalid_and_rejects_resize(tmp_path) -> None:
    path = tmp_path / "depth.npy"
    np.save(path, np.array([[1.0, 0.0], [np.inf, 2.0]], dtype=np.float32))

    loaded = load_external_z_depth(path, (2, 2))
    np.testing.assert_array_equal(loaded.valid, [[True, False], [False, True]])
    assert np.isnan(loaded.depth_z[0, 1])
    assert np.isnan(loaded.depth_z[1, 0])
    assert loaded.metadata["external_depth_shape"] == [2, 2]

    with pytest.raises(ValueError, match="does not match"):
        load_external_z_depth(path, (3, 2))


def test_external_depth_direct_and_tangent_do_not_construct_da_v2() -> None:
    camera = FisheyeCameraConfig(width=32, height=32, fx=14.0, fy=14.0, cx=15.5, cy=15.5)
    backbone = BackboneConfig(tangent_resolution=16, tangent_fov_degrees=100.0)
    runner = BackboneRunner(PathConfig(), backbone, camera)
    runner._normal_model = _FakeNormalModel()
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    depth = np.full((32, 32), 2.0, dtype=np.float32)
    rays = build_fisheye_rays(camera)

    direct = runner.run_direct(rgb, depth_override_z=depth)
    tangent = runner.run_tangent(rgb, rays.rays_cv, depth_override_z=depth)

    assert runner._depth_model is None
    np.testing.assert_array_equal(direct.depth0_z, depth)
    np.testing.assert_array_equal(tangent.depth0_z, depth)
    assert direct.valid.all()
    assert tangent.valid.any()
    assert np.isfinite(tangent.normal0[tangent.valid]).all()
