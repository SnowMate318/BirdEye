from __future__ import annotations

import numpy as np

from wide_fov_supervision_v2.config import make_default_config
from wide_fov_supervision_v2.datasets.nyu.quad_dataset import generate_frame_quad_manifest


def test_manifest_generation_is_deterministic() -> None:
    config = make_default_config()
    config.train.test_quads_per_frame = 3
    config.train.guided_quad_fraction = 0.0
    config.train.continuous_quad_fraction = 1.0
    config.train.manifest_max_attempts_per_quad = 30
    depth = np.full((120, 160), 2.0, dtype=np.float32)
    first = generate_frame_quad_manifest(depth, frame_index=17, split="test", config=config)
    second = generate_frame_quad_manifest(depth, frame_index=17, split="test", config=config)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    assert first["corners_xy"].shape == (3, 4, 2)
    assert np.all(first["source_continuous"])
