from __future__ import annotations

import numpy as np

from wide_fov_supervision_v2.modules.prepare.edge_estimate.config import EdgeDataConfig
from wide_fov_supervision_v2.modules.prepare.edge_estimate.pseudo_labels import (
    EDGE_CREASE,
    EDGE_JUNCTION,
    EDGE_OCCLUSION,
    build_pseudo_edges,
)


def _rays(size: int = 64, focal: float = 50.0) -> np.ndarray:
    y, x = np.mgrid[:size, :size]
    x = (x - (size - 1) * 0.5) / focal
    y = (y - (size - 1) * 0.5) / focal
    rays = np.stack([x, y, np.ones_like(x)], axis=-1).astype(np.float32)
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def test_single_plane_is_continuous() -> None:
    depth = np.full((64, 64), 2.0, dtype=np.float32)
    labels = build_pseudo_edges(depth, np.ones_like(depth, dtype=bool), _rays(), EdgeDataConfig())
    assert not np.any(labels.edge)
    assert not np.any(labels.ignore[3:-3, 3:-3])


def test_depth_step_is_occlusion_with_near_and_far_depth() -> None:
    depth = np.full((64, 64), 2.0, dtype=np.float32)
    depth[:, 32:] = 4.0
    labels = build_pseudo_edges(depth, np.ones_like(depth, dtype=bool), _rays(), EdgeDataConfig())
    mask = labels.edge_type == EDGE_OCCLUSION
    assert np.any(mask[:, 28:36])
    assert np.all(np.isfinite(labels.near_depth_z[mask]))
    assert np.all(np.isfinite(labels.far_depth_z[mask]))
    assert float(np.nanmedian(labels.near_depth_z[mask])) == 2.0
    assert float(np.nanmedian(labels.far_depth_z[mask])) == 4.0


def test_two_intersecting_planes_create_crease() -> None:
    size = 64
    y, x = np.mgrid[:size, :size]
    x_normalized = (x - (size - 1) * 0.5) / 50.0
    slope = np.where(x < size // 2, 0.8, -0.8)
    depth = (2.0 / (1.0 - slope * x_normalized)).astype(np.float32)
    labels = build_pseudo_edges(depth, np.ones_like(depth, dtype=bool), _rays(), EdgeDataConfig())
    crease_or_junction = (labels.edge_type == EDGE_CREASE) | (labels.edge_type == EDGE_JUNCTION)
    assert np.any(crease_or_junction[:, 28:36])
    assert np.sum(labels.edge_type == EDGE_CREASE) > np.sum(labels.edge_type == EDGE_JUNCTION)


def test_raw_depth_hole_boundary_is_ignored() -> None:
    depth = np.full((64, 64), 2.0, dtype=np.float32)
    raw_valid = np.ones_like(depth, dtype=bool)
    raw_valid[24:40, 24:40] = False
    labels = build_pseudo_edges(depth, raw_valid, _rays(), EdgeDataConfig())
    assert np.all(labels.ignore[27:37, 27:37])
    assert not np.any(labels.edge[27:37, 27:37])


def test_two_depth_edges_meeting_in_one_region_create_junction() -> None:
    size = 64
    y, x = np.mgrid[:size, :size]
    depth = np.full((size, size), 2.0, dtype=np.float32)
    depth[(x >= 32) & (y < 32)] = 4.0
    depth[(x >= 32) & (y >= 32)] = 6.0
    labels = build_pseudo_edges(depth, np.ones_like(depth, dtype=bool), _rays(), EdgeDataConfig())
    assert np.any(labels.edge_type[26:38, 26:38] == EDGE_JUNCTION)
