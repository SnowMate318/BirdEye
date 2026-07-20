from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wide_fov_supervision_v2.config import BevConfig, FisheyeCameraConfig, RaySamplerConfig
from wide_fov_supervision_v2.modules.adaptive_ray import spherical_bilerp
from wide_fov_supervision_v2.modules.camera_geometry import camera_to_world_points


@dataclass(frozen=True)
class DenseCoverageResult:
    """Result of streaming dense source-cell rays into the BEV map.

    The dense path is a BEV postprocess: it does not create a large in-memory
    `RayQuerySet`.  Each valid source 2x2 cell is subdivided, converted to
    deterministic camera rays, placed with the current D0 z-depth, and splatted
    only into BEV cells that were still unobserved before this stage.
    """

    bev_rgb: np.ndarray
    bev_valid: np.ndarray
    newly_covered: np.ndarray
    added_density: np.ndarray
    observed_top_occupancy: np.ndarray | None
    top_probability_map: np.ndarray | None
    observed_support_occupancy: np.ndarray
    metrics: dict[str, int | float | bool]


def query_fractions(subdivision: int) -> np.ndarray:
    """Return interior `(u, v)` samples for a source 2x2 pixel cell."""

    if subdivision < 2:
        raise ValueError("dense_coverage_subdivision must be at least 2")
    values = np.linspace(0.0, 1.0, subdivision + 1, dtype=np.float32)[1:-1]
    rel_u, rel_v = np.meshgrid(values, values, indexing="xy")
    return np.column_stack([rel_u.ravel(), rel_v.ravel()]).astype(np.float32)


def _bilerp_scalar(corners: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    fx = fractions[None, :, 0]
    fy = fractions[None, :, 1]
    top = corners[:, 0, None] * (1.0 - fx) + corners[:, 1, None] * fx
    bottom = corners[:, 2, None] * (1.0 - fx) + corners[:, 3, None] * fx
    return (top * (1.0 - fy) + bottom * fy).astype(np.float32)


def _bilerp_vector(corners: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    fx = fractions[None, :, 0, None]
    fy = fractions[None, :, 1, None]
    top = corners[:, 0, None, :] * (1.0 - fx) + corners[:, 1, None, :] * fx
    bottom = corners[:, 2, None, :] * (1.0 - fx) + corners[:, 3, None, :] * fx
    return (top * (1.0 - fy) + bottom * fy).astype(np.float32)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.divide(vectors, np.clip(norm, 1.0e-8, None)).astype(np.float32)


def _depth_derived_normals_world(point_corners_world: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    """Compute query normals from the local bilinear 3D patch.

    Dense BEV recovery should judge top-facing cells from geometry, not from a
    separately predicted DSINE prior.  The four source-corner 3D points define a
    local bilinear patch; its u/v tangent vectors give a normal at every query.
    The sign is flipped to keep the normal's world-z component non-negative.
    """

    p00 = point_corners_world[:, 0, None, :]
    p10 = point_corners_world[:, 1, None, :]
    p01 = point_corners_world[:, 2, None, :]
    p11 = point_corners_world[:, 3, None, :]
    rel_u = fractions[None, :, 0, None]
    rel_v = fractions[None, :, 1, None]
    tangent_u = (1.0 - rel_v) * (p10 - p00) + rel_v * (p11 - p01)
    tangent_v = (1.0 - rel_u) * (p01 - p00) + rel_u * (p11 - p10)
    normals = _normalize(np.cross(tangent_u, tangent_v))
    flip = normals[..., 2:3] < 0.0
    return np.where(flip, -normals, normals).astype(np.float32)


def _bev_indices(points_world: np.ndarray, config: BevConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    half = config.size_m * 0.5
    min_x = config.center_xy[0] - half
    min_y = config.center_xy[1] - half
    max_x = config.center_xy[0] + half
    max_y = config.center_xy[1] + half
    inside = (
        np.isfinite(points_world).all(axis=1)
        & (points_world[:, 0] >= min_x)
        & (points_world[:, 0] < max_x)
        & (points_world[:, 1] >= min_y)
        & (points_world[:, 1] < max_y)
    )
    cols = np.floor((points_world[:, 0] - min_x) / config.meters_per_pixel).astype(np.int64)
    rows = config.resolution - 1 - np.floor((points_world[:, 1] - min_y) / config.meters_per_pixel).astype(np.int64)
    inside &= (rows >= 0) & (rows < config.resolution) & (cols >= 0) & (cols < config.resolution)
    return rows, cols, inside


def _valid_source_cells(
    rays_cv: np.ndarray,
    source_valid: np.ndarray,
    depth0_z: np.ndarray,
    camera: FisheyeCameraConfig,
    sampler: RaySamplerConfig,
) -> np.ndarray:
    point_valid = (
        np.asarray(source_valid, dtype=bool)
        & np.isfinite(depth0_z)
        & (depth0_z > 0.0)
        & np.isfinite(rays_cv).all(axis=-1)
        & (rays_cv[..., 2] > camera.geometry_z_eps)
    )
    cell_valid = (
        point_valid[:-1, :-1]
        & point_valid[:-1, 1:]
        & point_valid[1:, :-1]
        & point_valid[1:, 1:]
    )
    safe_log = np.full(depth0_z.shape, np.nan, dtype=np.float32)
    safe_log[point_valid] = np.log(depth0_z[point_valid])
    l00 = safe_log[:-1, :-1]
    l10 = safe_log[:-1, 1:]
    l01 = safe_log[1:, :-1]
    l11 = safe_log[1:, 1:]
    log_min = np.minimum(np.minimum(l00, l10), np.minimum(l01, l11))
    log_max = np.maximum(np.maximum(l00, l10), np.maximum(l01, l11))
    continuous = (log_max - log_min) <= float(sampler.depth_discontinuity_log_threshold)
    return cell_valid & np.isfinite(log_min) & np.isfinite(log_max) & continuous


def _splat_rgb(
    bev_rgb: np.ndarray,
    bev_valid: np.ndarray,
    newly_covered: np.ndarray,
    best_z: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    colors: np.ndarray,
    world_z: np.ndarray,
    base_valid: np.ndarray,
) -> int:
    target = base_valid[rows, cols] == 0
    if not np.any(target):
        return 0
    rows = rows[target]
    cols = cols[target]
    colors = colors[target]
    world_z = world_z[target]
    cells = rows * bev_rgb.shape[1] + cols

    order = np.lexsort((world_z, cells))
    sorted_cells = cells[order]
    keep = np.empty(len(order), dtype=bool)
    keep[:-1] = sorted_cells[:-1] != sorted_cells[1:]
    keep[-1] = True
    selected = order[keep]

    selected_rows = rows[selected]
    selected_cols = cols[selected]
    selected_z = world_z[selected]
    replace = selected_z > best_z[selected_rows, selected_cols]
    if not np.any(replace):
        return 0

    selected_rows = selected_rows[replace]
    selected_cols = selected_cols[replace]
    bev_rgb[selected_rows, selected_cols, :3] = colors[selected][replace]
    bev_rgb[selected_rows, selected_cols, 3] = 255
    bev_valid[selected_rows, selected_cols] = 255
    newly_covered[selected_rows, selected_cols] = 255
    best_z[selected_rows, selected_cols] = selected_z[replace]
    return int(np.count_nonzero(replace))


def _splat_top(
    top_map: np.ndarray | None,
    prob_map: np.ndarray | None,
    rows: np.ndarray,
    cols: np.ndarray,
    points_world: np.ndarray,
    normals_world: np.ndarray,
    floor_z: float | None,
    bev_config: BevConfig,
) -> None:
    if top_map is None or prob_map is None or floor_z is None or not np.isfinite(floor_z):
        return
    normal_z = normals_world[:, 2]
    valid = np.isfinite(points_world).all(axis=1) & np.isfinite(normal_z)
    if not np.any(valid):
        return
    rows = rows[valid]
    cols = cols[valid]
    points_world = points_world[valid]
    normal_z = normal_z[valid]
    above_floor = points_world[:, 2] >= floor_z + float(bev_config.top_min_height_above_floor_m)
    probability = np.where(above_floor, np.clip(normal_z, 0.0, 1.0), 0.0)
    observed = np.where(probability >= float(bev_config.top_normal_z_threshold), 255, 0).astype(np.uint8)
    probability_u8 = (probability * 255.0).astype(np.uint8)
    cells = rows * top_map.shape[1] + cols
    np.maximum.at(top_map.reshape(-1), cells, observed)
    np.maximum.at(prob_map.reshape(-1), cells, probability_u8)


def build_dense_coverage_bev(
    rgb: np.ndarray,
    depth0_z: np.ndarray,
    rays_cv: np.ndarray,
    source_valid: np.ndarray,
    camera: FisheyeCameraConfig,
    bev_config: BevConfig,
    sampler_config: RaySamplerConfig,
    *,
    base_bev_rgb: np.ndarray,
    base_bev_valid: np.ndarray,
    base_top_occupancy: np.ndarray | None = None,
    base_top_probability: np.ndarray | None = None,
    floor_z: float | None = None,
) -> DenseCoverageResult:
    """Fill BEV holes by streaming dense deterministic rays inside source cells.

    This mirrors the original `wide_fov_supervision` source-quad recovery idea,
    but keeps it as a bounded BEV module.  It only fills cells that are invalid
    in `base_bev_valid`; existing source/adaptive BEV pixels are preserved.
    """

    fractions = query_fractions(int(sampler_config.dense_coverage_subdivision))
    query_count = int(len(fractions))
    cell_mask = _valid_source_cells(rays_cv, source_valid, depth0_z, camera, sampler_config)
    cell_rows, cell_cols = np.nonzero(cell_mask)

    bev_rgb = np.array(base_bev_rgb, copy=True)
    bev_valid = np.array(base_bev_valid, copy=True)
    base_valid = np.asarray(base_bev_valid, dtype=np.uint8)
    newly_covered = np.zeros_like(base_valid, dtype=np.uint8)
    best_z = np.full(base_valid.shape, -np.inf, dtype=np.float32)
    added_density = np.zeros(cell_mask.shape, dtype=np.float32)
    if len(cell_rows) > 0:
        added_density[cell_rows, cell_cols] = float(query_count)

    top_map = None if base_top_occupancy is None else np.array(base_top_occupancy, copy=True)
    prob_map = None if base_top_probability is None else np.array(base_top_probability, copy=True)

    support_offsets_y = np.array([0, 0, 1, 1], dtype=np.int64)
    support_offsets_x = np.array([0, 1, 0, 1], dtype=np.int64)
    chunk_cells = max(1, int(sampler_config.dense_coverage_chunk_cells))
    accepted_queries = 0
    projected_queries = 0
    updated_cells = 0

    for start in range(0, len(cell_rows), chunk_cells):
        stop = min(start + chunk_cells, len(cell_rows))
        y = cell_rows[start:stop]
        x = cell_cols[start:stop]
        sy = y[:, None] + support_offsets_y[None, :]
        sx = x[:, None] + support_offsets_x[None, :]

        ray_corners = rays_cv[sy, sx].astype(np.float32)
        depth_corners = depth0_z[sy, sx].astype(np.float32)
        query_rays = spherical_bilerp(
            ray_corners[:, 0, None, :],
            ray_corners[:, 1, None, :],
            ray_corners[:, 2, None, :],
            ray_corners[:, 3, None, :],
            fractions[None, :, 0],
            fractions[None, :, 1],
        )
        depth = _bilerp_scalar(depth_corners, fractions)
        colors = np.clip(_bilerp_vector(rgb[sy, sx].astype(np.float32), fractions), 0.0, 255.0).astype(np.uint8)
        corner_radial = depth_corners / np.clip(ray_corners[..., 2], camera.geometry_z_eps, None)
        corner_points_cv = ray_corners * corner_radial[..., None]
        corner_points_world = camera_to_world_points(
            corner_points_cv.reshape(-1, 3),
            camera,
        ).reshape(-1, 4, 3).astype(np.float32)
        query_normals_world = _depth_derived_normals_world(corner_points_world, fractions)

        rays_flat = query_rays.reshape(-1, 3)
        depth_flat = depth.reshape(-1)
        colors_flat = colors.reshape(-1, 3)
        valid = (
            np.isfinite(depth_flat)
            & (depth_flat > 0.0)
            & np.isfinite(rays_flat).all(axis=1)
            & (rays_flat[:, 2] > camera.geometry_z_eps)
        )
        if not np.any(valid):
            continue

        radial = depth_flat[valid] / rays_flat[valid, 2]
        points_cv = rays_flat[valid] * radial[:, None]
        points_world = camera_to_world_points(points_cv, camera).astype(np.float32)
        rows, cols, inside = _bev_indices(points_world, bev_config)
        accepted_queries += int(len(points_world))
        projected_queries += int(np.count_nonzero(inside))
        if not np.any(inside):
            continue

        points_inside = points_world[inside]
        rows_inside = rows[inside]
        cols_inside = cols[inside]
        colors_inside = colors_flat[valid][inside]
        updated_cells += _splat_rgb(
            bev_rgb,
            bev_valid,
            newly_covered,
            best_z,
            rows_inside,
            cols_inside,
            colors_inside,
            points_inside[:, 2].astype(np.float32),
            base_valid,
        )

        _splat_top(
            top_map,
            prob_map,
            rows_inside,
            cols_inside,
            points_inside,
            query_normals_world.reshape(-1, 3)[valid][inside],
            floor_z,
            bev_config,
        )

    top_added = 0
    if top_map is not None and base_top_occupancy is not None:
        top_added = int(np.count_nonzero((top_map > 0) & (np.asarray(base_top_occupancy) == 0)))
    observed_support = np.where(bev_valid > 0, 255, 0).astype(np.uint8)
    all_cell_valid = (
        source_valid[:-1, :-1]
        & source_valid[:-1, 1:]
        & source_valid[1:, :-1]
        & source_valid[1:, 1:]
    )
    metrics = {
        "dense_coverage_enabled": True,
        "dense_coverage_subdivision": int(sampler_config.dense_coverage_subdivision),
        "dense_coverage_query_count_per_cell": query_count,
        "dense_coverage_source_cells": int(len(cell_rows)),
        "dense_coverage_predicted_queries": int(len(cell_rows) * query_count),
        "dense_coverage_accepted_queries": int(accepted_queries),
        "dense_coverage_projected_queries": int(projected_queries),
        "dense_coverage_updated_cells": int(updated_cells),
        "dense_coverage_newly_covered_cells": int(np.count_nonzero(newly_covered)),
        "dense_coverage_top_added_cells": int(top_added),
        "dense_coverage_depth_discontinuity_skipped_cells": int(np.count_nonzero(all_cell_valid & ~cell_mask)),
    }
    return DenseCoverageResult(
        bev_rgb=bev_rgb,
        bev_valid=bev_valid,
        newly_covered=newly_covered,
        added_density=added_density,
        observed_top_occupancy=top_map,
        top_probability_map=prob_map,
        observed_support_occupancy=observed_support,
        metrics=metrics,
    )
