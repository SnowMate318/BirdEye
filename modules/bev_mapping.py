from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wide_fov_supervision_v2.config import BevConfig


@dataclass(frozen=True)
class BevResult:
    """BEV 산출물 묶음."""

    bev_rgb: np.ndarray
    bev_valid: np.ndarray
    observed_top_occupancy: np.ndarray
    top_probability_map: np.ndarray
    bounds_xy: tuple[float, float, float, float]
    floor_z: float
    metadata: dict


def build_bev_valid(
    world_points: np.ndarray,
    valid_mask: np.ndarray,
    config: BevConfig,
) -> np.ndarray:
    """world point가 실제로 닿은 BEV cell을 0/255 mask로 반환한다.

    추가 ray 전후 coverage를 같은 좌표 범위에서 비교하기 위한 함수다. free-space를
    ray tracing하지 않으므로 0인 cell은 free가 아니라 단순히 미관측일 수 있다.
    """

    half = config.size_m * 0.5
    min_x = config.center_xy[0] - half
    min_y = config.center_xy[1] - half
    max_x = config.center_xy[0] + half
    max_y = config.center_xy[1] + half
    points = np.asarray(world_points)[np.asarray(valid_mask, dtype=bool)]
    result = np.zeros((config.resolution, config.resolution), dtype=np.uint8)
    if len(points) == 0:
        return result
    inside = (
        np.isfinite(points).all(axis=1)
        & (points[:, 0] >= min_x)
        & (points[:, 0] < max_x)
        & (points[:, 1] >= min_y)
        & (points[:, 1] < max_y)
    )
    points = points[inside]
    if len(points) == 0:
        return result
    columns = np.floor((points[:, 0] - min_x) / config.meters_per_pixel).astype(np.int64)
    rows = config.resolution - 1 - np.floor((points[:, 1] - min_y) / config.meters_per_pixel).astype(np.int64)
    result[rows, columns] = 255
    return result


def build_bev_rgb(rgb: np.ndarray, world_points: np.ndarray, valid_mask: np.ndarray, config: BevConfig) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """world XY grid에 point color를 splat한다.

    같은 BEV cell에 여러 point가 들어오면 가장 높은 world-Z point의 RGB를 사용한다.
    """

    half = config.size_m * 0.5
    min_x = config.center_xy[0] - half
    max_x = config.center_xy[0] + half
    min_y = config.center_xy[1] - half
    max_y = config.center_xy[1] + half
    points = world_points[valid_mask]
    colors = rgb[valid_mask]
    inside = (
        (points[:, 0] >= min_x)
        & (points[:, 0] < max_x)
        & (points[:, 1] >= min_y)
        & (points[:, 1] < max_y)
        & np.isfinite(points).all(axis=1)
    )
    points = points[inside]
    colors = colors[inside]
    res = config.resolution
    bev = np.zeros((res, res, 4), dtype=np.uint8)
    valid = np.zeros((res, res), dtype=np.uint8)
    if len(points) == 0:
        return bev, valid, (min_x, min_y, max_x, max_y)
    cols = np.floor((points[:, 0] - min_x) / config.meters_per_pixel).astype(np.int64)
    rows = res - 1 - np.floor((points[:, 1] - min_y) / config.meters_per_pixel).astype(np.int64)
    cells = rows * res + cols
    order = np.lexsort((points[:, 2], cells))
    sorted_cells = cells[order]
    keep = np.empty(len(order), dtype=bool)
    keep[:-1] = sorted_cells[:-1] != sorted_cells[1:]
    keep[-1] = True
    selected = order[keep]
    bev[rows[selected], cols[selected], :3] = colors[selected]
    bev[rows[selected], cols[selected], 3] = 255
    valid[rows[selected], cols[selected]] = 255
    return bev, valid, (min_x, min_y, max_x, max_y)


def build_observed_top_maps(world_points: np.ndarray, normals_world: np.ndarray, valid_mask: np.ndarray, bounds: tuple[float, float, float, float], config: BevConfig) -> tuple[np.ndarray, np.ndarray, float]:
    """관측된 top-facing non-floor surface map을 만든다.

    검정 PNG로 저장되는 `observed_top_occupancy`는 classic occupancy/free-space가
    아니라, 관측된 점 중 "바닥보다 높고 위쪽을 향한 표면"만 표시한다.
    """

    min_x, min_y, max_x, max_y = bounds
    points = world_points[valid_mask]
    normals = normals_world[valid_mask]
    normal_z = normals[:, 2]
    inside = (
        (points[:, 0] >= min_x)
        & (points[:, 0] < max_x)
        & (points[:, 1] >= min_y)
        & (points[:, 1] < max_y)
        & np.isfinite(points).all(axis=1)
        & np.isfinite(normal_z)
    )
    points = points[inside]
    normal_z = normal_z[inside]
    res = config.resolution
    observed_top = np.zeros((res, res), dtype=np.uint8)
    probability = np.zeros((res, res), dtype=np.uint8)
    if len(points) == 0:
        return observed_top, probability, float("nan")
    upward = normal_z >= config.top_normal_z_threshold
    floor_samples = points[upward, 2] if np.any(upward) else points[:, 2]
    floor_z = float(np.percentile(floor_samples, config.floor_height_percentile))
    above_floor = points[:, 2] >= floor_z + config.top_min_height_above_floor_m
    prob = np.where(above_floor, np.clip(normal_z, 0.0, 1.0), 0.0)
    observed = np.where(prob >= config.top_normal_z_threshold, 255, 0).astype(np.uint8)
    prob_u8 = (prob * 255.0).astype(np.uint8)
    cols = np.floor((points[:, 0] - min_x) / config.meters_per_pixel).astype(np.int64)
    rows = res - 1 - np.floor((points[:, 1] - min_y) / config.meters_per_pixel).astype(np.int64)
    cells = rows * res + cols
    np.maximum.at(observed_top.reshape(-1), cells, observed)
    np.maximum.at(probability.reshape(-1), cells, prob_u8)
    return observed_top, probability, floor_z


def build_bev_outputs(
    rgb: np.ndarray,
    world_points: np.ndarray,
    normals_world: np.ndarray,
    valid_mask: np.ndarray,
    config: BevConfig,
    *,
    normal_valid_mask: np.ndarray | None = None,
) -> BevResult:
    """RGB BEV와 observed-top map을 함께 만든다.

    RGB coverage는 depth/point만 유효하면 보존하고, normal이 필요한 top map에만
    ``normal_valid_mask``를 추가한다. N* stencil이 무효라는 이유로 정상 3D point가
    BEV RGB에서 사라지는 것을 막는다.
    """

    bev_rgb, bev_valid, bounds = build_bev_rgb(rgb, world_points, valid_mask, config)
    top_valid = np.asarray(valid_mask, dtype=bool)
    if normal_valid_mask is not None:
        top_valid &= np.asarray(normal_valid_mask, dtype=bool)
    observed_top, prob, floor_z = build_observed_top_maps(world_points, normals_world, top_valid, bounds, config)
    return BevResult(
        bev_rgb=bev_rgb,
        bev_valid=bev_valid,
        observed_top_occupancy=observed_top,
        top_probability_map=prob,
        bounds_xy=bounds,
        floor_z=floor_z,
        metadata={
            "observed_top_occupancy_semantics": "black PNG pixels mean observed top-facing non-floor surface; white means free, non-top, or unobserved, not a classic occupancy grid",
            "floor_z": floor_z,
            "bounds_xy": bounds,
        },
    )
