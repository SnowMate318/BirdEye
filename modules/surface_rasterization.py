from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from wide_fov_supervision_v2.config import BevConfig


@dataclass(frozen=True)
class FloorSurfaceRasterResult:
    """연속 바닥 source cell을 BEV 면으로 rasterization한 결과.

    Attributes:
        bev_rgb: ``(R,R,4)`` uint8 RGBA. 바닥으로 판단된 source 2x2 cell의
            world-XY 사각형을 BEV image에 칠한 결과다.
        bev_valid: ``(R,R)`` uint8. ``255``인 cell은 surface fill이 닿은 BEV cell이다.
        source_cell_mask: ``(H-1,W-1)`` bool. 네 corner가 모두 유효하고 floor 높이
            근처이며 corner z-range가 작은 cell만 True다.
        floor_z: source point에서 추정한 floor 높이.
    """

    bev_rgb: np.ndarray
    bev_valid: np.ndarray
    source_cell_mask: np.ndarray
    floor_z: float


def _bev_pixel_xy(points_world: np.ndarray, config: BevConfig) -> tuple[np.ndarray, np.ndarray]:
    """world XY 좌표를 BEV image의 floating pixel 좌표로 변환한다."""

    half = float(config.size_m) * 0.5
    min_x = float(config.center_xy[0]) - half
    min_y = float(config.center_xy[1]) - half
    cols = (points_world[..., 0] - min_x) / max(float(config.meters_per_pixel), 1.0e-8)
    rows = float(config.resolution - 1) - (
        (points_world[..., 1] - min_y) / max(float(config.meters_per_pixel), 1.0e-8)
    )
    return cols, rows


def _points_inside_bev(points_world: np.ndarray, config: BevConfig) -> np.ndarray:
    """world point가 BEV bounds 안에 있는지 반환한다."""

    half = float(config.size_m) * 0.5
    min_x = float(config.center_xy[0]) - half
    max_x = float(config.center_xy[0]) + half
    min_y = float(config.center_xy[1]) - half
    max_y = float(config.center_xy[1]) + half
    return (
        np.isfinite(points_world).all(axis=-1)
        & (points_world[..., 0] >= min_x)
        & (points_world[..., 0] < max_x)
        & (points_world[..., 1] >= min_y)
        & (points_world[..., 1] < max_y)
    )


def rasterize_floor_surfaces(
    rgb: np.ndarray,
    world_points: np.ndarray,
    valid_mask: np.ndarray,
    config: BevConfig,
) -> FloorSurfaceRasterResult:
    """연속적인 source 2x2 바닥 cell을 BEV polygon으로 채운다.

    현재 ray completion은 query마다 3D point 하나를 만들고 BEV에 점으로 splat한다.
    바닥처럼 실제로는 연속 면인 영역은 이 방식만으로는 동심원/줄무늬 빈칸이 남는다.
    이 함수는 네 corner가 모두 floor 높이 근처인 source cell을 작은 3D quad로 보고,
    그 quad의 world XY projection을 BEV에 면으로 칠한다.

    안전 조건:
        - 네 corner source point가 모두 유효해야 한다.
        - 네 corner가 모두 BEV bounds 안에 있어야 한다.
        - 네 corner의 최대 z가 ``floor_z + floor_surface_fill_height_margin_m``
          이하여야 한다.
        - 네 corner의 z range가 ``floor_surface_fill_max_corner_z_range_m`` 이하여야
          한다. 그래서 랙/벽/큰 depth discontinuity를 가로지르는 cell은 채우지 않는다.

    이 결과는 floor coverage용 배경으로만 쓰며, top-facing non-floor occupancy에는
    넣지 않는다.
    """

    image = np.asarray(rgb, dtype=np.uint8)
    points = np.asarray(world_points, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(points).all(axis=-1)
    if image.shape[:2] != points.shape[:2] or valid.shape != points.shape[:2]:
        raise ValueError("rgb, world_points, valid_mask spatial shape가 같아야 합니다.")

    resolution = int(config.resolution)
    bev_color = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    bev_alpha = np.zeros((resolution, resolution), dtype=np.uint8)
    bev_rgb = np.zeros((resolution, resolution, 4), dtype=np.uint8)
    bev_valid = np.zeros((resolution, resolution), dtype=np.uint8)
    source_cell_mask = np.zeros((points.shape[0] - 1, points.shape[1] - 1), dtype=bool)

    valid_points = points[valid & _points_inside_bev(points, config)]
    if len(valid_points) == 0:
        return FloorSurfaceRasterResult(bev_rgb, bev_valid, source_cell_mask, float("nan"))
    floor_z = float(np.nanpercentile(valid_points[:, 2], float(config.floor_height_percentile)))

    p00, p10 = points[:-1, :-1], points[:-1, 1:]
    p01, p11 = points[1:, :-1], points[1:, 1:]
    v00, v10 = valid[:-1, :-1], valid[:-1, 1:]
    v01, v11 = valid[1:, :-1], valid[1:, 1:]
    cell_valid = v00 & v10 & v01 & v11
    cell_inside = (
        _points_inside_bev(p00, config)
        & _points_inside_bev(p10, config)
        & _points_inside_bev(p11, config)
        & _points_inside_bev(p01, config)
    )
    z_stack = np.stack([p00[..., 2], p10[..., 2], p11[..., 2], p01[..., 2]], axis=-1)
    z_min = np.full(source_cell_mask.shape, np.nan, dtype=np.float32)
    z_max = np.full(source_cell_mask.shape, np.nan, dtype=np.float32)
    if np.any(cell_valid):
        z_min[cell_valid] = np.min(z_stack[cell_valid], axis=-1)
        z_max[cell_valid] = np.max(z_stack[cell_valid], axis=-1)
    source_cell_mask = (
        cell_valid
        & cell_inside
        & np.isfinite(z_min)
        & np.isfinite(z_max)
        & (z_max <= floor_z + float(config.floor_surface_fill_height_margin_m))
        & ((z_max - z_min) <= float(config.floor_surface_fill_max_corner_z_range_m))
    )

    ys, xs = np.nonzero(source_cell_mask)
    if len(xs) == 0:
        return FloorSurfaceRasterResult(bev_rgb, bev_valid, source_cell_mask, floor_z)

    corner_points = np.stack([p00[ys, xs], p10[ys, xs], p11[ys, xs], p01[ys, xs]], axis=1)
    cols, rows = _bev_pixel_xy(corner_points, config)
    polygons = np.rint(np.stack([cols, rows], axis=-1)).astype(np.int32)
    corner_colors = np.stack(
        [image[:-1, :-1][ys, xs], image[:-1, 1:][ys, xs], image[1:, 1:][ys, xs], image[1:, :-1][ys, xs]],
        axis=1,
    )
    fill_colors = np.rint(corner_colors.astype(np.float32).mean(axis=1)).astype(np.uint8)

    for polygon, color in zip(polygons, fill_colors, strict=False):
        if cv2.contourArea(polygon.astype(np.float32)) <= 0.5:
            continue
        clipped = polygon.copy()
        clipped[:, 0] = np.clip(clipped[:, 0], 0, resolution - 1)
        clipped[:, 1] = np.clip(clipped[:, 1], 0, resolution - 1)
        cv2.fillConvexPoly(bev_color, clipped, tuple(int(v) for v in color.tolist()))
        cv2.fillConvexPoly(bev_alpha, clipped, 255)
        cv2.fillConvexPoly(bev_valid, clipped, 255)

    bev_rgb[..., :3] = bev_color
    bev_rgb[..., 3] = bev_alpha
    return FloorSurfaceRasterResult(bev_rgb, bev_valid, source_cell_mask, floor_z)
