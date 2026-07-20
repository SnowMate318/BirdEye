from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from wide_fov_supervision_v2.config import FisheyeCameraConfig


@dataclass(frozen=True)
class CameraRays:
    """source fisheye 영상의 pixel-center ray 묶음.

    Attributes:
        rays_cv: `(H, W, 3)` OpenCV camera 좌표계 단위 ray. +x 오른쪽, +y 아래,
            +z 전방이다. lens 밖 pixel은 0으로 채운다.
        valid: `(H, W)` bool. fisheye projection/unprojection round-trip이 성립하고
            geometry에 사용할 수 있는 pixel 여부이다. depth 유효성과는 별개이다.
        max_roundtrip_error_px: valid pixel에서 측정한 fisheye 재투영 오차 최대값.
    """

    rays_cv: np.ndarray
    valid: np.ndarray
    max_roundtrip_error_px: float


def camera_matrix(camera: FisheyeCameraConfig) -> np.ndarray:
    """OpenCV fisheye 함수에 넣는 3x3 intrinsics matrix를 만든다."""

    return np.array(
        [[camera.fx, 0.0, camera.cx], [0.0, camera.fy, camera.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def distortion_coeffs(camera: FisheyeCameraConfig) -> np.ndarray:
    """OpenCV fisheye distortion `(k1, k2, k3, k4)`를 column vector로 반환한다."""

    return np.asarray(camera.distortion, dtype=np.float64).reshape(4, 1)


def pixel_grid(width: int, height: int, *, centers: bool = True) -> np.ndarray:
    """`(H, W, 2)` pixel 좌표 grid를 만든다.

    centers=True이면 `(u+0.5, v+0.5)` pixel center를 사용한다. 이 프로젝트의
    source ray map은 모두 pixel center 기준이다.
    """

    offset = 0.5 if centers else 0.0
    u, v = np.meshgrid(
        np.arange(width, dtype=np.float64) + offset,
        np.arange(height, dtype=np.float64) + offset,
    )
    return np.stack([u, v], axis=-1)


def unproject_fisheye_pixels(uv: np.ndarray, camera: FisheyeCameraConfig) -> np.ndarray:
    """fisheye pixel 좌표를 OpenCV camera-frame 단위 ray로 역투영한다.

    Args:
        uv: `(..., 2)` pixel 좌표. 마지막 축 순서는 `(u, v)`.
        camera: fisheye camera 설정.

    Returns:
        `(..., 3)` 단위 ray. OpenCV fisheye undistortPoints는 normalized
        pinhole 좌표 `(x/z, y/z)`를 반환하므로 `(x, y, 1)`로 올린 뒤 정규화한다.
    """

    uv_flat = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
    normalized = cv2.fisheye.undistortPoints(
        uv_flat,
        camera_matrix(camera),
        distortion_coeffs(camera),
    ).reshape(-1, 2)
    rays = np.column_stack([normalized, np.ones(len(normalized), dtype=np.float64)])
    rays /= np.linalg.norm(rays, axis=1, keepdims=True).clip(min=1.0e-12)
    return rays.reshape(*np.asarray(uv).shape[:-1], 3)


def project_fisheye_rays(rays_cv: np.ndarray, camera: FisheyeCameraConfig) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV camera-frame ray를 fisheye pixel로 투영한다.

    이 함수는 OpenCV의 `cv2.fisheye.projectPoints` 대신 같은 Kannala-Brandt 식을
    직접 쓴다. 이유는 많은 ray를 batch로 다룰 때 더 빠르고, `ray_z <= 0` 같은
    전방 반구 경계 처리를 명확히 제어하기 위해서이다.

    Returns:
        uv: `(N, 2)` 또는 입력 shape에 대응되는 pixel 좌표.
        valid: 입력 ray가 finite이고 `ray_z > 0`인 투영 가능 여부.
    """

    original_shape = np.asarray(rays_cv).shape[:-1]
    rays = np.asarray(rays_cv, dtype=np.float64).reshape(-1, 3)
    uv = np.full((len(rays), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(rays).all(axis=1) & (rays[:, 2] > 1.0e-8)
    if not np.any(valid):
        return uv.reshape(*original_shape, 2), valid.reshape(original_shape)

    xyz = rays[valid]
    x = xyz[:, 0] / xyz[:, 2]
    y = xyz[:, 1] / xyz[:, 2]
    radius = np.hypot(x, y)
    theta = np.arctan(radius)
    theta2 = theta * theta
    k1, k2, k3, k4 = camera.distortion
    theta_distorted = theta * (
        1.0 + k1 * theta2 + k2 * theta2**2 + k3 * theta2**3 + k4 * theta2**4
    )
    scale = np.ones_like(radius)
    nonzero = radius > 1.0e-12
    scale[nonzero] = theta_distorted[nonzero] / radius[nonzero]

    uv_valid = np.column_stack(
        [camera.fx * x * scale + camera.cx, camera.fy * y * scale + camera.cy]
    )
    finite_uv = np.isfinite(uv_valid).all(axis=1)
    uv[np.flatnonzero(valid)[finite_uv]] = uv_valid[finite_uv]
    valid_indices = np.flatnonzero(valid)
    valid[valid_indices[~finite_uv]] = False
    return uv.reshape(*original_shape, 2), valid.reshape(original_shape)


def build_fisheye_rays(camera: FisheyeCameraConfig, *, validate_roundtrip: bool = True) -> CameraRays:
    """현재 fisheye 카메라의 source pixel ray map을 만든다.

    lens valid mask는 projection round-trip이 1e-3 pixel 이하인 pixel이다. 이 mask는
    "카메라 geometry상 해당 pixel이 lens 안에 있다"는 뜻이며, RGB-D depth 값이
    실제로 valid한지는 별도 mask에서 다룬다.
    """

    uv = pixel_grid(camera.width, camera.height, centers=True)
    rays = unproject_fisheye_pixels(uv, camera)
    projected, projected_valid = project_fisheye_rays(rays, camera)
    error = np.linalg.norm(projected - uv, axis=-1)
    valid = projected_valid & np.isfinite(error) & (error <= 1.0e-3)
    max_error = float(np.max(error[valid])) if np.any(valid) else float("inf")
    if validate_roundtrip:
        center_ray = rays[int(camera.cy), int(camera.cx)]
        if np.linalg.norm(center_ray[:2]) > 1.0e-4:
            raise RuntimeError(f"Principal ray is not close to optical axis: {center_ray}")
        if not np.any(valid):
            raise RuntimeError("No valid fisheye ray was generated.")
    rays = rays.astype(np.float32)
    rays[~valid] = 0.0
    return CameraRays(rays_cv=rays, valid=valid, max_roundtrip_error_px=max_error)


def angular_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """두 단위 ray 사이 각도 gap(rad)을 계산한다."""

    dot = np.sum(a * b, axis=-1)
    return np.arccos(np.clip(dot, -1.0, 1.0))


def cell_angular_gap(rays_cv: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """source pixel cell별 최대 edge angular gap을 계산한다.

    각 cell은 네 corner/source pixel-center ray `(v,u), (v,u+1), (v+1,u),
    (v+1,u+1)`로 정의한다. 반환 shape는 `(H-1, W-1)`이다.
    """

    r00 = rays_cv[:-1, :-1]
    r10 = rays_cv[:-1, 1:]
    r01 = rays_cv[1:, :-1]
    r11 = rays_cv[1:, 1:]
    v00 = valid[:-1, :-1]
    v10 = valid[:-1, 1:]
    v01 = valid[1:, :-1]
    v11 = valid[1:, 1:]
    cell_valid = v00 & v10 & v01 & v11
    gaps = np.stack(
        [
            angular_distance(r00, r10),
            angular_distance(r01, r11),
            angular_distance(r00, r01),
            angular_distance(r10, r11),
        ],
        axis=0,
    )
    max_gap = np.max(gaps, axis=0)
    max_gap[~cell_valid] = np.nan
    return max_gap.astype(np.float32), cell_valid


def central_median_gap(gap_map: np.ndarray, valid: np.ndarray, fraction: float) -> float:
    """중앙 `fraction` 영역에서 median ray gap(rad)을 구한다."""

    h, w = gap_map.shape
    margin_y = int(round(h * (1.0 - fraction) * 0.5))
    margin_x = int(round(w * (1.0 - fraction) * 0.5))
    center_gap = gap_map[margin_y : h - margin_y, margin_x : w - margin_x]
    center_valid = valid[margin_y : h - margin_y, margin_x : w - margin_x]
    samples = center_gap[center_valid & np.isfinite(center_gap)]
    if len(samples) == 0:
        raise RuntimeError("Cannot compute central median ray gap; central valid cells are empty.")
    return float(np.median(samples))


def cv_rays_to_world(rays_cv: np.ndarray, world_from_camera: np.ndarray) -> np.ndarray:
    """OpenCV camera ray를 world vector로 변환한다.

    Isaac/기존 BEV 코드와 맞추기 위해 OpenCV의 +y down을 camera +y up으로 뒤집은
    뒤 world-from-camera rotation을 적용한다.
    """

    rays_camera = np.asarray(rays_cv, dtype=np.float64).copy()
    rays_camera[..., 1] *= -1.0
    return rays_camera @ np.asarray(world_from_camera, dtype=np.float64).T


def world_vectors_to_cv(vectors_world: np.ndarray, world_from_camera: np.ndarray) -> np.ndarray:
    """world vector를 OpenCV camera-frame vector로 변환한다."""

    vectors_camera = np.asarray(vectors_world, dtype=np.float64) @ np.asarray(world_from_camera, dtype=np.float64)
    vectors_cv = vectors_camera.copy()
    vectors_cv[..., 1] *= -1.0
    return vectors_cv


def points_from_z_depth(
    depth_z: np.ndarray,
    rays_cv: np.ndarray,
    *,
    z_eps: float = 1.0e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """z-depth와 ray로 camera-frame 3D point를 만든다.

    `radial_depth = z_depth / ray_z`, `P = radial_depth * ray`를 사용한다.
    `ray_z <= z_eps`인 전방 180도 경계 ray는 z-depth geometry가 불안정하므로
    invalid 처리한다.
    """

    depth = np.asarray(depth_z, dtype=np.float32)
    rays = np.asarray(rays_cv, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0) & np.isfinite(rays).all(axis=-1) & (rays[..., 2] > z_eps)
    radial = np.divide(
        depth,
        rays[..., 2],
        out=np.full_like(depth, np.nan, dtype=np.float32),
        where=valid,
    )
    points = radial[..., None] * rays
    points[~valid] = np.nan
    return points.astype(np.float32), radial.astype(np.float32), valid


def camera_to_world_points(
    points_cv: np.ndarray,
    camera: FisheyeCameraConfig,
) -> np.ndarray:
    """OpenCV camera-frame point를 world point로 변환한다."""

    rays_world_like = cv_rays_to_world(points_cv, np.asarray(camera.world_from_camera, dtype=np.float64))
    return np.asarray(camera.camera_position_world, dtype=np.float64) + rays_world_like
