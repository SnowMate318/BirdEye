from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from wide_fov_supervision_v2.config import BackboneConfig, FisheyeCameraConfig
from wide_fov_supervision_v2.modules.camera_geometry import project_fisheye_rays


@dataclass(frozen=True)
class TangentView:
    """fisheye source에서 잘라낸 pinhole tangent view."""

    name: str
    rgb: np.ndarray
    rotation_tangent_to_source: np.ndarray
    intrinsics: np.ndarray
    rays_tangent: np.ndarray
    valid: np.ndarray


def tangent_directions(polar_degrees: float) -> list[tuple[str, np.ndarray]]:
    """중앙 1개와 polar 55도 상하좌우 4개 view 방향."""

    p = np.deg2rad(float(polar_degrees))
    return [
        ("center", np.array([0.0, 0.0, 1.0], dtype=np.float64)),
        ("right", np.array([np.sin(p), 0.0, np.cos(p)], dtype=np.float64)),
        ("left", np.array([-np.sin(p), 0.0, np.cos(p)], dtype=np.float64)),
        ("down", np.array([0.0, np.sin(p), np.cos(p)], dtype=np.float64)),
        ("up", np.array([0.0, -np.sin(p), np.cos(p)], dtype=np.float64)),
    ]


def tangent_basis(direction_source: np.ndarray) -> np.ndarray:
    """tangent camera frame을 source camera frame으로 보내는 3x3 rotation."""

    z = direction_source / np.linalg.norm(direction_source).clip(min=1.0e-12)
    up_hint = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(z, up_hint))) > 0.95:
        up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    x = np.cross(up_hint, z)
    x /= np.linalg.norm(x).clip(min=1.0e-12)
    y = np.cross(z, x)
    y /= np.linalg.norm(y).clip(min=1.0e-12)
    return np.column_stack([x, y, z]).astype(np.float32)


def pinhole_intrinsics(size: int, fov_degrees: float) -> np.ndarray:
    """정사각 tangent view용 pinhole intrinsics."""

    f = 0.5 * size / np.tan(np.deg2rad(float(fov_degrees)) * 0.5)
    c = (size - 1) * 0.5
    return np.array([[f, 0.0, c], [0.0, f, c], [0.0, 0.0, 1.0]], dtype=np.float32)


def pinhole_rays(size: int, intrinsics: np.ndarray) -> np.ndarray:
    """tangent view pixel-center pinhole ray map `(S,S,3)`."""

    u, v = np.meshgrid(np.arange(size, dtype=np.float32) + 0.5, np.arange(size, dtype=np.float32) + 0.5)
    x = (u - intrinsics[0, 2]) / intrinsics[0, 0]
    y = (v - intrinsics[1, 2]) / intrinsics[1, 1]
    rays = np.stack([x, y, np.ones_like(x)], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True).clip(min=1.0e-12)
    return rays.astype(np.float32)


def build_tangent_views(rgb: np.ndarray, camera: FisheyeCameraConfig, config: BackboneConfig) -> list[TangentView]:
    """fisheye RGB에서 5개 tangent pinhole RGB view를 생성한다."""

    size = int(config.tangent_resolution)
    intrinsics = pinhole_intrinsics(size, config.tangent_fov_degrees)
    rays_tan = pinhole_rays(size, intrinsics)
    views: list[TangentView] = []
    for name, direction in tangent_directions(config.tangent_polar_degrees):
        rotation = tangent_basis(direction)
        rays_source = rays_tan @ rotation.T
        uv, valid = project_fisheye_rays(rays_source, camera)
        inside = valid & (uv[..., 0] >= 0.0) & (uv[..., 0] < camera.width - 1) & (uv[..., 1] >= 0.0) & (uv[..., 1] < camera.height - 1)
        map_x = (uv[..., 0] - 0.5).astype(np.float32)
        map_y = (uv[..., 1] - 0.5).astype(np.float32)
        view_rgb = cv2.remap(rgb, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        view_rgb[~inside] = 0
        views.append(TangentView(name=name, rgb=view_rgb, rotation_tangent_to_source=rotation, intrinsics=intrinsics, rays_tangent=rays_tan, valid=inside))
    return views


def fuse_tangent_predictions(
    source_rays: np.ndarray,
    views: list[TangentView],
    depth_by_view: dict[str, np.ndarray],
    normal_by_view: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """tangent view별 depth/normal을 source fisheye ray 위치로 융합한다.

    tangent depth는 tangent view z-depth이므로 radial depth로 변환한 뒤 같은 source
    ray의 z 성분을 곱해 source camera z-depth로 복원한다.
    """

    h, w, _ = source_rays.shape
    depth_acc = np.zeros((h, w), dtype=np.float64)
    normal_acc = np.zeros((h, w, 3), dtype=np.float64)
    weight_acc = np.zeros((h, w), dtype=np.float64)
    flat_source = source_rays.reshape(-1, 3).astype(np.float64)
    for view in views:
        rotation = view.rotation_tangent_to_source.astype(np.float64)
        source_to_tangent = rotation.T
        rays_tangent = flat_source @ source_to_tangent.T
        rz = rays_tangent[:, 2]
        valid = rz > 1.0e-5
        uv = np.full((len(flat_source), 2), np.nan, dtype=np.float64)
        uv[valid, 0] = view.intrinsics[0, 0] * (rays_tangent[valid, 0] / rz[valid]) + view.intrinsics[0, 2]
        uv[valid, 1] = view.intrinsics[1, 1] * (rays_tangent[valid, 1] / rz[valid]) + view.intrinsics[1, 2]
        inside = valid & (uv[:, 0] >= 0.0) & (uv[:, 0] < view.rgb.shape[1] - 1) & (uv[:, 1] >= 0.0) & (uv[:, 1] < view.rgb.shape[0] - 1)
        if not np.any(inside):
            continue
        map_x = (uv[:, 0].reshape(h, w) - 0.5).astype(np.float32)
        map_y = (uv[:, 1].reshape(h, w) - 0.5).astype(np.float32)
        d_tan = cv2.remap(depth_by_view[view.name].astype(np.float32), map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
        n_tan = cv2.remap(normal_by_view[view.name].astype(np.float32), map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        ray_tan_z = rays_tangent[:, 2].reshape(h, w)
        source_z = np.divide(d_tan, ray_tan_z, out=np.full_like(d_tan, np.nan), where=ray_tan_z > 1.0e-5) * source_rays[..., 2]
        n_source = n_tan @ rotation.T
        n_source /= np.linalg.norm(n_source, axis=-1, keepdims=True).clip(min=1.0e-6)
        weight = np.maximum(ray_tan_z.reshape(h, w), 0.0)
        valid_map = inside.reshape(h, w) & np.isfinite(source_z) & (source_z > 0.0) & np.isfinite(n_source).all(axis=-1)
        depth_acc[valid_map] += source_z[valid_map] * weight[valid_map]
        normal_acc[valid_map] += n_source[valid_map] * weight[valid_map, None]
        weight_acc[valid_map] += weight[valid_map]
    valid_out = weight_acc > 1.0e-6
    depth = np.divide(depth_acc, weight_acc, out=np.full_like(depth_acc, np.nan), where=valid_out).astype(np.float32)
    normal = np.divide(normal_acc, weight_acc[..., None], out=np.zeros_like(normal_acc), where=valid_out[..., None])
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True).clip(min=1.0e-6)
    normal[~valid_out] = np.nan
    return depth, normal.astype(np.float32), valid_out


def fuse_tangent_normals(
    source_rays: np.ndarray,
    views: list[TangentView],
    normal_by_view: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """5개 tangent DSINE normal을 source fisheye 좌표계로 통합한다.

    외부 depth는 이미 source image grid의 z-depth이므로 tangent view에서
    depth를 다시 추론할 필요가 없다. 이 함수는 DSINE normal만 회전·융합하여
    DA-V2를 실행하지 않는 external-depth tangent branch를 구성한다.

    Returns:
        normal: ``(H,W,3) float32`` source-camera normal. invalid은 NaN.
        valid: ``(H,W) bool`` tangent view 하나 이상에서 관측된 위치.
    """

    h, w, _ = source_rays.shape
    normal_acc = np.zeros((h, w, 3), dtype=np.float64)
    weight_acc = np.zeros((h, w), dtype=np.float64)
    flat_source = source_rays.reshape(-1, 3).astype(np.float64)
    for view in views:
        rotation = view.rotation_tangent_to_source.astype(np.float64)
        rays_tangent = flat_source @ rotation
        ray_tan_z = rays_tangent[:, 2]
        projected = ray_tan_z > 1.0e-5
        uv = np.full((len(flat_source), 2), np.nan, dtype=np.float64)
        uv[projected, 0] = (
            view.intrinsics[0, 0] * (rays_tangent[projected, 0] / ray_tan_z[projected])
            + view.intrinsics[0, 2]
        )
        uv[projected, 1] = (
            view.intrinsics[1, 1] * (rays_tangent[projected, 1] / ray_tan_z[projected])
            + view.intrinsics[1, 2]
        )
        inside = (
            projected
            & (uv[:, 0] >= 0.0)
            & (uv[:, 0] < view.rgb.shape[1] - 1)
            & (uv[:, 1] >= 0.0)
            & (uv[:, 1] < view.rgb.shape[0] - 1)
        )
        if not np.any(inside):
            continue
        map_x = (uv[:, 0].reshape(h, w) - 0.5).astype(np.float32)
        map_y = (uv[:, 1].reshape(h, w) - 0.5).astype(np.float32)
        n_tangent = cv2.remap(
            normal_by_view[view.name].astype(np.float32),
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        n_source = n_tangent @ rotation.T
        n_source /= np.linalg.norm(n_source, axis=-1, keepdims=True).clip(min=1.0e-6)
        weight = np.maximum(ray_tan_z.reshape(h, w), 0.0)
        valid_map = inside.reshape(h, w) & np.isfinite(n_source).all(axis=-1)
        normal_acc[valid_map] += n_source[valid_map] * weight[valid_map, None]
        weight_acc[valid_map] += weight[valid_map]

    valid = weight_acc > 1.0e-6
    normal = np.divide(
        normal_acc,
        weight_acc[..., None],
        out=np.zeros_like(normal_acc),
        where=valid[..., None],
    )
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True).clip(min=1.0e-6)
    normal[~valid] = np.nan
    return normal.astype(np.float32), valid
