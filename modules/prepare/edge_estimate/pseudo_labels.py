"""Dense/Raw NYU depth만으로 continuous·crease·occlusion·junction pseudo-GT를 만든다."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import EdgeDataConfig


EDGE_NONE = 0
EDGE_CREASE = 1
EDGE_OCCLUSION = 2
EDGE_JUNCTION = 3


@dataclass(frozen=True)
class EdgePseudoLabels:
    """dense RGB-D에서 자동으로 만든 3D edge 감독 map.

    ``edge_soft``는 query grid가 1-pixel centerline을 놓치지 않도록 distance
    transform으로 확장한 soft target이다. ``ignore``는 센서 결측 또는 scale별
    판정 불일치 영역이며 loss에서 제외한다.
    """

    edge: np.ndarray
    edge_soft: np.ndarray
    edge_type: np.ndarray
    confidence: np.ndarray
    ignore: np.ndarray
    near_depth_z: np.ndarray
    far_depth_z: np.ndarray
    normals: np.ndarray
    normal_valid: np.ndarray


def _shift(array: np.ndarray, dy: int, dx: int, fill: float | bool = np.nan) -> np.ndarray:
    result = np.full_like(array, fill)
    h, w = array.shape[:2]
    src_y0, src_y1 = max(0, -dy), min(h, h - dy)
    src_x0, src_x1 = max(0, -dx), min(w, w - dx)
    dst_y0, dst_y1 = max(0, dy), min(h, h + dy)
    dst_x0, dst_x1 = max(0, dx), min(w, w + dx)
    result[dst_y0:dst_y1, dst_x0:dst_x1] = array[src_y0:src_y1, src_x0:src_x1]
    return result


def depth_normals(depth_z: np.ndarray, rays: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """z-depth와 ray로 camera-frame normal을 계산한다.

    좌우·상하 두 tangent의 cross product를 사용하며, 계산된 normal 방향은 camera를
    향하도록 정렬한다. 이 normal은 학습 target이 아니라 crease pseudo-label에만 쓴다.
    """

    depth = np.asarray(depth_z, dtype=np.float32)
    rays_f = np.asarray(rays, dtype=np.float32)
    good = np.asarray(valid, dtype=bool) & np.isfinite(depth) & (depth > 0.0) & (rays_f[..., 2] > 1.0e-6)
    radial = np.divide(depth, rays_f[..., 2], out=np.full_like(depth, np.nan), where=good)
    points = radial[..., None] * rays_f
    left, right = _shift(points, 0, 1), _shift(points, 0, -1)
    up, down = _shift(points, 1, 0), _shift(points, -1, 0)
    tangent_x = right - left
    tangent_y = down - up
    normal = np.cross(tangent_x, tangent_y)
    norm = np.linalg.norm(normal, axis=-1)
    normal_valid = good & _shift(good, 0, 1, False) & _shift(good, 0, -1, False)
    normal_valid &= _shift(good, 1, 0, False) & _shift(good, -1, 0, False) & np.isfinite(norm) & (norm > 1.0e-8)
    normal = np.divide(normal, norm[..., None], out=np.zeros_like(normal), where=normal_valid[..., None])
    facing = np.sum(normal * rays_f, axis=-1) > 0.0
    normal[facing] *= -1.0
    normal[~normal_valid] = np.nan
    return normal.astype(np.float32), normal_valid


def _max_neighbor_difference(value: np.ndarray, valid: np.ndarray, step: int) -> tuple[np.ndarray, np.ndarray]:
    differences: list[np.ndarray] = []
    pair_valids: list[np.ndarray] = []
    for dy, dx in ((0, step), (0, -step), (step, 0), (-step, 0)):
        other = _shift(value, dy, dx)
        other_valid = _shift(valid, dy, dx, False)
        pair_valid = valid & other_valid
        differences.append(np.where(pair_valid, np.abs(value - other), 0.0))
        pair_valids.append(pair_valid)
    return np.max(np.stack(differences), axis=0), np.any(np.stack(pair_valids), axis=0)


def _crease_cluster_evidence(
    normal: np.ndarray,
    valid: np.ndarray,
    step: int,
    crease_degrees: float,
    within_cluster_degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    """한쪽에는 다른 normal cluster가 있고 같은 면 이웃도 남아 있는지 검사한다.

    매 픽셀마다 4방향 이웃 normal 각도를 모은다. 최소 한 방향은 crease 각도보다
    커야 하고, 최소 두 방향은 같은 면으로 볼 수 있을 만큼 작아야 한다. 이 조건은
    단순 센서 잡음 때문에 모든 이웃 normal이 흔들리는 지점을 crease로 쓰는 것을
    억제하는 저비용 two-cluster 근사다.
    """

    angles: list[np.ndarray] = []
    pair_valids: list[np.ndarray] = []
    for dy, dx in ((0, step), (0, -step), (step, 0), (-step, 0)):
        other = _shift(normal, dy, dx)
        other_valid = _shift(valid, dy, dx, False)
        pair_valid = valid & other_valid
        cosine = np.sum(normal * other, axis=-1)
        angle = np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))
        angles.append(np.where(pair_valid, angle, np.nan))
        pair_valids.append(pair_valid)
    angle_stack = np.stack(angles)
    valid_stack = np.stack(pair_valids)
    high = np.any(valid_stack & (np.nan_to_num(angle_stack, nan=-1.0) > crease_degrees), axis=0)
    same_surface_count = np.sum(
        valid_stack & (np.nan_to_num(angle_stack, nan=np.inf) < within_cluster_degrees), axis=0
    )
    enough_pairs = np.sum(valid_stack, axis=0) >= 3
    return high & (same_surface_count >= 2) & enough_pairs, enough_pairs


def build_pseudo_edges(
    depth_z: np.ndarray,
    raw_valid: np.ndarray,
    rays: np.ndarray,
    config: EdgeDataConfig,
) -> EdgePseudoLabels:
    """RGB-D annotation 없이 depth discontinuity와 crease pseudo-GT를 만든다."""

    depth = np.asarray(depth_z, dtype=np.float32)
    ray_map = np.asarray(rays, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0) & np.isfinite(ray_map).all(axis=-1) & (ray_map[..., 2] > 1.0e-6)
    raw = np.asarray(raw_valid, dtype=bool) & valid
    log_depth = np.log(np.clip(depth, 1.0e-6, None))
    normals, normal_valid = depth_normals(depth, ray_map, valid)

    occlusion_votes = np.zeros(depth.shape, dtype=np.int16)
    crease_votes = np.zeros(depth.shape, dtype=np.int16)
    for step in config.stability_scales_px:
        depth_jump, depth_pair_valid = _max_neighbor_difference(log_depth, valid, int(step))
        crease_evidence, normal_pair_valid = _crease_cluster_evidence(
            normals,
            normal_valid,
            int(step),
            config.crease_normal_degrees,
            config.normal_cluster_degrees,
        )
        occlusion_votes += (depth_pair_valid & (depth_jump > config.depth_jump_log_threshold)).astype(np.int16)
        crease_votes += (
            normal_pair_valid
            & (depth_jump <= config.depth_jump_log_threshold)
            & crease_evidence
        ).astype(np.int16)

    occlusion = occlusion_votes >= int(config.stability_min_count)
    crease = (crease_votes >= int(config.stability_min_count)) & ~occlusion
    raw_fraction = cv2.boxFilter(raw.astype(np.float32), -1, (5, 5), normalize=True)
    reliable = valid & (raw_fraction >= float(config.raw_valid_fraction))
    stable = (occlusion_votes >= config.stability_min_count) | (crease_votes >= config.stability_min_count)
    unstable_votes = ((occlusion_votes > 0) | (crease_votes > 0)) & ~stable
    ignore = ~reliable | unstable_votes
    occlusion &= ~ignore
    crease &= ~ignore

    # 직선 edge의 두꺼운 band를 junction으로 오인하지 않도록 edge map의 2D corner
    # response가 실제로 존재하는 곳만 junction 후보로 사용한다.
    edge_union = occlusion | crease
    corner_response = cv2.cornerHarris(edge_union.astype(np.float32), blockSize=3, ksize=3, k=0.04)
    response_max = float(np.max(corner_response)) if corner_response.size else 0.0
    if response_max > 1.0e-8:
        junction_region = cv2.dilate(
            (corner_response > 0.05 * response_max).astype(np.uint8), np.ones((3, 3), np.uint8)
        ) > 0
    else:
        junction_region = np.zeros_like(edge_union)
    junction = edge_union & junction_region
    edge_type = np.full(depth.shape, EDGE_NONE, dtype=np.uint8)
    edge_type[crease] = EDGE_CREASE
    edge_type[occlusion] = EDGE_OCCLUSION
    edge_type[junction] = EDGE_JUNCTION
    edge = (edge_type > EDGE_NONE) & ~ignore

    # centerline 주변의 near/far 후보는 네 cardinal neighbor 중 최솟값/최댓값이다.
    neighbor_depth = np.stack(
        [_shift(depth, 0, 1), _shift(depth, 0, -1), _shift(depth, 1, 0), _shift(depth, -1, 0)],
        axis=0,
    )
    valid_neighbor = np.isfinite(neighbor_depth) & (neighbor_depth > 0.0)
    near = np.min(np.where(valid_neighbor, neighbor_depth, np.inf), axis=0)
    far = np.max(np.where(valid_neighbor, neighbor_depth, -np.inf), axis=0)
    near[~np.isfinite(near)] = np.nan
    far[~np.isfinite(far)] = np.nan
    near[crease] = depth[crease]
    far[crease] = np.nan
    near[~edge] = np.nan
    far[(edge_type != EDGE_OCCLUSION) | ~edge] = np.nan

    if np.any(edge):
        inverse = (~edge).astype(np.uint8)
        distance = cv2.distanceTransform(inverse, cv2.DIST_L2, 3)
        sigma = max(float(config.edge_soft_sigma_px), 1.0e-3)
        scaled = np.minimum(distance.astype(np.float64) / sigma, 20.0)
        edge_soft = np.exp(-0.5 * scaled**2).astype(np.float32)
    else:
        edge_soft = np.zeros(depth.shape, dtype=np.float32)
    edge_soft[ignore] = 0.0
    confidence = np.clip(raw_fraction * stable.astype(np.float32), 0.0, 1.0)
    confidence[ignore] = 0.0
    return EdgePseudoLabels(
        edge=edge,
        edge_soft=edge_soft,
        edge_type=edge_type,
        confidence=confidence.astype(np.float32),
        ignore=ignore,
        near_depth_z=near.astype(np.float32),
        far_depth_z=far.astype(np.float32),
        normals=normals,
        normal_valid=normal_valid,
    )
