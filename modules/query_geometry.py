from __future__ import annotations

import torch
import torch.nn.functional as F


def normalize_vector(value: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """마지막 축 vector를 단위화한다."""

    dtype_eps = torch.finfo(value.dtype).eps if value.dtype.is_floating_point else eps
    return value / value.norm(dim=-1, keepdim=True).clamp_min(max(float(eps), float(dtype_eps)))


def pixel_uv_to_grid(uv: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """pixel 좌표 `(u, v)`를 `grid_sample`의 `[-1, 1]` 좌표로 바꾼다.

    Args:
        uv: `(B, Q, 2)` pixel coordinate. `u`는 x/width 방향, `v`는 y/height 방향.
    Returns:
        `(B, Q, 1, 2)` grid. `align_corners=True` 기준이다.
    """

    # 이 프로젝트의 ray/source_uv는 pixel-center를 0.5..W-0.5로 둔다.
    # grid_sample의 align_corners=True 좌표계에서는 첫/마지막 pixel center가
    # -1/+1이므로 0.5 offset을 빼고 정규화한다.
    x = (uv[..., 0] - 0.5) / max(width - 1, 1) * 2.0 - 1.0
    y = (uv[..., 1] - 0.5) / max(height - 1, 1) * 2.0 - 1.0
    return torch.stack([x, y], dim=-1).unsqueeze(2)


def sample_at_uv(feature: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """dense feature map에서 query pixel 좌표를 bilinear sampling한다.

    Args:
        feature: `(B, C, H, W)` dense map.
        uv: `(B, Q, 2)` pixel 좌표.

    Returns:
        `(B, Q, C)` sampled feature.
    """

    b, c, h, w = feature.shape
    if uv.dim() != 3 or uv.shape[0] != b or uv.shape[-1] != 2:
        raise ValueError(f"uv must be (B,Q,2), got {tuple(uv.shape)} for feature {tuple(feature.shape)}")
    grid = pixel_uv_to_grid(uv, h, w)
    sampled = F.grid_sample(feature, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return sampled.squeeze(-1).transpose(1, 2)


def query_stencil_rays(query_rays: torch.Tensor, step_rad: float) -> torch.Tensor:
    """query마다 local tangent 방향의 left/right/up/down stencil ray를 만든다.

    Args:
        query_rays: `(B, Q, 3)` OpenCV camera-frame 단위 ray.
        step_rad: 중심 ray에서 이동할 작은 각도(rad).

    Returns:
        `(B, Q, 4, 3)` stencil rays. 순서는 left, right, up, down이다.
    """

    r = normalize_vector(query_rays)
    ex = torch.tensor([1.0, 0.0, 0.0], dtype=r.dtype, device=r.device)
    ey = torch.tensor([0.0, 1.0, 0.0], dtype=r.dtype, device=r.device)
    ex = ex.view(1, 1, 3)
    ey = ey.view(1, 1, 3)

    tangent_x = ex - (ex * r).sum(dim=-1, keepdim=True) * r
    fallback_x = ey - (ey * r).sum(dim=-1, keepdim=True) * r
    use_fallback = tangent_x.norm(dim=-1, keepdim=True) < 1.0e-5
    tangent_x = normalize_vector(torch.where(use_fallback, fallback_x, tangent_x))
    tangent_y = normalize_vector(torch.cross(r, tangent_x, dim=-1))

    cos_s = torch.cos(torch.tensor(float(step_rad), dtype=r.dtype, device=r.device))
    sin_s = torch.sin(torch.tensor(float(step_rad), dtype=r.dtype, device=r.device))
    left = normalize_vector(cos_s * r - sin_s * tangent_x)
    right = normalize_vector(cos_s * r + sin_s * tangent_x)
    up = normalize_vector(cos_s * r - sin_s * tangent_y)
    down = normalize_vector(cos_s * r + sin_s * tangent_y)
    return torch.stack([left, right, up, down], dim=2)


def z_depth_to_points(depth_z: torch.Tensor, rays: torch.Tensor, z_eps: float = 1.0e-3) -> tuple[torch.Tensor, torch.Tensor]:
    """z-depth를 ray 방향의 3D camera point로 변환한다.

    Formula:
        radial_depth = depth_z / ray_z
        P_camera = radial_depth * ray

    Args:
        depth_z: `(B, Q)` 또는 `(B, Q, K)` z-depth.
        rays: `(..., 3)` depth와 prefix shape가 같은 단위 ray.

    Returns:
        points: `(..., 3)` camera-frame point. invalid 위치는 autograd 안전한 0.
        valid: `(...)` bool mask. `ray_z > z_eps`와 finite depth를 모두 만족해야 한다.
    """

    rz = rays[..., 2]
    valid = torch.isfinite(depth_z) & (depth_z > 0.0) & torch.isfinite(rays).all(dim=-1) & (rz > z_eps)
    # invalid query를 NaN point로 만든 뒤 mask-out하면 backward에서 0*NaN이 생길 수
    # 있다. 계산 graph 안에서는 0으로 채우고 유효 여부는 별도 bool mask로 전달한다.
    safe_depth = torch.nan_to_num(depth_z, nan=0.0, posinf=0.0, neginf=0.0)
    safe_rays = torch.nan_to_num(rays, nan=0.0, posinf=0.0, neginf=0.0)
    radial = safe_depth / safe_rays[..., 2].clamp_min(z_eps)
    points = radial.unsqueeze(-1) * safe_rays
    points = torch.where(valid.unsqueeze(-1), points, torch.zeros_like(points))
    return points, valid


def normals_from_stencil_depths(
    center_depth_z: torch.Tensor,
    stencil_depth_z: torch.Tensor,
    center_rays: torch.Tensor,
    stencil_rays: torch.Tensor,
    *,
    z_eps: float = 1.0e-3,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """불규칙 query의 depth-derived normal `N*`를 계산한다.

    Args:
        center_depth_z: `(B, Q)` 중심 query z-depth.
        stencil_depth_z: `(B, Q, 4)` left/right/up/down stencil z-depth.
        center_rays: `(B, Q, 3)` 중심 ray.
        stencil_rays: `(B, Q, 4, 3)` stencil ray.

    Returns:
        normals: `(B, Q, 3)` OpenCV camera-frame normal.
        valid: `(B, Q)` 중심과 네 stencil depth/ray가 모두 valid한지 여부.
    """

    # AMP half precision에서는 거의 퇴화된 stencil의 cross product가 0에 가까워져
    # normalize backward가 NaN/Inf를 만들 수 있다. 보조 normal loss의 geometry는
    # float32로 계산하고, 길이가 작은 법선은 loss mask 밖으로 뺀다.
    center_depth_z = center_depth_z.float()
    stencil_depth_z = stencil_depth_z.float()
    center_rays = center_rays.float()
    stencil_rays = stencil_rays.float()

    _, center_valid = z_depth_to_points(center_depth_z, center_rays, z_eps=z_eps)
    stencil_points, stencil_valid = z_depth_to_points(stencil_depth_z, stencil_rays, z_eps=z_eps)
    p_left = stencil_points[:, :, 0]
    p_right = stencil_points[:, :, 1]
    p_up = stencil_points[:, :, 2]
    p_down = stencil_points[:, :, 3]
    tangent_x = p_right - p_left
    tangent_y = p_down - p_up
    normals_raw = torch.cross(tangent_x, tangent_y, dim=-1)
    normal_norm = normals_raw.norm(dim=-1)
    normal_eps = max(float(eps), 1.0e-6)
    normals = normals_raw / normal_norm.clamp_min(normal_eps).unsqueeze(-1)
    valid = center_valid & stencil_valid.all(dim=-1) & (normal_norm > normal_eps) & torch.isfinite(normals).all(dim=-1)

    # normal 방향은 camera-facing으로 맞춘다. ray는 camera->surface이므로
    # normal dot ray가 양수면 카메라 반대쪽을 향하는 것으로 보고 뒤집는다.
    facing = (normals * center_rays).sum(dim=-1, keepdim=True) <= 0.0
    normals = torch.where(facing, normals, -normals)
    return normals, valid


def dense_normals_from_depth(depth_z: torch.Tensor, rays: torch.Tensor, valid: torch.Tensor, z_eps: float = 1.0e-3) -> tuple[torch.Tensor, torch.Tensor]:
    """정규 image-grid depth에서 finite-difference normal을 계산한다.

    Inference에서 source pixel의 final normal을 만들거나 analytic test에 사용할 수 있다.
    입력 shape는 `depth_z=(B,1,H,W)`, `rays=(B,3,H,W)`, `valid=(B,1,H,W)`이다.
    """

    if depth_z.dim() != 4 or rays.dim() != 4:
        raise ValueError("depth_z and rays must be BCHW tensors.")
    rays_hwc = rays.permute(0, 2, 3, 1)
    depth_hw = depth_z[:, 0]
    points, point_valid = z_depth_to_points(depth_hw, rays_hwc, z_eps=z_eps)
    center_valid = point_valid[:, 1:-1, 1:-1] & valid[:, 0, 1:-1, 1:-1].bool()
    neighbor_valid = (
        center_valid
        & point_valid[:, 1:-1, :-2]
        & point_valid[:, 1:-1, 2:]
        & point_valid[:, :-2, 1:-1]
        & point_valid[:, 2:, 1:-1]
    )
    tx = points[:, 1:-1, 2:] - points[:, 1:-1, :-2]
    ty = points[:, 2:, 1:-1] - points[:, :-2, 1:-1]
    n = normalize_vector(torch.cross(tx, ty, dim=-1))
    center_rays = rays_hwc[:, 1:-1, 1:-1]
    facing = (n * center_rays).sum(dim=-1, keepdim=True) <= 0.0
    n = torch.where(facing, n, -n)

    out = torch.zeros_like(rays_hwc)
    out_valid = torch.zeros_like(depth_hw, dtype=torch.bool)
    valid_n = neighbor_valid & torch.isfinite(n).all(dim=-1)
    out[:, 1:-1, 1:-1] = torch.where(valid_n.unsqueeze(-1), n, torch.zeros_like(n))
    out_valid[:, 1:-1, 1:-1] = valid_n
    return out.permute(0, 3, 1, 2), out_valid.unsqueeze(1)
