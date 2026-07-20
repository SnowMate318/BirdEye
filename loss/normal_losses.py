from __future__ import annotations

import torch

from wide_fov_supervision_v2.loss.depth_losses import masked_mean
from wide_fov_supervision_v2.modules.query_geometry import normalize_vector


def align_camera_facing(normals: torch.Tensor, rays: torch.Tensor) -> torch.Tensor:
    """normal 방향을 camera-facing convention으로 맞춘다.

    ray는 camera에서 surface로 향한다. 따라서 normal이 camera를 향하려면
    `normal dot ray <= 0`이어야 한다.
    """

    n = normalize_vector(normals)
    r = normalize_vector(rays)
    facing = (n * r).sum(dim=-1, keepdim=True) <= 0.0
    return torch.where(facing, n, -n)


def partial_normal_cosine_loss(pred_normal: torch.Tensor, target_normal: torch.Tensor, query_rays: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """DSINE prior normal과 depth-derived `N*` 사이 cosine loss."""

    valid = (
        mask.bool()
        & torch.isfinite(pred_normal).all(dim=-1)
        & torch.isfinite(target_normal).all(dim=-1)
        & torch.isfinite(query_rays).all(dim=-1)
    )
    # mask 밖의 NaN을 정규화한 뒤 masked mean에서 제외하면,
    # forward loss는 유한해도 normalize의 NaN gradient가 모델로 전파될 수 있다.
    # 따라서 정규화 전에 invalid 항목을 gradient가 0인 안전한 상수로 치환한다.
    fallback_normal = torch.zeros_like(pred_normal)
    fallback_normal[..., 2] = -1.0
    fallback_ray = torch.zeros_like(query_rays)
    fallback_ray[..., 2] = 1.0
    valid_vector = valid.unsqueeze(-1)
    safe_pred = torch.where(valid_vector, pred_normal, fallback_normal)
    safe_target = torch.where(valid_vector, target_normal, fallback_normal)
    safe_rays = torch.where(valid_vector, query_rays, fallback_ray)

    pred = align_camera_facing(safe_pred, safe_rays)
    target = align_camera_facing(safe_target, safe_rays)
    cosine = (pred * target).sum(dim=-1).clamp(-1.0, 1.0)
    return masked_mean(1.0 - cosine, valid, pred_normal)
