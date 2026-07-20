from __future__ import annotations

import torch


def graph_connected_zero(reference: torch.Tensor) -> torch.Tensor:
    """mask가 비었을 때도 autograd graph에 연결된 scalar zero를 반환한다."""

    return torch.nan_to_num(reference, nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0


def masked_mean(value: torch.Tensor, mask: torch.Tensor, reference: torch.Tensor | None = None) -> torch.Tensor:
    """bool mask 평균. mask가 비면 graph-connected zero."""

    mask = mask.bool()
    if not torch.any(mask):
        return graph_connected_zero(reference if reference is not None else value)
    return value[mask].mean()


def masked_log_depth_loss(pred_depth_z: torch.Tensor, target_depth_z: torch.Tensor, mask: torch.Tensor, eps: float = 1.0e-4) -> torch.Tensor:
    """masked log-depth L1 loss.

    depth scale error를 직접 depth 차이보다 안정적으로 다루기 위해 log 공간에서
    `|log(D*) - log(D_gt)|`를 계산한다.
    """

    valid = mask.bool() & torch.isfinite(pred_depth_z) & torch.isfinite(target_depth_z) & (pred_depth_z > eps) & (target_depth_z > eps)
    safe_pred = torch.where(valid, pred_depth_z, torch.ones_like(pred_depth_z))
    safe_target = torch.where(valid, target_depth_z, torch.ones_like(target_depth_z))
    value = (safe_pred.clamp_min(eps).log() - safe_target.clamp_min(eps).log()).abs()
    return masked_mean(value, valid, pred_depth_z)


def radial_depth_consistency_loss(pred_depth_z: torch.Tensor, base_depth_z: torch.Tensor, query_ray_z: torch.Tensor, source_ray_z: torch.Tensor, mask: torch.Tensor, eps: float = 1.0e-4) -> torch.Tensor:
    """source/query radial-depth consistency loss.

    같은 source 위치에서 bilinear sampling한 D0와 query D*가 서로 완전히 독립으로
    튀지 않게 radial depth `D_z / ray_z` 기준의 약한 제약을 준다.
    """

    valid = (
        mask.bool()
        & torch.isfinite(pred_depth_z)
        & torch.isfinite(base_depth_z)
        & torch.isfinite(query_ray_z)
        & torch.isfinite(source_ray_z)
        & (query_ray_z > eps)
        & (source_ray_z > eps)
        & (pred_depth_z > eps)
        & (base_depth_z > eps)
    )
    safe_pred = torch.where(valid, pred_depth_z, torch.ones_like(pred_depth_z))
    safe_base = torch.where(valid, base_depth_z, torch.ones_like(base_depth_z))
    safe_query_z = torch.where(valid, query_ray_z, torch.ones_like(query_ray_z))
    safe_source_z = torch.where(valid, source_ray_z, torch.ones_like(source_ray_z))
    pred_radial = safe_pred / safe_query_z.clamp_min(eps)
    base_radial = safe_base / safe_source_z.clamp_min(eps)
    value = (pred_radial.clamp_min(eps).log() - base_radial.clamp_min(eps).log()).abs()
    return masked_mean(value, valid, pred_depth_z)


def delta_log_depth_regularization(delta_log_depth: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """refiner residual이 불필요하게 커지는 것을 막는 L2 regularization."""

    if mask is None:
        return (delta_log_depth**2).mean()
    return masked_mean(delta_log_depth**2, mask.bool() & torch.isfinite(delta_log_depth), delta_log_depth)
