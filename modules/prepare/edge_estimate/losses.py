"""Coarse/subpixel edge, type, near/far depth, confidence와 contour 일관성 손실."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .config import EdgeLossConfig
from .model import EdgeEstimateResult


@dataclass
class EdgeLossResult:
    total: torch.Tensor
    coarse: torch.Tensor
    query_focal: torch.Tensor
    query_dice: torch.Tensor
    edge_type: torch.Tensor
    near_depth: torch.Tensor
    far_depth: torch.Tensor
    confidence: torch.Tensor
    bev_keep: torch.Tensor
    boundary_consistency: torch.Tensor
    tangent: torch.Tensor


def _masked_mean(value: torch.Tensor, mask: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    good = mask.bool() & torch.isfinite(value)
    if not torch.any(good):
        return anchor.sum() * 0.0
    # NaN이 있는 전체 tensor를 먼저 연산하고 마지막에 indexing하면 일부 autograd
    # 연산에서 ``0 * NaN`` gradient가 남을 수 있다. invalid 원소를 graph에서 명시적으로
    # 0으로 치환한 뒤 valid 개수로 나누어 mask 밖 값이 역전파에 영향을 주지 않게 한다.
    safe = torch.where(good, value, torch.zeros_like(value))
    return safe.sum() / good.to(value.dtype).sum().clamp_min(1.0)


def _focal_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    alpha: float,
    gamma: float,
) -> torch.Tensor:
    safe_target = torch.nan_to_num(target).clamp(0.0, 1.0)
    bce = F.binary_cross_entropy_with_logits(logits, safe_target, reduction="none")
    probability = torch.sigmoid(logits)
    pt = safe_target * probability + (1.0 - safe_target) * (1.0 - probability)
    alpha_t = safe_target * alpha + (1.0 - safe_target) * (1.0 - alpha)
    return _masked_mean(alpha_t * (1.0 - pt).pow(gamma) * bce, mask, logits)


def _dice_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    valid = mask.to(probability.dtype)
    target = torch.nan_to_num(target).clamp(0.0, 1.0)
    intersection = torch.sum(probability * target * valid, dim=(-1, -2, -3))
    denominator = torch.sum((probability + target) * valid, dim=(-1, -2, -3))
    dice = 1.0 - (2.0 * intersection + eps) / (denominator + eps)
    return dice.mean()


def _log_depth_huber(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    delta: float,
    *,
    scale_invariant: bool = False,
) -> torch.Tensor:
    safe_prediction = torch.log(torch.nan_to_num(prediction, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(1.0e-6))
    safe_target = torch.log(torch.nan_to_num(target, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(1.0e-6))
    good = mask.bool() & (target > 0.0) & torch.isfinite(safe_prediction) & torch.isfinite(safe_target)
    error = safe_prediction - safe_target
    if scale_invariant and torch.any(good):
        good_float = good.to(error.dtype)
        reduce_dims = tuple(range(1, error.ndim))
        mean_error = (error * good_float).sum(dim=reduce_dims, keepdim=True) / good_float.sum(
            dim=reduce_dims, keepdim=True
        ).clamp_min(1.0)
        error = error - mean_error
    loss = F.huber_loss(error, torch.zeros_like(error), reduction="none", delta=delta)
    return _masked_mean(loss, good, prediction)


def _shared_boundary_consistency(result: EdgeEstimateResult, query_mask: torch.Tensor) -> torch.Tensor:
    """인접 cell의 맞닿는 query strip이 같은 edge/깊이를 예측하도록 한다."""

    probability = torch.sigmoid(result.query_edge_logit)
    q = probability.shape[-1]
    side = int(round(math.sqrt(q)))
    if side * side != q:
        return probability.sum() * 0.0
    p = probability.reshape(*probability.shape[:-1], side, side)
    d = torch.log(result.query_depth_near_z.clamp_min(1.0e-6)).reshape(*probability.shape[:-1], side, side)
    m = query_mask.reshape(*query_mask.shape[:-1], side, side)
    horizontal_mask = m[:, :, :-1, :, -1] & m[:, :, 1:, :, 0]
    vertical_mask = m[:, :-1, :, -1, :] & m[:, 1:, :, 0, :]
    horizontal = torch.abs(p[:, :, :-1, :, -1] - p[:, :, 1:, :, 0])
    vertical = torch.abs(p[:, :-1, :, -1, :] - p[:, 1:, :, 0, :])
    depth_horizontal = torch.abs(d[:, :, :-1, :, -1] - d[:, :, 1:, :, 0])
    depth_vertical = torch.abs(d[:, :-1, :, -1, :] - d[:, 1:, :, 0, :])
    edge_h = (p[:, :, :-1, :, -1] > 0.5) | (p[:, :, 1:, :, 0] > 0.5)
    edge_v = (p[:, :-1, :, -1, :] > 0.5) | (p[:, 1:, :, 0, :] > 0.5)
    anchor = result.query_edge_logit
    return (
        _masked_mean(horizontal, horizontal_mask, anchor)
        + _masked_mean(vertical, vertical_mask, anchor)
        + 0.25 * _masked_mean(depth_horizontal, horizontal_mask & edge_h, anchor)
        + 0.25 * _masked_mean(depth_vertical, vertical_mask & edge_v, anchor)
    ) * 0.5


def _points(depth: torch.Tensor, rays: torch.Tensor) -> torch.Tensor:
    radial = depth / rays[..., 2].clamp_min(1.0e-6)
    return radial.unsqueeze(-1) * rays


def _tangent_loss(
    result: EdgeEstimateResult,
    target_depth: torch.Tensor,
    target_edge: torch.Tensor,
    target_valid: torch.Tensor,
    query_rays: torch.Tensor,
) -> torch.Tensor:
    q = target_edge.shape[-1]
    side = int(round(math.sqrt(q)))
    if side * side != q:
        return result.query_edge_logit.sum() * 0.0
    pred = _points(result.query_depth_near_z, query_rays).reshape(*target_edge.shape[:-1], side, side, 3)
    safe_target_depth = torch.nan_to_num(target_depth, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(1.0e-6)
    target = _points(safe_target_depth, query_rays).reshape(*target_edge.shape[:-1], side, side, 3)
    edge = (target_edge >= 0.5).reshape(*target_edge.shape[:-1], side, side)
    valid = target_valid.reshape(*target_valid.shape[:-1], side, side) & edge
    losses: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for axis in (-3, -2):
        pred_diff = torch.diff(pred, dim=axis)
        target_diff = torch.diff(target, dim=axis)
        pred_norm = F.normalize(pred_diff, dim=-1, eps=1.0e-6)
        target_norm = F.normalize(target_diff, dim=-1, eps=1.0e-6)
        losses.append(1.0 - torch.abs(torch.sum(pred_norm * target_norm, dim=-1)))
        masks.append(valid[..., 1:, :] & valid[..., :-1, :] if axis == -3 else valid[..., :, 1:] & valid[..., :, :-1])
    return 0.5 * (
        _masked_mean(losses[0], masks[0], result.query_edge_logit)
        + _masked_mean(losses[1], masks[1], result.query_edge_logit)
    )


class EdgeEstimateLoss:
    def __init__(self, config: EdgeLossConfig) -> None:
        self.config = config

    def __call__(self, result: EdgeEstimateResult, batch: dict[str, torch.Tensor]) -> EdgeLossResult:
        """Cache target의 독립 mask를 적용하고 빈 mask에는 graph-connected zero를 반환한다."""
        cell_valid = batch["cell_valid"].bool()
        query_mask = batch["query_mask"].bool()
        coarse = _focal_bce(
            result.cell_edge_logit,
            batch["target_cell_edge"],
            cell_valid,
            alpha=self.config.focal_alpha,
            gamma=self.config.focal_gamma,
        )
        query_focal = _focal_bce(
            result.query_edge_logit,
            batch["target_query_edge"],
            query_mask,
            alpha=self.config.focal_alpha,
            gamma=self.config.focal_gamma,
        )
        query_dice = _dice_loss(result.query_edge_logit, batch["target_query_edge"], query_mask, self.config.eps)
        type_target = batch["target_cell_type"].long()
        type_mask = cell_valid & (type_target > 0)
        type_loss_raw = F.cross_entropy(
            result.cell_type_logits.reshape(-1, 3),
            (type_target.clamp_min(1) - 1).reshape(-1),
            reduction="none",
        ).reshape_as(type_target)
        edge_type = _masked_mean(type_loss_raw, type_mask, result.cell_type_logits)
        near_depth = _log_depth_huber(
            result.query_depth_near_z,
            batch["target_near_depth_z"],
            batch["target_near_valid"].bool() & query_mask,
            self.config.depth_huber_delta,
            scale_invariant=self.config.scale_invariant_depth,
        )
        far_depth = _log_depth_huber(
            result.query_depth_far_z,
            batch["target_far_depth_z"],
            batch["target_far_valid"].bool() & query_mask,
            self.config.depth_huber_delta,
            scale_invariant=self.config.scale_invariant_depth,
        )
        confidence = _masked_mean(
            F.binary_cross_entropy_with_logits(
                result.query_confidence_logit,
                batch["target_confidence"].clamp(0.0, 1.0),
                reduction="none",
            ),
            query_mask,
            result.query_confidence_logit,
        )
        bev_keep = _focal_bce(
            result.query_bev_keep_logit,
            batch["target_bev_keep"],
            batch["target_bev_keep_valid"].bool() & query_mask,
            alpha=self.config.focal_alpha,
            gamma=self.config.focal_gamma,
        )
        consistency = _shared_boundary_consistency(result, query_mask)
        tangent = _tangent_loss(
            result,
            batch["target_near_depth_z"],
            batch["target_query_edge"],
            batch["target_near_valid"].bool() & query_mask,
            batch["query_ray_dir"],
        )
        total = (
            self.config.coarse_focal_weight * coarse
            + self.config.query_focal_weight * query_focal
            + self.config.query_dice_weight * query_dice
            + self.config.type_weight * edge_type
            + self.config.near_depth_weight * near_depth
            + self.config.far_depth_weight * far_depth
            + self.config.confidence_weight * confidence
            + self.config.bev_keep_weight * bev_keep
            + self.config.boundary_consistency_weight * consistency
            + self.config.tangent_weight * tangent
        )
        return EdgeLossResult(
            total=total,
            coarse=coarse,
            query_focal=query_focal,
            query_dice=query_dice,
            edge_type=edge_type,
            near_depth=near_depth,
            far_depth=far_depth,
            confidence=confidence,
            bev_keep=bev_keep,
            boundary_consistency=consistency,
            tangent=tangent,
        )
