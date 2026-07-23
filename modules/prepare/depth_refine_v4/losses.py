from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .config import V4LossConfig
from .model import DepthRefineResult


@dataclass
class V4LossResult:
    total: torch.Tensor
    full_depth: torch.Tensor
    edge_depth: torch.Tensor
    gradient: torch.Tensor
    occlusion_jump: torch.Tensor
    contour_tangent: torch.Tensor
    non_edge_anchor: torch.Tensor
    scale_drift: torch.Tensor


def masked_mean(value: torch.Tensor, mask: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    good = mask.bool() & torch.isfinite(value)
    if not torch.any(good):
        return anchor.sum() * 0.0
    safe = torch.where(good, value, torch.zeros_like(value))
    return safe.sum() / good.to(value.dtype).sum().clamp_min(1.0)


def log_huber(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta: float) -> torch.Tensor:
    pred = torch.log(torch.nan_to_num(prediction, nan=1.0).clamp_min(1.0e-6))
    tgt = torch.log(torch.nan_to_num(target, nan=1.0).clamp_min(1.0e-6))
    error = pred - tgt
    loss = F.huber_loss(error, torch.zeros_like(error), reduction="none", delta=delta)
    return masked_mean(loss, mask & (target > 0.0), prediction)


def gradient_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, edge_band: torch.Tensor) -> torch.Tensor:
    pred = torch.log(torch.nan_to_num(prediction, nan=1.0).clamp_min(1.0e-6))
    tgt = torch.log(torch.nan_to_num(target, nan=1.0).clamp_min(1.0e-6))
    dx = torch.abs((pred[:, :, :, 1:] - pred[:, :, :, :-1]) - (tgt[:, :, :, 1:] - tgt[:, :, :, :-1]))
    dy = torch.abs((pred[:, :, 1:, :] - pred[:, :, :-1, :]) - (tgt[:, :, 1:, :] - tgt[:, :, :-1, :]))
    mx = mask[:, :, :, 1:] & mask[:, :, :, :-1] & (edge_band[:, :, :, 1:] | edge_band[:, :, :, :-1])
    my = mask[:, :, 1:, :] & mask[:, :, :-1, :] & (edge_band[:, :, 1:, :] | edge_band[:, :, :-1, :])
    return 0.5 * (masked_mean(dx, mx, prediction) + masked_mean(dy, my, prediction))


def occlusion_jump_loss(prediction: torch.Tensor, edge_condition: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Encourage a local depth jump when V2 predicts near/far occlusion structure."""

    log_pred = torch.log(torch.nan_to_num(prediction, nan=1.0).clamp_min(1.0e-6))
    near_ratio = edge_condition[:, 3:4]
    far_ratio = edge_condition[:, 4:5]
    occlusion = edge_condition[:, 5:6] > 0.5
    expected_jump = (far_ratio - near_ratio).clamp_min(0.0)
    grad_x = torch.zeros_like(log_pred)
    grad_y = torch.zeros_like(log_pred)
    grad_x[:, :, :, 1:] = torch.abs(log_pred[:, :, :, 1:] - log_pred[:, :, :, :-1])
    grad_y[:, :, 1:, :] = torch.abs(log_pred[:, :, 1:, :] - log_pred[:, :, :-1, :])
    observed_jump = torch.maximum(grad_x, grad_y)
    loss = torch.relu(expected_jump - observed_jump)
    return masked_mean(loss, mask & occlusion & (expected_jump > 0.05), prediction)


def contour_tangent_loss(prediction: torch.Tensor, edge_condition: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Keep depth residual locally smooth along high-confidence edge contours."""

    edge = (edge_condition[:, 0:1] > 0.3) & (edge_condition[:, 1:2] > 0.3)
    delta = torch.log(torch.nan_to_num(prediction, nan=1.0).clamp_min(1.0e-6))
    dxx = torch.abs(delta[:, :, :, 2:] - 2.0 * delta[:, :, :, 1:-1] + delta[:, :, :, :-2])
    dyy = torch.abs(delta[:, :, 2:, :] - 2.0 * delta[:, :, 1:-1, :] + delta[:, :, :-2, :])
    mx = mask[:, :, :, 1:-1] & edge[:, :, :, 1:-1]
    my = mask[:, :, 1:-1, :] & edge[:, :, 1:-1, :]
    return 0.5 * (masked_mean(dxx, mx, prediction) + masked_mean(dyy, my, prediction))


class DepthRefineV4Loss:
    def __init__(self, config: V4LossConfig) -> None:
        self.config = config

    def __call__(self, result: DepthRefineResult, batch: dict[str, torch.Tensor]) -> V4LossResult:
        valid = batch["valid"].bool()
        edge_band = batch["edge_band"].bool()
        target = batch["depth_gt_z"]
        depth0 = batch["depth0_z"]
        full = log_huber(result.depth_final_z, target, valid, self.config.huber_delta)
        edge = log_huber(result.depth_final_z, target, valid & edge_band, self.config.huber_delta)
        grad = gradient_loss(result.depth_final_z, target, valid, edge_band)
        occlusion = occlusion_jump_loss(result.depth_final_z, batch["edge_condition"], valid)
        tangent = contour_tangent_loss(result.depth_final_z, batch["edge_condition"], valid)
        anchor = log_huber(result.depth_final_z, depth0, valid & ~edge_band, self.config.huber_delta)
        valid_float = valid.to(result.delta_log_depth.dtype)
        drift = torch.abs((result.delta_log_depth * valid_float).sum() / valid_float.sum().clamp_min(1.0))
        total = (
            self.config.full_depth_weight * full
            + self.config.edge_depth_weight * edge
            + self.config.gradient_weight * grad
            + self.config.occlusion_jump_weight * occlusion
            + self.config.contour_tangent_weight * tangent
            + self.config.non_edge_anchor_weight * anchor
            + self.config.scale_drift_weight * drift
        )
        return V4LossResult(
            total=total,
            full_depth=full,
            edge_depth=edge,
            gradient=grad,
            occlusion_jump=occlusion,
            contour_tangent=tangent,
            non_edge_anchor=anchor,
            scale_drift=drift,
        )
