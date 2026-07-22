from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from wide_fov_supervision_v2.config import LossConfig, StageToggles
from wide_fov_supervision_v2.modules.quad_completion.model import QuadCompletionResult


def _connected_zero(reference: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(reference).sum() * 0.0


def _masked_mean(value: torch.Tensor, mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    valid = mask.bool() & torch.isfinite(value)
    if not torch.any(valid):
        return _connected_zero(reference)
    return value[valid].mean()


def _support_scale(depth: torch.Tensor, valid: torch.Tensor, eps: float = 1.0e-4) -> torch.Tensor:
    finite = valid.bool() & torch.isfinite(depth) & (depth > eps)
    safe = torch.where(finite, depth, torch.full_like(depth, float("inf")))
    sorted_depth, _ = torch.sort(safe, dim=-1)
    count = finite.sum(dim=-1)
    index = torch.div((count - 1).clamp_min(0), 2, rounding_mode="floor")
    median = sorted_depth.gather(-1, index.unsqueeze(-1)).squeeze(-1)
    return torch.where(count > 0, median, torch.ones_like(median))


def _bilinear_basis(relative_uv: torch.Tensor) -> torch.Tensor:
    """relative-UV query를 ``a0 + a1*u + a2*v + a3*u*v`` basis로 바꾼다.

    Cycle reconstruction loss에서는 모델이 예측한 내부 query RGB-D만으로 작은
    bilinear patch를 다시 맞춘 뒤, 그 patch가 원래 네 support corner를 복원하는지
    확인한다. 이 basis는 그 least-squares fitting에 사용된다.
    """

    u, v = relative_uv.unbind(dim=-1)
    ones = torch.ones_like(u)
    return torch.stack((ones, u, v, u * v), dim=-1)


def _corner_basis(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """p00, p10, p11, p01 corner에서의 bilinear basis."""

    return torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
        ],
        dtype=dtype,
        device=device,
    )


def _reverse_cycle_loss(
    *,
    result: QuadCompletionResult,
    support_rgb: torch.Tensor | None,
    support_depth_z: torch.Tensor,
    support_valid: torch.Tensor,
    query_relative_uv: torch.Tensor | None,
    query_mask: torch.Tensor,
    target_valid: torch.Tensor,
    target_confidence: torch.Tensor,
    corner_query_mask: torch.Tensor,
    config: LossConfig,
) -> tuple[torch.Tensor, int]:
    """내부 query 예측으로 원래 2x2 support RGB-D를 재구성하는 cycle loss.

    직접 corner query를 쓰면 기존 corner reconstruction loss와 같은 제약이 되므로,
    여기서는 ``corner_query_mask``가 꺼진 내부 query만 사용한다. 내부 query의 예측
    RGB와 support-relative log-depth에 대해 bilinear least-squares patch를 맞추고,
    그 patch를 네 corner에서 평가한 값이 원래 support RGB-D와 가까운지 비교한다.

    이 손실은 절대 scale을 새로 추정하지 않는다. depth는 support median scale로
    나눈 log-depth에서 비교하므로, GT depth와 DA-V2 depth처럼 서로 다른 source가
    들어와도 "입력 support scale을 보존한 국소 일관성"만 학습한다.
    """

    if support_rgb is None or query_relative_uv is None:
        return _connected_zero(result.depth_z), 0

    work_depth = result.depth_z.float()
    work_rgb = result.rgb.float()
    work_support_rgb = support_rgb.float()
    work_support_depth = support_depth_z.float()
    work_relative_uv = query_relative_uv.float()
    query_mask = query_mask.bool()
    support_valid = support_valid.bool()
    cycle_query_mask = (
        query_mask
        & ~corner_query_mask.bool()
        & target_valid.bool()
        & target_confidence.bool()
        & torch.isfinite(work_relative_uv).all(dim=-1)
        & torch.isfinite(work_rgb).all(dim=-1)
        & torch.isfinite(work_depth)
        & (work_depth > 1.0e-4)
    )
    enough_queries = cycle_query_mask.sum(dim=-1) >= int(config.cycle_min_internal_queries)

    basis = _bilinear_basis(torch.nan_to_num(work_relative_uv))
    weights = cycle_query_mask.to(dtype=work_depth.dtype)
    weighted_basis = basis * weights.unsqueeze(-1)
    identity = torch.eye(4, dtype=work_depth.dtype, device=work_depth.device).unsqueeze(0)
    normal_matrix = torch.einsum("bqi,bqj->bij", weighted_basis, basis)
    normal_matrix = normal_matrix + identity * float(config.cycle_least_squares_ridge)

    scale = _support_scale(work_support_depth, support_valid)
    pred_relative_depth = torch.log(work_depth.clamp_min(1.0e-4) / scale[:, None].clamp_min(1.0e-4))
    rgb_rhs = torch.einsum("bqi,bqc->bic", weighted_basis, torch.nan_to_num(work_rgb))
    depth_rhs = torch.einsum("bqi,bq->bi", weighted_basis, torch.nan_to_num(pred_relative_depth))
    rgb_coeff = torch.linalg.solve(normal_matrix, rgb_rhs)
    depth_coeff = torch.linalg.solve(normal_matrix, depth_rhs.unsqueeze(-1)).squeeze(-1)

    corners = _corner_basis(work_depth.dtype, work_depth.device)
    reconstructed_rgb = torch.einsum("ki,bic->bkc", corners, rgb_coeff)
    reconstructed_depth = torch.einsum("ki,bi->bk", corners, depth_coeff)

    safe_support_depth = torch.where(support_valid, work_support_depth, scale[:, None]).clamp_min(1.0e-4)
    support_relative_depth = torch.log(safe_support_depth / scale[:, None].clamp_min(1.0e-4))
    depth_values = F.huber_loss(
        reconstructed_depth,
        support_relative_depth,
        reduction="none",
        delta=float(config.depth_huber_delta),
    )
    rgb_values = torch.abs(reconstructed_rgb - torch.nan_to_num(work_support_rgb)).mean(dim=-1)
    support_mask = (
        enough_queries[:, None]
        & support_valid
        & torch.isfinite(work_support_rgb).all(dim=-1)
        & torch.isfinite(work_support_depth)
        & (work_support_depth > 1.0e-4)
    )
    # RGB는 기존 direct RGB loss와 같은 상대 비율을 사용한다.
    cycle_values = depth_values + float(config.rgb_weight) * rgb_values
    cycle_loss = _masked_mean(cycle_values, support_mask, work_depth)
    return cycle_loss, int(support_mask.sum().detach().item())


@dataclass
class CompletionLossResult:
    total: torch.Tensor
    depth: torch.Tensor
    rgb: torch.Tensor
    valid: torch.Tensor
    confidence: torch.Tensor
    normal: torch.Tensor
    residual: torch.Tensor
    cycle: torch.Tensor
    depth_count: int
    rgb_count: int
    normal_count: int
    cycle_count: int


class QuadCompletionLoss:
    """RGB/depth/valid/confidence/normal loss를 서로 독립적인 mask로 계산한다."""

    def __init__(self, config: LossConfig, toggles: StageToggles) -> None:
        self.config = config
        self.toggles = toggles

    def __call__(
        self,
        *,
        result: QuadCompletionResult,
        support_depth_z: torch.Tensor,
        support_valid: torch.Tensor,
        target_rgb: torch.Tensor,
        target_depth_z: torch.Tensor,
        target_valid: torch.Tensor,
        target_confidence: torch.Tensor,
        query_mask: torch.Tensor,
        corner_query_mask: torch.Tensor,
        support_rgb: torch.Tensor | None = None,
        query_relative_uv: torch.Tensor | None = None,
        pred_normal: torch.Tensor | None = None,
        target_normal: torch.Tensor | None = None,
        normal_mask: torch.Tensor | None = None,
    ) -> CompletionLossResult:
        query_mask = query_mask.bool()
        target_valid = target_valid.bool()
        depth_mask = query_mask & target_valid & torch.isfinite(target_depth_z) & (target_depth_z > 1.0e-4)
        rgb_mask = query_mask & target_valid & torch.isfinite(target_rgb).all(dim=-1)
        corner_weight = 1.0 + corner_query_mask.to(result.depth_z.dtype) * float(
            self.config.corner_reconstruction_weight
        )

        scale = _support_scale(support_depth_z, support_valid)
        pred_relative = torch.log(result.depth_z.clamp_min(1.0e-4) / scale[:, None].clamp_min(1.0e-4))
        safe_target_depth = torch.where(depth_mask, target_depth_z, scale[:, None])
        target_relative = torch.log(safe_target_depth.clamp_min(1.0e-4) / scale[:, None].clamp_min(1.0e-4))
        depth_values = F.huber_loss(
            pred_relative,
            target_relative,
            reduction="none",
            delta=float(self.config.depth_huber_delta),
        ) * corner_weight
        safe_target_rgb = torch.where(rgb_mask.unsqueeze(-1), target_rgb, result.rgb.detach())
        rgb_values = torch.abs(result.rgb - safe_target_rgb).mean(dim=-1) * corner_weight

        depth_loss = _masked_mean(depth_values, depth_mask, result.depth_z)
        rgb_loss = _masked_mean(rgb_values, rgb_mask, result.rgb)
        valid_values = F.binary_cross_entropy_with_logits(
            result.valid_logit,
            target_valid.to(result.valid_logit.dtype),
            reduction="none",
        )
        valid_loss = _masked_mean(valid_values, query_mask, result.valid_logit)
        confidence_values = F.binary_cross_entropy_with_logits(
            result.confidence_logit,
            target_confidence.to(result.confidence_logit.dtype),
            reduction="none",
        )
        confidence_loss = _masked_mean(confidence_values, query_mask, result.confidence_logit)

        if pred_normal is None or target_normal is None or normal_mask is None:
            normal_loss = _connected_zero(result.depth_z)
            normal_valid = torch.zeros_like(query_mask)
        else:
            normal_valid = (
                normal_mask.bool()
                & query_mask
                & target_confidence.bool()
                & torch.isfinite(pred_normal).all(dim=-1)
                & torch.isfinite(target_normal).all(dim=-1)
            )
            cosine = 1.0 - torch.sum(pred_normal * target_normal, dim=-1).clamp(-1.0, 1.0)
            normal_loss = _masked_mean(cosine, normal_valid, pred_normal)

        residual_values = result.delta_log_depth.square() + result.rgb_residual.square().mean(dim=-1)
        residual_loss = _masked_mean(residual_values, query_mask, result.delta_log_depth)
        cycle_loss, cycle_count = _reverse_cycle_loss(
            result=result,
            support_rgb=support_rgb,
            support_depth_z=support_depth_z,
            support_valid=support_valid,
            query_relative_uv=query_relative_uv,
            query_mask=query_mask,
            target_valid=target_valid,
            target_confidence=target_confidence,
            corner_query_mask=corner_query_mask,
            config=self.config,
        )
        total = _connected_zero(result.depth_z)
        if self.toggles.enable_depth_loss:
            total = total + self.config.depth_weight * depth_loss
        if self.toggles.enable_rgb_loss:
            total = total + self.config.rgb_weight * rgb_loss
        if self.toggles.enable_valid_loss:
            total = total + self.config.valid_weight * valid_loss
        if self.toggles.enable_confidence_loss:
            total = total + self.config.confidence_weight * confidence_loss
        if self.toggles.enable_normal_loss:
            total = total + self.config.normal_weight * normal_loss
        total = total + self.config.residual_weight * residual_loss
        if self.toggles.enable_cycle_loss:
            total = total + self.config.cycle_reconstruction_weight * cycle_loss
        return CompletionLossResult(
            total=total,
            depth=depth_loss,
            rgb=rgb_loss,
            valid=valid_loss,
            confidence=confidence_loss,
            normal=normal_loss,
            residual=residual_loss,
            cycle=cycle_loss,
            depth_count=int(depth_mask.sum().detach().item()),
            rgb_count=int(rgb_mask.sum().detach().item()),
            normal_count=int(normal_valid.sum().detach().item()),
            cycle_count=cycle_count,
        )
