from __future__ import annotations

from dataclasses import dataclass

import torch

from wide_fov_supervision_v2.config import LossConfig, StageToggles
from wide_fov_supervision_v2.loss.depth_losses import (
    delta_log_depth_regularization,
    graph_connected_zero,
    masked_log_depth_loss,
    radial_depth_consistency_loss,
)
from wide_fov_supervision_v2.loss.normal_losses import partial_normal_cosine_loss


@dataclass(frozen=True)
class PartialLossMasks:
    """각 loss 항에 독립적으로 적용할 `(B,Q)` boolean mask.

    `normal` mask에는 중심과 네 stencil depth의 유효성 및 depth discontinuity
    검사를 포함할 수 있다. 이 조건은 `depth` mask와 분리되어 있으므로 stencil이
    유효하지 않아도 중심 query의 RGB-D depth supervision은 그대로 사용된다.
    `radial`은 source/query 대응 및 연속성 조건, `delta`는 residual 정규화를
    허용할 query를 각각 나타낸다.
    """

    depth: torch.Tensor
    normal: torch.Tensor
    radial: torch.Tensor
    delta: torch.Tensor


@dataclass
class PartialLossResult:
    """학습 loop에서 loss와 항별 실제 유효 query 수를 함께 기록하는 결과."""

    total: torch.Tensor
    depth: torch.Tensor
    normal: torch.Tensor
    radial: torch.Tensor
    delta: torch.Tensor
    depth_valid_count: int
    normal_valid_count: int
    radial_valid_count: int
    delta_valid_count: int

    @property
    def valid_count(self) -> int:
        """기존 log 코드 호환용 값이며 depth supervision 개수를 뜻한다."""

        return self.depth_valid_count


def _checked_mask(name: str, mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """mask shape를 검사하고 boolean tensor로 바꾼다."""

    if mask.shape != reference.shape:
        raise ValueError(f"{name} mask must have shape {tuple(reference.shape)}, got {tuple(mask.shape)}")
    return mask.bool()


def _count(mask: torch.Tensor) -> int:
    return int(mask.detach().sum().item())


class PartialSupervisionLoss:
    """query ray의 depth·normal·radial·residual loss를 독립 mask로 계산한다.

    호출자가 sampler/GT/stencil 조건을 결합해 :class:`PartialLossMasks`를 만든다.
    이 클래스는 각 항에 필요한 finite/positive 조건만 추가한다. unknown ray를
    제외하는 `observed` 조건도 호출자가 각 mask에 명시해야 한다. 어떤 mask가
    비어 있어도 해당 입력 tensor와 autograd graph가 연결된 scalar zero를
    반환하므로 정상적으로 `backward()`할 수 있다.
    """

    _DEPTH_EPS = 1.0e-4

    def __init__(self, config: LossConfig, toggles: StageToggles) -> None:
        self.config = config
        self.toggles = toggles

    def __call__(
        self,
        *,
        pred_depth_z: torch.Tensor,
        target_depth_z: torch.Tensor | None,
        pred_normal: torch.Tensor | None,
        target_normal: torch.Tensor | None,
        query_rays: torch.Tensor,
        source_rays_query: torch.Tensor,
        depth0_query_z: torch.Tensor,
        delta_log_depth: torch.Tensor,
        masks: PartialLossMasks,
    ) -> PartialLossResult:
        depth_mask = _checked_mask("depth", masks.depth, pred_depth_z)
        normal_mask = _checked_mask("normal", masks.normal, pred_depth_z)
        radial_mask = _checked_mask("radial", masks.radial, pred_depth_z)
        delta_mask = _checked_mask("delta", masks.delta, pred_depth_z)

        eps = self._DEPTH_EPS
        if target_depth_z is None:
            depth_valid = torch.zeros_like(depth_mask)
        else:
            depth_valid = (
                depth_mask
                & torch.isfinite(pred_depth_z)
                & torch.isfinite(target_depth_z)
                & (pred_depth_z > eps)
                & (target_depth_z > eps)
            )

        if pred_normal is None or target_normal is None:
            normal_valid = torch.zeros_like(normal_mask)
        else:
            normal_valid = (
                normal_mask
                & torch.isfinite(pred_normal).all(dim=-1)
                & torch.isfinite(target_normal).all(dim=-1)
                & torch.isfinite(query_rays).all(dim=-1)
            )

        radial_valid = (
            radial_mask
            & torch.isfinite(pred_depth_z)
            & torch.isfinite(depth0_query_z)
            & torch.isfinite(query_rays[..., 2])
            & torch.isfinite(source_rays_query[..., 2])
            & (query_rays[..., 2] > eps)
            & (source_rays_query[..., 2] > eps)
            & (pred_depth_z > eps)
            & (depth0_query_z > eps)
        )
        delta_valid = delta_mask & torch.isfinite(delta_log_depth)

        zero = graph_connected_zero(pred_depth_z)
        depth = zero
        normal = zero
        radial = zero
        delta = zero

        if self.toggles.enable_depth_loss and target_depth_z is not None:
            depth = masked_log_depth_loss(pred_depth_z, target_depth_z, depth_valid, eps=eps)
        if self.toggles.enable_partial_normal_loss and pred_normal is not None and target_normal is not None:
            normal = partial_normal_cosine_loss(pred_normal, target_normal, query_rays, normal_valid)
        if self.toggles.enable_view_loss:
            radial = radial_depth_consistency_loss(
                pred_depth_z,
                depth0_query_z,
                query_rays[..., 2],
                source_rays_query[..., 2],
                radial_valid,
                eps=eps,
            )
        delta = delta_log_depth_regularization(delta_log_depth, delta_valid)
        total = (
            self.config.depth_weight * depth
            + self.config.normal_weight * normal
            + self.config.radial_weight * radial
            + self.config.delta_weight * delta
        )
        return PartialLossResult(
            total=total,
            depth=depth,
            normal=normal,
            radial=radial,
            delta=delta,
            depth_valid_count=_count(depth_valid),
            normal_valid_count=_count(normal_valid),
            radial_valid_count=_count(radial_valid),
            delta_valid_count=_count(delta_valid),
        )
