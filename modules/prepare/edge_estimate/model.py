"""5x5 RGB/ray support에서 4x4 cell의 subpixel 3D edge를 예측한다."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import EdgeModelConfig, Variant, VARIANTS


@dataclass
class EdgeEstimateResult:
    """cell-level edge와 query-level 3D edge/depth 예측 결과."""

    cell_edge_logit: torch.Tensor
    cell_type_logits: torch.Tensor
    query_edge_logit: torch.Tensor
    query_depth_near_z: torch.Tensor
    query_depth_far_z: torch.Tensor
    query_confidence_logit: torch.Tensor
    query_bev_keep_logit: torch.Tensor
    query_delta_log_depth: torch.Tensor


class ResidualContextBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.block(value)


class EdgeEstimateModel(nn.Module):
    """5x5 support lattice에서 3D edge와 prior-relative depth를 예측한다.

    Depth head는 query prior depth가 있을 때 absolute metre 값을 직접 회귀하지 않고
    ``D = D0 * exp(delta)`` 형태의 local residual만 예측한다. delta는 query set 안에서
    zero-mean으로 만들어 전역 scale을 바꾸기 어렵게 한다.
    """

    def __init__(
        self,
        config: EdgeModelConfig,
        variant: Variant,
        *,
        log_depth_mean: float | None = None,
        log_depth_std: float | None = None,
        checkpoint_version: int = 4,
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"지원하지 않는 edge model variant입니다: {variant}")
        self.config = config
        self.variant = variant
        if checkpoint_version not in (3, 4):
            raise ValueError(f"지원하지 않는 edge checkpoint version입니다: v{checkpoint_version}")
        self.checkpoint_version = int(checkpoint_version)
        self.checkpoint_schema = f"edge_estimate_{variant}_v{self.checkpoint_version}"
        self.log_depth_mean = float(config.log_depth_mean if log_depth_mean is None else log_depth_mean)
        self.log_depth_std = max(float(config.log_depth_std if log_depth_std is None else log_depth_std), 1.0e-3)
        point_hidden = int(config.point_hidden_dim)
        cell_hidden = int(config.cell_hidden_dim)
        query_hidden = int(config.query_hidden_dim)
        self.point_encoder = nn.Sequential(
            nn.Conv2d(10, point_hidden, 1),
            nn.GELU(),
            nn.Conv2d(point_hidden, point_hidden, 1),
            nn.GroupNorm(8, point_hidden),
            nn.GELU(),
        )
        self.cell_encoder = nn.Sequential(
            nn.Conv2d(point_hidden * 4, cell_hidden, 1),
            nn.GELU(),
            nn.Conv2d(cell_hidden, cell_hidden, 1),
            nn.GroupNorm(8, cell_hidden),
            nn.GELU(),
        )
        self.context = nn.Sequential(*[ResidualContextBlock(cell_hidden) for _ in range(int(config.context_blocks))])
        self.cell_edge_head = nn.Conv2d(cell_hidden, 1, 1)
        self.cell_type_head = nn.Conv2d(cell_hidden, 3, 1)
        self.query_decoder = nn.Sequential(
            nn.Linear(cell_hidden + 5, query_hidden),
            nn.GELU(),
            nn.Linear(query_hidden, query_hidden),
            nn.GELU(),
            nn.Linear(query_hidden, query_hidden),
            nn.GELU(),
            nn.Linear(query_hidden, 7 if self.checkpoint_version >= 4 else 6),
        )

    def encode_cells(
        self,
        support_rgb: torch.Tensor,
        support_ray_dir: torch.Tensor,
        support_valid: torch.Tensor,
        da_relative_log_depth: torch.Tensor | None = None,
        da_valid: torch.Tensor | None = None,
        support_edge_2d: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`(B,5,5,*)` support를 `(B,C,4,4)` cell feature와 coarse head로 바꾼다."""

        if support_rgb.ndim != 4 or support_rgb.shape[1:] != (5, 5, 3):
            raise ValueError(f"support_rgb must be (B,5,5,3), got {tuple(support_rgb.shape)}")
        batch = support_rgb.shape[0]
        if support_ray_dir.shape != (batch, 5, 5, 3) or support_valid.shape != (batch, 5, 5):
            raise ValueError("support_ray_dir/support_valid must be (B,5,5,3)/(B,5,5)")
        if da_relative_log_depth is None:
            da_relative_log_depth = torch.zeros((batch, 5, 5), dtype=support_rgb.dtype, device=support_rgb.device)
        if da_valid is None:
            da_valid = torch.zeros((batch, 5, 5), dtype=torch.bool, device=support_rgb.device)
        if support_edge_2d is None:
            support_edge_2d = torch.zeros((batch, 5, 5), dtype=support_rgb.dtype, device=support_rgb.device)
        if da_relative_log_depth.shape != (batch, 5, 5) or da_valid.shape != (batch, 5, 5):
            raise ValueError("DA inputs must be (B,5,5)")
        if support_edge_2d.shape != (batch, 5, 5):
            raise ValueError("support_edge_2d must be (B,5,5)")
        if self.variant != "rgb_da_context":
            da_relative_log_depth = torch.zeros_like(da_relative_log_depth)
            da_valid = torch.zeros_like(da_valid)
        point_input = torch.cat(
            [
                torch.nan_to_num(support_rgb),
                torch.nan_to_num(support_ray_dir),
                support_valid.to(support_rgb.dtype).unsqueeze(-1),
                torch.nan_to_num(support_edge_2d).unsqueeze(-1),
                torch.nan_to_num(da_relative_log_depth).unsqueeze(-1),
                da_valid.to(support_rgb.dtype).unsqueeze(-1),
            ],
            dim=-1,
        ).permute(0, 3, 1, 2)
        points = self.point_encoder(point_input)
        cell_corners = torch.cat(
            [points[:, :, :-1, :-1], points[:, :, :-1, 1:], points[:, :, 1:, 1:], points[:, :, 1:, :-1]],
            dim=1,
        )
        cells = self.cell_encoder(cell_corners)
        if self.variant != "rgb_local":
            cells = self.context(cells)
        return cells, self.cell_edge_head(cells).squeeze(1), self.cell_type_head(cells).permute(0, 2, 3, 1)

    def _depth_from_prior(
        self,
        decoded: torch.Tensor,
        query_prior_depth_z: torch.Tensor | None,
        query_prior_valid: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_min = math.log(self.config.min_depth_m)
        log_max = math.log(self.config.max_depth_m)
        fallback_log = (self.log_depth_mean + self.log_depth_std * decoded[..., 4]).clamp(log_min, log_max)
        if query_prior_depth_z is None:
            valid = torch.zeros_like(decoded[..., 1], dtype=torch.bool)
            prior = torch.exp(fallback_log)
        else:
            prior = torch.nan_to_num(query_prior_depth_z, nan=0.0, posinf=0.0, neginf=0.0)
            valid = torch.isfinite(query_prior_depth_z) & (prior > 0.0)
            if query_prior_valid is not None:
                valid &= query_prior_valid.bool()
            prior = prior.clamp_min(self.config.min_depth_m).clamp_max(self.config.max_depth_m)
        delta = float(self.config.max_delta_log_depth) * torch.tanh(decoded[..., 1])
        valid_float = valid.to(delta.dtype)
        reduce_dims = tuple(range(1, delta.ndim))
        mean_delta = (delta * valid_float).sum(dim=reduce_dims, keepdim=True) / valid_float.sum(
            dim=reduce_dims, keepdim=True
        ).clamp_min(1.0)
        delta = torch.where(valid, delta - mean_delta, torch.zeros_like(delta))
        near_log = torch.where(valid, torch.log(prior.clamp_min(1.0e-6)) + delta, fallback_log).clamp(log_min, log_max)
        far_gap = self.log_depth_std * F.softplus(decoded[..., 2] + 0.25 * torch.tanh(decoded[..., 5]))
        return torch.exp(near_log), torch.exp((near_log + far_gap).clamp(log_min, log_max)), delta

    def decode_queries(
        self,
        cell_features: torch.Tensor,
        query_ray_dir: torch.Tensor,
        query_relative_uv: torch.Tensor,
        query_prior_depth_z: torch.Tensor | None = None,
        query_prior_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """각 cell query에서 edge와 near/far depth를 예측한다."""

        batch = cell_features.shape[0]
        if query_ray_dir.ndim != 5 or query_ray_dir.shape[:3] != (batch, 4, 4) or query_ray_dir.shape[-1] != 3:
            raise ValueError("query_ray_dir must be (B,4,4,Q,3)")
        if query_relative_uv.shape != query_ray_dir.shape[:-1] + (2,):
            raise ValueError("query_relative_uv must be (B,4,4,Q,2)")
        query_count = query_ray_dir.shape[-2]
        cells = cell_features.permute(0, 2, 3, 1).unsqueeze(-2).expand(-1, -1, -1, query_count, -1)
        decoded = self.query_decoder(
            torch.cat([cells, torch.nan_to_num(query_ray_dir), torch.nan_to_num(query_relative_uv)], dim=-1)
        )
        near, far, delta = self._depth_from_prior(decoded, query_prior_depth_z, query_prior_valid)
        bev_keep = (
            decoded[..., 6]
            if self.checkpoint_version >= 4
            else torch.full_like(decoded[..., 0], 20.0)
        )
        return decoded[..., 0], near, far, decoded[..., 3], bev_keep, delta

    def decode_selected_queries(
        self,
        cell_features: torch.Tensor,
        query_ray_dir: torch.Tensor,
        query_relative_uv: torch.Tensor,
        query_prior_depth_z: torch.Tensor | None = None,
        query_prior_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """coarse scan에서 선택된 cell만 `(B,Q)` 형태로 정밀 추론한다."""

        if cell_features.ndim != 2:
            raise ValueError("cell_features must be (B,H)")
        if query_ray_dir.ndim != 3 or query_ray_dir.shape[-1] != 3:
            raise ValueError("query_ray_dir must be (B,Q,3)")
        if query_relative_uv.shape != query_ray_dir.shape[:-1] + (2,):
            raise ValueError("query_relative_uv must be (B,Q,2)")
        expanded = cell_features[:, None, :].expand(-1, query_ray_dir.shape[1], -1)
        decoded = self.query_decoder(
            torch.cat([expanded, torch.nan_to_num(query_ray_dir), torch.nan_to_num(query_relative_uv)], dim=-1)
        )
        near, far, delta = self._depth_from_prior(decoded, query_prior_depth_z, query_prior_valid)
        bev_keep = (
            decoded[..., 6]
            if self.checkpoint_version >= 4
            else torch.full_like(decoded[..., 0], 20.0)
        )
        return decoded[..., 0], near, far, decoded[..., 3], bev_keep, delta

    def forward(
        self,
        support_rgb: torch.Tensor,
        support_ray_dir: torch.Tensor,
        support_valid: torch.Tensor,
        query_ray_dir: torch.Tensor,
        query_relative_uv: torch.Tensor,
        da_relative_log_depth: torch.Tensor | None = None,
        da_valid: torch.Tensor | None = None,
        support_edge_2d: torch.Tensor | None = None,
        query_prior_depth_z: torch.Tensor | None = None,
        query_prior_valid: torch.Tensor | None = None,
    ) -> EdgeEstimateResult:
        """공개 학습 인터페이스."""

        cells, cell_edge, cell_type = self.encode_cells(
            support_rgb, support_ray_dir, support_valid, da_relative_log_depth, da_valid, support_edge_2d
        )
        query_edge, near, far, confidence, bev_keep, delta = self.decode_queries(
            cells, query_ray_dir, query_relative_uv, query_prior_depth_z, query_prior_valid
        )
        return EdgeEstimateResult(
            cell_edge_logit=cell_edge,
            cell_type_logits=cell_type,
            query_edge_logit=query_edge,
            query_depth_near_z=near,
            query_depth_far_z=far,
            query_confidence_logit=confidence,
            query_bev_keep_logit=bev_keep,
            query_delta_log_depth=delta,
        )
