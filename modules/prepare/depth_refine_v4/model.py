from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .config import V4ModelConfig


@dataclass
class DepthRefineResult:
    depth_final_z: torch.Tensor
    delta_log_depth: torch.Tensor
    gate: torch.Tensor


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(x + self.net(x))


class EdgeConditionedDepthRefiner(nn.Module):
    """Predicts a zero-mean log-depth residual from frozen DA-V2 features."""

    checkpoint_schema = "depth_refine_v4_da_feature_edge_conditioned_v2"

    def __init__(self, config: V4ModelConfig) -> None:
        super().__init__()
        self.config = config
        hidden = int(config.hidden_channels)
        feature_layers = int(config.da_feature_layers)
        feature_channels = int(config.da_feature_channels)
        projection_channels = int(config.da_feature_projection_channels)
        self.da_projectors = nn.ModuleList(
            [nn.Conv2d(feature_channels, projection_channels, 1) for _ in range(feature_layers)]
        )
        self.da_fuser = nn.Sequential(
            nn.Conv2d(feature_layers * projection_channels, hidden, 1),
            nn.GroupNorm(8 if hidden % 8 == 0 else 1, hidden),
            nn.GELU(),
        )
        self.stem = nn.Sequential(
            nn.Conv2d(int(config.input_channels), hidden, 3, padding=1),
            nn.GroupNorm(8 if hidden % 8 == 0 else 1, hidden),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden) for _ in range(int(config.blocks))])
        self.delta_head = nn.Conv2d(hidden, 1, 3, padding=1)
        self.gate_head = nn.Conv2d(hidden, 1, 3, padding=1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, float(config.gate_bias))

    def forward(
        self,
        rgb: torch.Tensor,
        depth0_z: torch.Tensor,
        rays: torch.Tensor,
        source_valid: torch.Tensor,
        edge_condition: torch.Tensor,
        da_features: torch.Tensor | None = None,
    ) -> DepthRefineResult:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"rgb must be (B,3,H,W), got {tuple(rgb.shape)}")
        if depth0_z.shape[1] != 1 or rays.shape[1] != 3 or edge_condition.shape[1] != 6:
            raise ValueError("depth0/rays/edge_condition channel mismatch")
        valid = source_valid.bool() & torch.isfinite(depth0_z) & (depth0_z > 0.0)
        log_d0 = torch.log(torch.nan_to_num(depth0_z, nan=1.0).clamp_min(1.0e-6))
        x = torch.cat(
            [
                torch.nan_to_num(rgb),
                torch.nan_to_num(rays),
                log_d0,
                valid.to(rgb.dtype),
                torch.nan_to_num(edge_condition),
            ],
            dim=1,
        )
        feature = self.stem(x)
        if da_features is not None:
            feature = feature + self._project_da_features(da_features, feature.shape[-2:])
        feature = self.blocks(feature)
        raw_delta = float(self.config.max_delta_log_depth) * torch.tanh(self.delta_head(feature))
        gate = torch.sigmoid(self.gate_head(feature))
        delta = raw_delta * gate
        good = valid.to(delta.dtype)
        mean_delta = (delta * good).sum(dim=(2, 3), keepdim=True) / good.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        delta = torch.where(valid, delta - mean_delta, torch.zeros_like(delta))
        depth = torch.exp(log_d0 + delta)
        depth = torch.where(valid, depth, torch.full_like(depth, float("nan")))
        return DepthRefineResult(depth_final_z=depth, delta_log_depth=delta, gate=gate)

    def _project_da_features(self, da_features: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        if da_features.ndim != 5:
            raise ValueError(f"da_features must be (B,L,C,H,W), got {tuple(da_features.shape)}")
        if da_features.shape[1] != len(self.da_projectors):
            raise ValueError("da_features layer count mismatch")
        maps = []
        for index, projector in enumerate(self.da_projectors):
            projected = projector(torch.nan_to_num(da_features[:, index]))
            maps.append(F.interpolate(projected, target_hw, mode="bilinear", align_corners=False))
        return self.da_fuser(torch.cat(maps, dim=1))
