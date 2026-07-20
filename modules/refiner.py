from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from wide_fov_supervision_v2.config import RefinerConfig
from wide_fov_supervision_v2.modules.query_geometry import normalize_vector, sample_at_uv


def scale_pixel_center_uv(uv: torch.Tensor, stride: int) -> torch.Tensor:
    """원본 영상 pixel-center 좌표를 stride feature map 좌표로 변환한다.

    이 프로젝트의 pixel center는 `0.5, 1.5, ...`로 정의된다. 따라서 단순히
    `uv / stride`로 나누면 반 pixel offset이 생기며, 올바른 변환은
    `(uv - 0.5) / stride + 0.5`이다.
    """

    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    return (uv - 0.5) / float(stride) + 0.5


class ResidualBlock(nn.Module):
    """작은 CNN encoder용 residual block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class RayAwareQueryRefiner(nn.Module):
    """임의 개수 query ray의 z-depth를 보정하는 ray-aware 모델.

    Source input:
        source_rgb: `(B,3,H,W)`, 0..1 RGB.
        source_depth0_z: `(B,1,H,W)`, Depth Anything V2 source z-depth.
        source_normal0: `(B,3,H,W)`, DSINE camera-frame normal.
        source_rays: `(B,3,H,W)`, source pixel-center ray map.
        source_valid: `(B,1,H,W)`, lens/depth/backbone finite 여부.

    Query input:
        query_rays: `(B,Q,3)`, query ray.
        query_source_uv: `(B,Q,2)`, source map에서 sampling할 pixel 좌표.
        query_relative_uv: `(B,Q,2)`, parent cell 내부 상대 좌표.
        query_sampling_features: `(B,Q,3)`. 순서대로 source cell의
            `angular_gap_ratio`, `surface_gap_ratio`, `bev_gap_ratio`이다.
            각 값은 sampler가 측정한 gap을 해당 목표 gap으로 나눈 무차원 비율이다.
        query_observed: `(B,Q)` 또는 `(B,Q,1)`, observed flag. unknown ray는
            모델을 통과할 수는 있지만 loss/point/BEV에서 제외된다.

    Output:
        dict. 핵심 값은 `depth_final_z=(B,Q)`와 `delta_log_depth=(B,Q)`이다.
        마지막 MLP layer는 0으로 초기화되므로 학습 전에는 `D*=D0_query`이다.
    """

    def __init__(self, config: RefinerConfig | None = None) -> None:
        super().__init__()
        self.config = config or RefinerConfig()
        c = int(self.config.base_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(self.config.source_channels, c, kernel_size=3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(inplace=True),
            ResidualBlock(c),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, c * 2),
            nn.SiLU(inplace=True),
            ResidualBlock(c * 2),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, c * 4),
            nn.SiLU(inplace=True),
            ResidualBlock(c * 4),
        )
        self.lat1 = nn.Conv2d(c * 2, c, kernel_size=1)
        self.lat2 = nn.Conv2d(c * 4, c, kernel_size=1)
        sampled_dim = c + c + c
        # D0(1), N0(3), source ray(3), query ray(3), relative UV(2),
        # source-query angle(1), sampler의 세 gap ratio, observed flag(1).
        query_extra_dim = 1 + 3 + 3 + 3 + 2 + 1 + int(self.config.query_sampling_feature_dim) + 1
        mlp_in = sampled_dim + query_extra_dim
        layers: list[nn.Module] = []
        hidden = int(self.config.hidden_dim)
        for i in range(int(self.config.mlp_layers) - 1):
            layers.append(nn.Linear(mlp_in if i == 0 else hidden, hidden))
            layers.append(nn.SiLU(inplace=True))
        layers.append(nn.Linear(hidden, 1))
        self.decoder = nn.Sequential(*layers)
        final = self.decoder[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def encode_source(self, source_rgb: torch.Tensor, source_depth0_z: torch.Tensor, source_normal0: torch.Tensor, source_rays: torch.Tensor, source_valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """11채널 dense source 입력을 한 번 encoding해 3단계 pyramid를 만든다.

        반환값 `(f0, f1, f2)`의 stride는 각각 1, 2, 4이다. 같은 source에서
        center/stencil 또는 여러 query chunk를 처리할 때 이 결과를 재사용하면
        CNN encoder의 중복 실행을 피하면서 autograd graph도 유지할 수 있다.
        """

        x = torch.cat([source_rgb, source_depth0_z, source_normal0, source_rays, source_valid], dim=1)
        if x.shape[1] != self.config.source_channels:
            raise ValueError(f"source input must have {self.config.source_channels} channels, got {x.shape[1]}")
        f0 = self.stem(x)
        f1_raw = self.down1(f0)
        f2_raw = self.down2(f1_raw)
        f1 = self.lat1(f1_raw)
        f2 = self.lat2(f2_raw)
        return f0, f1, f2

    def decode_queries(
        self,
        source_depth0_z: torch.Tensor,
        source_normal0: torch.Tensor,
        source_rays: torch.Tensor,
        source_valid: torch.Tensor,
        source_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        query_rays: torch.Tensor,
        query_source_uv: torch.Tensor,
        query_relative_uv: torch.Tensor,
        query_sampling_features: torch.Tensor,
        query_observed: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """공유 source pyramid에서 임의 개수 query의 z-depth를 decode한다.

        Args:
            source_depth0_z/source_normal0/source_rays/source_valid:
                `encode_source`에 사용한 것과 동일한 dense source tensor.
            source_features:
                `encode_source`가 반환한 `(f0, f1, f2)` tuple. stride는 1/2/4이다.
            query_rays/query_source_uv/query_relative_uv/
            query_sampling_features/query_observed:
                class docstring에 정의된 query 입력. 여러 query chunk가 동일한
                `source_features`를 공유할 수 있다.

        Returns:
            `forward`와 같은 depth residual 결과 dictionary.
        """

        if len(source_features) != 3:
            raise ValueError(f"source_features must contain three pyramid levels, got {len(source_features)}")
        f0, f1, f2 = source_features
        expected_sampling_dim = int(self.config.query_sampling_feature_dim)
        if (
            query_sampling_features.dim() != 3
            or query_sampling_features.shape[:2] != query_rays.shape[:2]
            or query_sampling_features.shape[-1] != expected_sampling_dim
        ):
            raise ValueError(
                "query_sampling_features must be "
                f"(B,Q,{expected_sampling_dim}), got {tuple(query_sampling_features.shape)}"
            )
        if query_observed.dim() == 2:
            query_observed = query_observed.unsqueeze(-1)
        query_rays = normalize_vector(query_rays)

        uv0 = query_source_uv
        uv1 = scale_pixel_center_uv(query_source_uv, 2)
        uv2 = scale_pixel_center_uv(query_source_uv, 4)
        sampled = torch.cat([sample_at_uv(f0, uv0), sample_at_uv(f1, uv1), sample_at_uv(f2, uv2)], dim=-1)

        d0_query = sample_at_uv(source_depth0_z, uv0).squeeze(-1)
        n0_query = normalize_vector(sample_at_uv(source_normal0, uv0))
        r_source_query = normalize_vector(sample_at_uv(source_rays, uv0))
        valid_query = sample_at_uv(source_valid, uv0).squeeze(-1)
        ray_dot = (r_source_query * query_rays).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        ray_delta = torch.acos(ray_dot)

        # unknown query의 uv는 NaN일 수 있으므로 sampled feature가 NaN으로 번지는 것을 막는다.
        finite = torch.isfinite(sampled).all(dim=-1, keepdim=True) & torch.isfinite(d0_query).unsqueeze(-1)
        sampled = torch.nan_to_num(sampled, nan=0.0, posinf=0.0, neginf=0.0)
        d0_query = torch.nan_to_num(d0_query, nan=0.0, posinf=0.0, neginf=0.0)
        n0_query = torch.nan_to_num(n0_query, nan=0.0, posinf=0.0, neginf=0.0)
        r_source_query = torch.nan_to_num(r_source_query, nan=0.0, posinf=0.0, neginf=0.0)
        query_extra = torch.cat(
            [
                d0_query.unsqueeze(-1),
                n0_query,
                r_source_query,
                query_rays,
                torch.nan_to_num(query_relative_uv, nan=0.0),
                torch.nan_to_num(ray_delta, nan=0.0),
                torch.nan_to_num(query_sampling_features, nan=0.0, posinf=0.0, neginf=0.0),
                query_observed.float(),
            ],
            dim=-1,
        )
        decoder_in = torch.cat([sampled, query_extra], dim=-1)
        delta = self.decoder(decoder_in).squeeze(-1)
        delta = delta.clamp(-float(self.config.max_delta_log_depth), float(self.config.max_delta_log_depth))
        depth_base = d0_query.clamp_min(float(self.config.min_depth_m))
        depth_final = depth_base * torch.exp(delta)
        depth_final = torch.where(finite.squeeze(-1) & (valid_query > 0.5), depth_final, torch.full_like(depth_final, float("nan")))
        return {
            "depth_final_z": depth_final,
            "delta_log_depth": delta,
            "depth0_query_z": d0_query,
            "normal0_query": n0_query,
            "source_ray_query": r_source_query,
            "source_valid_query": valid_query,
        }

    def forward(
        self,
        source_rgb: torch.Tensor,
        source_depth0_z: torch.Tensor,
        source_normal0: torch.Tensor,
        source_rays: torch.Tensor,
        source_valid: torch.Tensor,
        query_rays: torch.Tensor,
        query_source_uv: torch.Tensor,
        query_relative_uv: torch.Tensor,
        query_sampling_features: torch.Tensor,
        query_observed: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        source_features = self.encode_source(source_rgb, source_depth0_z, source_normal0, source_rays, source_valid)
        return self.decode_queries(
            source_depth0_z,
            source_normal0,
            source_rays,
            source_valid,
            source_features,
            query_rays,
            query_source_uv,
            query_relative_uv,
            query_sampling_features,
            query_observed,
        )
