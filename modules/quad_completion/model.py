from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from wide_fov_supervision_v2.config import CompletionConfig


@dataclass
class QuadCompletionResult:
    """Quad completion의 query별 출력.

    모든 tensor의 앞 두 축은 ``(B,Q)``이다. ``rgb``는 0..1, ``depth_z``는
    support와 같은 scale의 camera z-depth다. valid와 confidence는 logit으로
    반환해 학습 시 BCEWithLogits를 직접 적용한다.
    """

    rgb: torch.Tensor
    depth_z: torch.Tensor
    valid_logit: torch.Tensor
    confidence_logit: torch.Tensor
    delta_log_depth: torch.Tensor
    rgb_residual: torch.Tensor
    base_rgb: torch.Tensor
    base_depth_z: torch.Tensor


def _bilinear_weights(relative_uv: torch.Tensor) -> torch.Tensor:
    u, v = relative_uv.unbind(dim=-1)
    return torch.stack(
        ((1.0 - u) * (1.0 - v), u * (1.0 - v), u * v, (1.0 - u) * v),
        dim=-1,
    )


def _valid_median(depth: torch.Tensor, valid: torch.Tensor, eps: float) -> torch.Tensor:
    """네 support 중 valid positive depth의 lower median을 안정적으로 구한다."""

    finite = valid.bool() & torch.isfinite(depth) & (depth > eps)
    safe = torch.where(finite, depth, torch.full_like(depth, float("inf")))
    sorted_depth, _ = torch.sort(safe, dim=-1)
    count = finite.sum(dim=-1)
    index = torch.div((count - 1).clamp_min(0), 2, rounding_mode="floor")
    median = sorted_depth.gather(-1, index.unsqueeze(-1)).squeeze(-1)
    return torch.where(count > 0, median, torch.ones_like(median))


def _masked_bilerp(values: torch.Tensor, weights: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """invalid support의 weight를 제거한 뒤 다시 정규화해 bilerp한다."""

    valid_float = valid.to(dtype=values.dtype)
    weighted = weights * valid_float[:, None, :]
    denominator = weighted.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    weighted = weighted / denominator
    if values.ndim == 3:
        return torch.einsum("bqk,bkc->bqc", weighted, values)
    return torch.einsum("bqk,bk->bq", weighted, values)


class CrossAttentionBlock(nn.Module):
    """Query token이 네 support token만 읽는 pre-norm cross-attention block."""

    def __init__(self, hidden_dim: int, heads: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.support_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        support_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        q = self.query_norm(query)
        memory = self.support_norm(support)
        attended, _ = self.attention(
            q,
            memory,
            memory,
            key_padding_mask=support_padding_mask,
            need_weights=False,
        )
        query = query + attended
        return query + self.ffn(self.ffn_norm(query))


class QuadRayCompletionModel(nn.Module):
    """네 sparse RGB-D support로 convex quad 내부 query의 RGB-D를 복원한다.

    이 모델은 dense image feature, DSINE normal, camera pose를 받지 않는다. 깊이는
    네 support의 valid median으로 정규화하므로 support에 공통 scale을 곱하면
    network 입력은 같고 최종 ``depth_z``만 같은 배율로 변한다.
    """

    checkpoint_schema = "quad_rgbd_completion_v1"

    def __init__(self, config: CompletionConfig) -> None:
        super().__init__()
        self.config = config
        hidden = int(config.hidden_dim)
        # support: ray(3), rgb(3), normalized log-depth(1), valid(1), corner uv(2)
        self.support_encoder = nn.Sequential(
            nn.Linear(10, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.LayerNorm(hidden)
        )
        # query: ray(3), relative uv(2), bilinear RGB(3), bilinear log-depth(1)
        self.query_encoder = nn.Sequential(
            nn.Linear(9, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.LayerNorm(hidden)
        )
        self.blocks = nn.ModuleList(
            [CrossAttentionBlock(hidden, int(config.attention_heads)) for _ in range(int(config.attention_blocks))]
        )
        self.rgb_head = nn.Linear(hidden, 3)
        self.depth_head = nn.Linear(hidden, 1)
        self.valid_head = nn.Linear(hidden, 1)
        self.confidence_head = nn.Linear(hidden, 1)
        # checkpoint 없이 실행해도 최초 출력은 정확히 corner bilinear baseline이다.
        nn.init.zeros_(self.rgb_head.weight)
        nn.init.zeros_(self.rgb_head.bias)
        nn.init.zeros_(self.depth_head.weight)
        nn.init.zeros_(self.depth_head.bias)

    def forward(
        self,
        support_ray_dir: torch.Tensor,
        support_rgb: torch.Tensor,
        support_depth_z: torch.Tensor,
        support_valid: torch.Tensor,
        query_ray_dir: torch.Tensor,
        query_relative_uv: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> QuadCompletionResult:
        if support_ray_dir.ndim != 3 or support_ray_dir.shape[1:] != (4, 3):
            raise ValueError(f"support_ray_dir must be (B,4,3), got {tuple(support_ray_dir.shape)}")
        batch, query_count = query_ray_dir.shape[:2]
        if support_rgb.shape != (batch, 4, 3) or support_depth_z.shape != (batch, 4):
            raise ValueError("support_rgb/support_depth_z must have shapes (B,4,3)/(B,4)")
        if support_valid.shape != (batch, 4):
            raise ValueError(f"support_valid must be (B,4), got {tuple(support_valid.shape)}")
        if query_ray_dir.shape != (batch, query_count, 3):
            raise ValueError("query_ray_dir must be (B,Q,3)")
        if query_relative_uv.shape != (batch, query_count, 2) or query_mask.shape != (batch, query_count):
            raise ValueError("query_relative_uv/query_mask must be (B,Q,2)/(B,Q)")

        eps = float(self.config.min_depth)
        support_valid = support_valid.bool() & torch.isfinite(support_depth_z) & (support_depth_z > eps)
        scale = _valid_median(support_depth_z, support_valid, eps)
        safe_depth = torch.where(support_valid, support_depth_z, scale[:, None]).clamp_min(eps)
        normalized_log_depth = torch.log(safe_depth / scale[:, None].clamp_min(eps))

        weights = _bilinear_weights(query_relative_uv)
        base_rgb = _masked_bilerp(torch.nan_to_num(support_rgb), weights, support_valid)
        base_log_depth = _masked_bilerp(normalized_log_depth, weights, support_valid)
        base_depth = scale[:, None] * torch.exp(base_log_depth)

        corner_uv = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=support_rgb.dtype,
            device=support_rgb.device,
        ).expand(batch, -1, -1)
        support_input = torch.cat(
            [
                torch.nan_to_num(support_ray_dir),
                torch.nan_to_num(support_rgb),
                normalized_log_depth.unsqueeze(-1),
                support_valid.to(support_rgb.dtype).unsqueeze(-1),
                corner_uv,
            ],
            dim=-1,
        )
        query_input = torch.cat(
            [
                torch.nan_to_num(query_ray_dir),
                torch.nan_to_num(query_relative_uv),
                base_rgb,
                base_log_depth.unsqueeze(-1),
            ],
            dim=-1,
        )
        support_tokens = self.support_encoder(support_input)
        query_tokens = self.query_encoder(query_input)
        padding_mask = ~support_valid
        # MultiheadAttention은 한 batch의 모든 key가 masked이면 NaN을 만들 수 있다.
        all_invalid = padding_mask.all(dim=-1)
        if torch.any(all_invalid):
            padding_mask = padding_mask.clone()
            padding_mask[all_invalid, 0] = False
        for block in self.blocks:
            query_tokens = block(query_tokens, support_tokens, padding_mask)

        rgb_residual = torch.tanh(self.rgb_head(query_tokens)) * float(self.config.max_rgb_residual)
        delta = torch.tanh(self.depth_head(query_tokens).squeeze(-1)) * float(self.config.max_delta_log_depth)
        query_mask_float = query_mask.to(dtype=query_tokens.dtype)
        rgb_residual = rgb_residual * query_mask_float.unsqueeze(-1)
        delta = delta * query_mask_float
        rgb = (base_rgb + rgb_residual).clamp(0.0, 1.0)
        depth = scale[:, None] * torch.exp(base_log_depth + delta)
        return QuadCompletionResult(
            rgb=rgb,
            depth_z=depth,
            valid_logit=self.valid_head(query_tokens).squeeze(-1),
            confidence_logit=self.confidence_head(query_tokens).squeeze(-1),
            delta_log_depth=delta,
            rgb_residual=rgb_residual,
            base_rgb=base_rgb,
            base_depth_z=base_depth,
        )
