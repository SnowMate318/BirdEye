from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

from wide_fov_supervision_v2.config import PipelineConfig, make_default_config


@dataclass
class V4DataConfig:
    """NYU virtual-fisheye cache settings for the depth refiner."""

    schema_version: int = 2
    image_size: int = 256
    train_frames: int | None = None
    test_frames: int | None = None
    cache_da_v2: bool = True
    synthetic_d0_blur_sigma_px: float = 3.0
    edge_band_radius_px: int = 3
    depth_edge_log_threshold: float = 0.10
    v2_teacher_for_cache: bool = True


@dataclass
class V4ModelConfig:
    """DA-V2 feature conditioned residual decoder."""

    input_channels: int = 14
    da_feature_layers: int = 4
    da_feature_channels: int = 1024
    da_feature_projection_channels: int = 32
    hidden_channels: int = 64
    blocks: int = 5
    max_delta_log_depth: float = 0.35
    gate_bias: float = -2.0


@dataclass
class V4LossConfig:
    """Loss weights for scale-preserving DA-V2 refinement."""

    full_depth_weight: float = 1.0
    edge_depth_weight: float = 1.0
    gradient_weight: float = 0.5
    occlusion_jump_weight: float = 0.25
    contour_tangent_weight: float = 0.10
    non_edge_anchor_weight: float = 0.10
    scale_drift_weight: float = 0.01
    huber_delta: float = 0.10
    eps: float = 1.0e-6


@dataclass
class V4TrainConfig:
    seed: int = 0
    epochs: int = 20
    batch_size: int = 2
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    num_workers: int = 2
    amp: bool = True
    gradient_clip: float = 1.0
    condition_dropout_probability: float = 0.10
    condition_jitter_probability: float = 0.20


@dataclass
class V4InferenceConfig:
    edge_variant: str = "rgb_context"
    run_v2_if_missing: bool = True
    depth_min_m: float = 0.05
    depth_max_m: float = 20.0


@dataclass
class DepthRefineV4Config:
    base: PipelineConfig = field(default_factory=make_default_config)
    data: V4DataConfig = field(default_factory=V4DataConfig)
    model: V4ModelConfig = field(default_factory=V4ModelConfig)
    loss: V4LossConfig = field(default_factory=V4LossConfig)
    train: V4TrainConfig = field(default_factory=V4TrainConfig)
    inference: V4InferenceConfig = field(default_factory=V4InferenceConfig)

    @property
    def output_root(self) -> Path:
        return self.base.paths.outputs / "depth_refine" / "v4"

    @property
    def cache_root(self) -> Path:
        return self.output_root / "cache" / cache_hash(self)


def make_v4_config() -> DepthRefineV4Config:
    return DepthRefineV4Config()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def config_to_dict(config: DepthRefineV4Config) -> dict[str, Any]:
    return _jsonable(config)


def save_v4_config(config: DepthRefineV4Config, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config_to_dict(config), indent=2, ensure_ascii=False), encoding="utf-8")


def cache_hash(config: DepthRefineV4Config) -> str:
    payload = {
        "schema": config.data.schema_version,
        "camera": _jsonable(config.base.camera),
        "data": _jsonable(config.data),
        "backbone": _jsonable(config.base.backbone),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
