"""기존 파이프라인과 분리된 3D edge 실험의 데이터·모델·학습·추론 설정."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from wide_fov_supervision_v2.config import PipelineConfig, make_default_config


Variant = Literal["rgb_local", "rgb_context", "rgb_da_context"]
VARIANTS: tuple[Variant, ...] = ("rgb_local", "rgb_context", "rgb_da_context")
DEFAULT_TRAIN_VARIANTS: tuple[Variant, ...] = ("rgb_local", "rgb_context")


@dataclass
class EdgeDataConfig:
    """NYU에서 연결된 2x2 cell patch와 pseudo edge GT를 만드는 설정."""

    schema_version: int = 6
    orientations_degrees: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (-55.0, 0.0),
        (55.0, 0.0),
        (0.0, -55.0),
        (0.0, 55.0),
    )
    lattice_points: int = 5
    query_grid_size: int = 8
    cell_spans_px: tuple[int, ...] = (1, 2, 4, 8, 16)
    train_patches_per_frame: int = 16
    test_patches_per_frame: int = 4
    edge_center_fraction: float = 0.70
    raw_valid_fraction: float = 0.75
    depth_jump_log_threshold: float = 0.20
    depth_prior_blur_sigma_px: float = 1.5
    crease_normal_degrees: float = 30.0
    normal_cluster_degrees: float = 12.0
    stability_scales_px: tuple[int, ...] = (1, 2, 4)
    stability_min_count: int = 2
    edge_soft_sigma_px: float = 0.75
    max_attempts_per_patch: int = 200
    max_train_frames: int | None = None
    max_test_frames: int | None = None
    cache_da_v2: bool = False


@dataclass
class EdgeModelConfig:
    """5x5 point encoder, 4x4 context block, subpixel query decoder 설정."""

    point_hidden_dim: int = 128
    cell_hidden_dim: int = 256
    context_blocks: int = 3
    query_hidden_dim: int = 256
    min_depth_m: float = 0.05
    max_depth_m: float = 20.0
    log_depth_mean: float = 1.0
    log_depth_std: float = 0.70
    max_delta_log_depth: float = 0.70


@dataclass
class EdgePriorConfig:
    """RGB에서 계산한 2D edge prior를 모델 보조 입력과 후보 선택에 쓰는 설정."""

    enabled: bool = True
    blur_sigma: float = 1.0
    canny_sigma: float = 0.33
    gradient_weight: float = 0.55
    laplacian_weight: float = 0.25
    canny_weight: float = 0.20
    candidate_weight: float = 0.15


@dataclass
class EdgeLossConfig:
    """3D edge 검출·깊이·인접 cell 일관성 loss 가중치."""

    coarse_focal_weight: float = 1.00
    query_focal_weight: float = 1.00
    query_dice_weight: float = 0.50
    type_weight: float = 0.25
    near_depth_weight: float = 1.00
    far_depth_weight: float = 0.50
    confidence_weight: float = 0.20
    boundary_consistency_weight: float = 0.20
    tangent_weight: float = 0.05
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    depth_huber_delta: float = 0.10
    scale_invariant_depth: bool = True
    eps: float = 1.0e-6


@dataclass
class EdgeTrainConfig:
    """세 variant를 순차적으로 학습하는 기본 설정."""

    seed: int = 0
    epochs: int = 20
    batch_size: int = 4
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    num_workers: int = 2
    amp: bool = True
    amp_initial_scale: float = 1024.0
    gradient_clip: float = 1.0


@dataclass
class EdgeInferenceConfig:
    """전체 2x2 cell coarse scan과 subpixel query 선택 설정."""

    coarse_threshold: float = 0.30
    query_edge_threshold: float = 0.30  #0.5
    confidence_threshold: float = 0.30  #0.5
    coarse_batch_size: int = 512
    query_batch_size: int = 256
    candidate_dilation_cells: int = 1
    max_candidate_cells: int = 50_000
    dedupe_uv_decimals: int = 3
    contour_radius_px: float = 1.5
    hash_guard_evaluation_depth: bool = True


@dataclass
class EdgeEstimateConfig:
    """격리된 edge completion 실험 전체 설정."""

    base: PipelineConfig = field(default_factory=make_default_config)
    data: EdgeDataConfig = field(default_factory=EdgeDataConfig)
    model: EdgeModelConfig = field(default_factory=EdgeModelConfig)
    edge_prior: EdgePriorConfig = field(default_factory=EdgePriorConfig)
    loss: EdgeLossConfig = field(default_factory=EdgeLossConfig)
    train: EdgeTrainConfig = field(default_factory=EdgeTrainConfig)
    inference: EdgeInferenceConfig = field(default_factory=EdgeInferenceConfig)

    @property
    def output_root(self) -> Path:
        return self.base.paths.outputs / "edge_estimate"

    @property
    def cache_root(self) -> Path:
        return self.output_root / "cache" / edge_cache_hash(self)


def make_edge_config() -> EdgeEstimateConfig:
    return EdgeEstimateConfig()


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


def config_to_dict(config: EdgeEstimateConfig) -> dict[str, Any]:
    return _jsonable(config)


def save_edge_config(config: EdgeEstimateConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config_to_dict(config), indent=2, ensure_ascii=False), encoding="utf-8")


def edge_cache_hash(config: EdgeEstimateConfig) -> str:
    """cache tensor 형태와 pseudo-label 정의에 영향을 주는 항목만 hash한다."""

    payload = {
        "schema": config.data.schema_version,
        "camera": _jsonable(config.base.camera),
        "data": _jsonable(config.data),
        "edge_prior": _jsonable(config.edge_prior),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def ensure_edge_output_roots(config: EdgeEstimateConfig) -> None:
    for name in ("cache", "train", "eval", "inference"):
        (config.output_root / name).mkdir(parents=True, exist_ok=True)
