from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent


@dataclass
class StageToggles:
    """파이프라인 stage별 실행 여부."""

    enable_direct_backbone: bool = True
    enable_tangent_backbone: bool = True
    enable_adaptive_ray_generation: bool = True
    enable_front_hemisphere_queries: bool = True
    enable_completion: bool = True
    enable_depth_loss: bool = True
    enable_rgb_loss: bool = True
    enable_valid_loss: bool = False
    enable_confidence_loss: bool = False
    enable_normal_loss: bool = False
    enable_residual_loss: bool = False
    enable_cycle_loss: bool = True
    enable_bev: bool = True
    enable_floor_surface_rasterization: bool = True
    enable_gt_evaluation: bool = True
    enable_html: bool = True


@dataclass
class PathConfig:
    """입력·데이터셋·외부 backbone·산출물 경로."""

    root: Path = ROOT
    input_rgb: Path = ROOT / "rgb.png"
    external_depth_z: Path = ROOT / "compare" / "depth_z.npy"
    outputs: Path = ROOT / "outputs"
    nyu_mat: Path = REPO_ROOT / "wide_fov_supervision" / "nyu" / "data" / "raw" / "nyu_depth_v2_labeled.mat"
    nyu_split_train: Path = REPO_ROOT / "experiment" / "foundation_models" / "DSINE" / "data" / "datasets" / "nyuv2" / "split" / "train.txt"
    nyu_split_test: Path = REPO_ROOT / "experiment" / "foundation_models" / "DSINE" / "data" / "datasets" / "nyuv2" / "split" / "test.txt"
    depth_anything_root: Path = REPO_ROOT / "experiment" / "foundation_models" / "depth_anything_v2" / "metric_depth"
    depth_anything_vitl_ckpt: Path = REPO_ROOT / "experiment" / "foundation_models" / "depth_anything_v2" / "metric_depth" / "checkpoints" / "depth_anything_v2_metric_hypersim_vitl.pth"
    depth_anything_vits_ckpt: Path = REPO_ROOT / "experiment" / "foundation_models" / "depth_anything_v2" / "metric_depth" / "checkpoints" / "depth_anything_v2_metric_hypersim_vits.pth"
    dsine_root: Path = REPO_ROOT / "experiment" / "foundation_models" / "DSINE"
    dsine_ckpt: Path = REPO_ROOT / "experiment" / "foundation_models" / "DSINE" / "projects" / "dsine" / "checkpoints" / "exp001_cvpr2024" / "dsine.pt"
    isaac_reference_run: Path = REPO_ROOT / "experiment" / "assets" / "excute" / "2026_06_23_13_50_41_TestCamera"
    checkpoint: Path | None = None


@dataclass
class FisheyeCameraConfig:
    """OpenCV fisheye camera와 world 변환 설정.

    camera 좌표는 +x 오른쪽, +y 아래, +z 전방이다. 모든 depth는 +z 성분인
    z-depth이고 3D point는 ``P=(D_z/ray_z)*ray``로 복원한다.
    """

    width: int = 1024
    height: int = 1024
    fx: float = 390.0
    fy: float = 390.0
    cx: float = 511.5
    cy: float = 511.5
    distortion: tuple[float, float, float, float] = (0.05, 0.01, -0.003, -0.0005)
    clipping_near_m: float = 0.05
    clipping_far_m: float = 100.0
    geometry_z_eps: float = 1.0e-3
    world_from_camera: tuple[tuple[float, float, float], ...] = (
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, -1.0),
    )
    camera_position_world: tuple[float, float, float] = (-2.0, 12.0, 10.0)


@dataclass
class RaySamplerConfig:
    """3D·BEV 희소도 기반 adaptive query 생성 설정."""

    central_fraction: float = 0.25
    target_gap_rad: float | None = None
    target_gap_multiplier: float = 1.0
    target_surface_gap_m: float = 0.04
    target_bev_gap_cells: float = 1.0
    depth_discontinuity_log_threshold: float = 0.20
    max_subdivision: int = 8
    min_gap_multiplier_to_add: float = 1.0
    floor_edge_priority_weight: float = 40.0
    floor_edge_height_margin_m: float = 0.20
    prefer_new_bev_cells_for_continuous_queries: bool = True
    max_added_queries_inference: int = 250_000
    max_added_queries_train: int = 16_384
    guided_train_fraction: float = 0.75
    front_fov_degrees: float = 180.0
    hemisphere_step_degrees: float = 0.5
    hemisphere_gap_multiplier: float = 1.0
    stencil_step_rad: float = 0.003
    normal_stencil_relative_step: float = 0.01
    dedupe_uv_decimals: int = 3
    max_queries_per_inference: int | None = None
    query_chunk_size: int = 65_536


@dataclass
class BackboneConfig:
    """Depth Anything V2, external z-depth, DSINE 실행 설정."""

    depth_source: str = "da_v2"
    da_encoder: str = "vitl"
    da_max_depth_m: float = 20.0
    da_input_size: int = 518
    dsine_pinhole_fov_degrees: float = 100.0
    tangent_resolution: int = 518
    tangent_fov_degrees: float = 100.0
    tangent_polar_degrees: float = 55.0
    use_amp_for_backbone: bool = True
    allow_synthetic_backbone_fallback: bool = False


@dataclass
class CompletionConfig:
    """네 corner support를 읽는 RGB-D query completion 모델 설정."""

    hidden_dim: int = 256
    attention_heads: int = 4
    attention_blocks: int = 3
    min_depth: float = 1.0e-4
    max_delta_log_depth: float = 1.0
    max_rgb_residual: float = 0.5
    valid_probability_threshold: float = 0.5
    confidence_probability_threshold: float = 0.5
    use_base_depth_for_continuous_queries: bool = True


@dataclass
class LossConfig:
    """Quad completion loss 가중치와 연속 표면 기준."""

    depth_weight: float = 1.00
    rgb_weight: float = 0.25
    valid_weight: float = 0.10
    confidence_weight: float = 0.20
    normal_weight: float = 0.10
    residual_weight: float = 0.01
    cycle_reconstruction_weight: float = 0.10
    cycle_least_squares_ridge: float = 1.0e-4
    cycle_min_internal_queries: int = 4
    corner_reconstruction_weight: float = 1.0
    depth_huber_delta: float = 0.10
    depth_discontinuity_log_threshold: float = 0.20
    normal_eps: float = 1.0e-6


@dataclass
class TrainConfig:
    """NYU convex quad manifest와 completion 학습 설정."""

    seed: int = 0
    epochs: int = 20
    batch_size: int = 2
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    amp: bool = True
    amp_initial_scale: float = 1024.0
    gradient_clip: float = 1.0
    num_workers: int = 2
    quad_manifest_schema_version: int = 2
    train_quads_per_frame: int = 64
    test_quads_per_frame: int = 16
    queries_per_quad: int = 64
    quad_min_side_px: float = 4.0
    quad_max_side_px: float = 96.0
    quad_max_rotation_degrees: float = 45.0
    quad_corner_jitter_fraction: float = 0.25
    guided_quad_fraction: float = 0.70
    continuous_quad_fraction: float = 0.60
    common_depth_scale_min: float = 0.5
    common_depth_scale_max: float = 2.0
    support_log_depth_noise_std: float = 0.03
    manifest_max_attempts_per_quad: int = 300
    max_train_items: int | None = None
    max_eval_items: int | None = None


@dataclass
class BevConfig:
    """world XY BEV splat 설정."""

    center_xy: tuple[float, float] = (-2.0, 12.0)
    size_m: float = 40.96
    meters_per_pixel: float = 0.04
    floor_height_percentile: float = 5.0
    floor_surface_fill_height_margin_m: float = 0.20
    floor_surface_fill_max_corner_z_range_m: float = 0.10
    top_min_height_above_floor_m: float = 0.10
    top_normal_z_threshold: float = 0.70

    @property
    def resolution(self) -> int:
        return int(round(self.size_m / self.meters_per_pixel))


@dataclass
class EvaluationConfig:
    """현재 Isaac 입력과 GT를 비교할 때 사용하는 안전 조건."""

    input_rgb_sha256: str = "e80a622103f59f6b19c765e3711977472cf1ed954240abda308449b6a8342bcd"
    use_isaac_gt_only_on_hash_match: bool = True


@dataclass
class PipelineConfig:
    """전체 파이프라인 설정 묶음."""

    paths: PathConfig = field(default_factory=PathConfig)
    camera: FisheyeCameraConfig = field(default_factory=FisheyeCameraConfig)
    ray: RaySamplerConfig = field(default_factory=RaySamplerConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    completion: CompletionConfig = field(default_factory=CompletionConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    bev: BevConfig = field(default_factory=BevConfig)
    eval: EvaluationConfig = field(default_factory=EvaluationConfig)
    toggles: StageToggles = field(default_factory=StageToggles)


def make_default_config() -> PipelineConfig:
    return PipelineConfig()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: _jsonable(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def config_to_dict(config: PipelineConfig) -> dict[str, Any]:
    return _jsonable(config)


def save_config(config: PipelineConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config_to_dict(config), indent=2, ensure_ascii=False), encoding="utf-8")


def ensure_output_roots(config: PipelineConfig) -> None:
    for name in ("cache", "train", "inference", "eval"):
        (config.paths.outputs / name).mkdir(parents=True, exist_ok=True)
