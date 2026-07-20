from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
import json
from typing import Any


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent


@dataclass
class StageToggles:
    """파이프라인 stage별 실행 여부.

    True/False 값만 바꾸면 같은 `run.py` entrypoint에서 backbone, adaptive ray,
    refiner, loss, BEV, HTML 생성을 독립적으로 켜고 끌 수 있다.
    """

    enable_direct_backbone: bool = True
    enable_tangent_backbone: bool = True
    enable_adaptive_ray_generation: bool = True
    enable_front_hemisphere_queries: bool = True
    enable_dense_coverage_bev: bool = True
    enable_depth_loss: bool = True
    enable_partial_normal_loss: bool = True
    enable_view_loss: bool = True
    enable_refiner: bool = True
    enable_bev: bool = True
    enable_gt_evaluation: bool = True
    enable_html: bool = True


@dataclass
class PathConfig:
    """프로젝트에서 사용하는 파일 경로.

    foundation model은 사용자의 기존 `experiment/foundation_models`를 재사용한다.
    이 구현은 checkpoint가 없다고 자동 clone하지 않는다. clone은 재현성을 흐리기 때문에
    명시적으로 실패시키고 경로를 고치게 하는 편이 안전하다.
    """

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
    """OpenCV fisheye/Kannala-Brandt 좌표계 카메라 설정.

    ray는 OpenCV camera frame을 따른다. 즉 +x는 영상 오른쪽, +y는 영상 아래,
    +z는 카메라 전방이다. z-depth는 이 +z축 성분의 metric depth이다.
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
    world_from_camera: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] = (
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, -1.0),
    )
    camera_position_world: tuple[float, float, float] = (-2.0, 12.0, 10.0)


@dataclass
class RaySamplerConfig:
    """Adaptive ray 생성 설정.

    주 sampler는 depth와 ray로 복원한 3D surface 간격을 사용하며, Isaac inference의
    ``surface_bev`` 모드에서는 world-XY BEV 간격도 함께 사용한다. angular gap 관련
    항목은 기존 baseline과 진단용으로 남겨 둔다.

    ``max_added_queries_*``는 원본 source ray나 front-hemisphere ray를 제외한
    *추가 observed query*에만 적용되는 결정적 예산이다. NYU 학습에서는 전체 예산의
    ``guided_train_fraction``만 guided sampler가 사용하고 나머지는 trainer가 균일
    subpixel query로 채운다.
    """

    central_fraction: float = 0.25
    target_gap_rad: float | None = None
    target_gap_multiplier: float = 1.0
    target_surface_gap_m: float = 0.04
    target_bev_gap_cells: float = 1.0
    depth_discontinuity_log_threshold: float = 0.20
    max_subdivision: int = 8
    min_gap_multiplier_to_add: float = 1.0
    max_added_queries_inference: int = 250_000
    max_added_queries_train: int = 16_384
    guided_train_fraction: float = 0.75
    front_fov_degrees: float = 180.0
    hemisphere_step_degrees: float = 0.5
    hemisphere_gap_multiplier: float = 1.0
    stencil_step_rad: float = 0.003
    dedupe_uv_decimals: int = 3
    max_queries_per_inference: int | None = None
    query_chunk_size: int = 65_536
    dense_coverage_subdivision: int = 5
    dense_coverage_chunk_cells: int = 65_536


@dataclass
class BackboneConfig:
    """Depth Anything V2와 DSINE 실행 설정."""

    # ``da_v2``는 metric DA-V2를 사용하고, ``external_npy``는
    # PathConfig.external_depth_z에 있는 source-camera z-depth(m)를 사용한다.
    # DSINE normal은 depth source와 독립적으로 계속 추론한다.
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
class RefinerConfig:
    """Ray-aware query refiner 모델 설정."""

    source_channels: int = 11
    # angular/surface/BEV gap ratio를 이 순서로 전달한다.
    query_sampling_feature_dim: int = 3
    base_channels: int = 64
    hidden_dim: int = 256
    mlp_layers: int = 4
    min_depth_m: float = 1.0e-4
    max_delta_log_depth: float = 2.0


@dataclass
class LossConfig:
    """Partial supervision loss 가중치와 mask 기준.

    depth/normal/radial/delta 항은 서로 독립된 mask를 사용한다. 특히 normal
    계산에 필요한 stencil 유효성이 center depth supervision을 막지 않는다.
    """

    depth_weight: float = 1.0
    normal_weight: float = 0.10
    radial_weight: float = 0.10
    delta_weight: float = 0.01
    depth_discontinuity_log_threshold: float = 0.20
    normal_eps: float = 1.0e-6


@dataclass
class TrainConfig:
    """NYU 학습 기본값."""

    seed: int = 0
    epochs: int = 20
    batch_size: int = 2
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    amp: bool = True
    gradient_clip: float = 1.0
    num_workers: int = 2
    cache_dtype: str = "float16"
    # (yaw, pitch) degree. OpenCV camera 좌표에서 +yaw는 오른쪽, +pitch는 위쪽이다.
    # MAT frame index를 이 tuple 길이로 나눈 나머지를 사용하므로 cache 재생성 시에도
    # orientation이 결정적으로 유지된다.
    nyu_virtual_orientations_degrees: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (55.0, 0.0),
        (-55.0, 0.0),
        (0.0, 55.0),
        (0.0, -55.0),
    )
    enable_query_sidecar_cache: bool = True
    query_cache_schema_version: int = 2
    max_train_items: int | None = None
    max_eval_items: int | None = None


@dataclass
class BevConfig:
    """world XY BEV splat 설정.

    `observed_top_occupancy`는 classic free/occupied grid가 아니다. 검정 PNG는
    관측된 top-facing non-floor surface를 뜻하고, 흰색은 그 외 상태를 모두 포함한다.
    """

    center_xy: tuple[float, float] = (-2.0, 12.0)
    size_m: float = 40.96
    meters_per_pixel: float = 0.04
    floor_height_percentile: float = 5.0
    top_min_height_above_floor_m: float = 0.10
    top_normal_z_threshold: float = 0.70

    @property
    def resolution(self) -> int:
        return int(round(self.size_m / self.meters_per_pixel))


@dataclass
class EvaluationConfig:
    """평가와 artifact 비교 설정."""

    input_rgb_sha256: str = "e80a622103f59f6b19c765e3711977472cf1ed954240abda308449b6a8342bcd"
    use_isaac_gt_only_on_hash_match: bool = True


@dataclass
class PipelineConfig:
    """전체 파이프라인 설정 묶음."""

    paths: PathConfig = field(default_factory=PathConfig)
    camera: FisheyeCameraConfig = field(default_factory=FisheyeCameraConfig)
    ray: RaySamplerConfig = field(default_factory=RaySamplerConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    refiner: RefinerConfig = field(default_factory=RefinerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    bev: BevConfig = field(default_factory=BevConfig)
    eval: EvaluationConfig = field(default_factory=EvaluationConfig)
    toggles: StageToggles = field(default_factory=StageToggles)


def make_default_config() -> PipelineConfig:
    """기본 설정 객체를 만든다."""

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
    """dataclass config를 JSON 저장 가능한 dict로 변환한다."""

    return _jsonable(config)


def save_config(config: PipelineConfig, path: Path) -> None:
    """현재 config를 `config.json` 같은 파일에 저장한다."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config_to_dict(config), indent=2, ensure_ascii=False), encoding="utf-8")


def ensure_output_roots(config: PipelineConfig) -> None:
    """출력 root/cache/train/inference 폴더를 만든다."""

    for name in ("cache", "train", "inference", "eval"):
        (config.paths.outputs / name).mkdir(parents=True, exist_ok=True)
