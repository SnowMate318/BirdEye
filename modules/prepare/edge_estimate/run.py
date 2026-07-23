"""격리 실험의 validate/cache/train/evaluate/infer/all 명령행 진입점."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


# 이 파일을 저장소 루트에서 직접 실행해도 ``wide_fov_supervision_v2`` 패키지를
# 찾을 수 있도록 프로젝트 부모 디렉터리만 import 경로에 추가한다.
PROJECT_PARENT = Path(__file__).resolve().parents[4]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from wide_fov_supervision_v2.modules.prepare.edge_estimate.config import (  # noqa: E402
    DEFAULT_TRAIN_VARIANTS,
    VARIANTS,
    EdgeEstimateConfig,
    Variant,
    make_edge_config,
)
from wide_fov_supervision_v2.modules.prepare.edge_estimate.dataset import build_edge_cache  # noqa: E402
from wide_fov_supervision_v2.modules.prepare.edge_estimate.pipeline import (  # noqa: E402
    evaluate_checkpoints,
    latest_checkpoints,
    run_inference,
    train_variant,
    validate_environment,
)


def _path(value: str | None) -> Path | None:
    return Path(value).expanduser().resolve() if value else None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="격리된 NYU 기반 fisheye 3D edge completion 가설 검증 파이프라인"
    )
    parser.add_argument("--mode", choices=("validate", "cache", "train", "evaluate", "infer", "all"), required=True)
    parser.add_argument("--variant", choices=(*VARIANTS, "all"), default="rgb_local")
    parser.add_argument("--input-rgb", type=str, default=None)
    parser.add_argument("--evaluation-depth", type=str, default=None)
    parser.add_argument("--prior-depth", type=str, default=None)
    parser.add_argument("--base-bev-run", type=str, default=None)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="단일 --variant 학습 모델. evaluate/infer에서 variant별 옵션보다 우선합니다.",
    )
    for variant in VARIANTS:
        parser.add_argument(f"--checkpoint-{variant.replace('_', '-')}", type=str, default=None)
    parser.add_argument("--skip-da-cache", action="store_true", help="cache에서 DA-V2 prior 생성을 생략합니다.")
    parser.add_argument("--max-train-frames", type=int, default=None)
    parser.add_argument("--max-test-frames", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--disable-amp", action="store_true")
    return parser


def _configure(args: argparse.Namespace) -> EdgeEstimateConfig:
    config = make_edge_config()
    if args.max_train_frames is not None:
        config.data.max_train_frames = args.max_train_frames
    if args.max_test_frames is not None:
        config.data.max_test_frames = args.max_test_frames
    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    if args.num_workers is not None:
        config.train.num_workers = args.num_workers
    if args.disable_amp:
        config.train.amp = False
    if args.skip_da_cache:
        config.data.cache_da_v2 = False
    return config


def _selected_variants(value: str) -> tuple[Variant, ...]:
    if value == "all":
        return DEFAULT_TRAIN_VARIANTS
    return (value,)  # type: ignore[return-value]


def _checkpoint_map(
    config: EdgeEstimateConfig,
    args: argparse.Namespace,
    variants: tuple[Variant, ...],
) -> dict[Variant, Path]:
    explicit: dict[Variant, Path] = {}
    for variant in VARIANTS:
        value = getattr(args, f"checkpoint_{variant}")
        if value:
            explicit[variant] = _path(value)  # type: ignore[assignment]
    if args.checkpoint:
        if len(variants) != 1:
            raise ValueError("--checkpoint는 단일 --variant와 함께 사용하세요. 여러 모델은 variant별 옵션을 사용합니다.")
        explicit[variants[0]] = _path(args.checkpoint)  # type: ignore[assignment]
    latest = latest_checkpoints(config)
    result = {variant: explicit.get(variant, latest.get(variant)) for variant in variants}
    missing = [variant for variant, path in result.items() if path is None or not path.exists()]
    if missing:
        raise FileNotFoundError(
            "checkpoint를 찾지 못했습니다: " + ", ".join(missing) + ". 먼저 --mode train을 실행하세요."
        )
    return {variant: path for variant, path in result.items() if path is not None}


def main() -> int:
    args = _parser().parse_args()
    config = _configure(args)
    input_rgb = _path(args.input_rgb)
    evaluation_depth = _path(args.evaluation_depth)
    prior_depth = _path(args.prior_depth)
    if evaluation_depth is None and args.mode in ("infer", "all") and config.base.paths.external_depth_z.exists():
        evaluation_depth = config.base.paths.external_depth_z.resolve()
    base_bev_run = _path(args.base_bev_run)
    variants = _selected_variants(args.variant)

    if args.mode == "validate":
        print(json.dumps(validate_environment(config, input_rgb, evaluation_depth), indent=2, ensure_ascii=False))
        return 0

    if args.mode in ("cache", "all"):
        include_da = config.data.cache_da_v2 and not args.skip_da_cache
        train_cache = build_edge_cache(config, "train", include_da=include_da)
        test_cache = build_edge_cache(config, "test", include_da=include_da)
        print(f"train cache: {train_cache}")
        print(f"test cache:  {test_cache}")
        if args.mode == "cache":
            return 0

    trained: dict[Variant, Path] = {}
    if args.mode in ("train", "all"):
        train_variants = DEFAULT_TRAIN_VARIANTS if args.mode == "all" else variants
        for variant in train_variants:
            checkpoint = train_variant(config, variant)
            trained[variant] = checkpoint
            print(f"{variant} best checkpoint: {checkpoint}")
        if args.mode == "train":
            return 0

    if args.mode in ("evaluate", "all"):
        checkpoints = trained or _checkpoint_map(config, args, variants)
        eval_dir = evaluate_checkpoints(config, checkpoints)
        print(f"evaluation: {eval_dir}")
        if args.mode == "evaluate":
            return 0

    if args.mode in ("infer", "all"):
        checkpoints = trained or _checkpoint_map(config, args, variants)
        inference_dir = run_inference(
            config,
            checkpoints,
            input_rgb=input_rgb,
            evaluation_depth_path=evaluation_depth,
            prior_depth_path=prior_depth,
            base_bev_run=base_bev_run,
        )
        print(f"inference: {inference_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
