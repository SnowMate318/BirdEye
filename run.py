from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wide_fov_supervision_v2.config import ensure_output_roots, make_default_config
from wide_fov_supervision_v2.datasets.nyu.quad_dataset import build_nyu_quad_manifest
from wide_fov_supervision_v2.pipeline import run_inference, validate_environment
from wide_fov_supervision_v2.train.evaluate import evaluate_cached_predictions
from wide_fov_supervision_v2.train.trainer import train_completion_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convex Quad 기반 RGB-D Ray Completion 파이프라인")
    parser.add_argument("--mode", choices=["validate", "cache", "train", "evaluate", "infer", "all"], default="infer")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Quad completion checkpoint path")
    parser.add_argument("--input-rgb", type=Path, default=None)
    parser.add_argument(
        "--depth-source",
        choices=["da_v2", "external_npy"],
        default=None,
        help="fisheye 2x2 support에 사용할 z-depth source",
    )
    parser.add_argument("--depth-npy", type=Path, default=None, help="external_npy z-depth (H,W, metre)")
    parser.add_argument("--disable-direct", action="store_true")
    parser.add_argument("--disable-tangent", action="store_true")
    parser.add_argument("--disable-html", action="store_true")
    parser.add_argument("--disable-bev", action="store_true")
    parser.add_argument("--disable-completion", action="store_true")
    parser.add_argument("--disable-amp", action="store_true", help="train/eval 모델 계산에서 AMP mixed precision을 끕니다.")
    parser.add_argument("--max-train-items", type=int, default=None)
    parser.add_argument("--max-eval-items", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--floor-edge-boost", type=float, default=None)
    parser.add_argument("--floor-edge-margin", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def apply_overrides(config, args: argparse.Namespace):
    if args.checkpoint is not None:
        config.paths.checkpoint = args.checkpoint
    if args.input_rgb is not None:
        config.paths.input_rgb = args.input_rgb
    if args.depth_source is not None:
        config.backbone.depth_source = args.depth_source
    if args.depth_npy is not None:
        config.paths.external_depth_z = args.depth_npy
    if args.disable_direct:
        config.toggles.enable_direct_backbone = False
    if args.disable_tangent:
        config.toggles.enable_tangent_backbone = False
    if args.disable_html:
        config.toggles.enable_html = False
    if args.disable_bev:
        config.toggles.enable_bev = False
    if args.disable_completion:
        config.toggles.enable_completion = False
    if args.disable_amp:
        config.train.amp = False
        config.backbone.use_amp_for_backbone = False
    if args.max_train_items is not None:
        config.train.max_train_items = args.max_train_items
    if args.max_eval_items is not None:
        config.train.max_eval_items = args.max_eval_items
    if args.max_queries is not None:
        config.ray.max_queries_per_inference = args.max_queries
        config.ray.max_added_queries_inference = args.max_queries
    if args.floor_edge_boost is not None:
        config.ray.floor_edge_priority_weight = args.floor_edge_boost
    if args.floor_edge_margin is not None:
        config.ray.floor_edge_height_margin_m = args.floor_edge_margin
    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    return config


def main() -> int:
    args = parse_args()
    config = apply_overrides(make_default_config(), args)
    ensure_output_roots(config)
    if args.mode == "validate":
        print(json.dumps(validate_environment(config), indent=2, ensure_ascii=False))
        return 0
    if args.mode == "cache":
        build_nyu_quad_manifest(config, "train")
        build_nyu_quad_manifest(config, "test")
        return 0
    if args.mode == "train":
        checkpoint = train_completion_model(config)
        print(f"checkpoint={checkpoint}")
        return 0
    if args.mode == "evaluate":
        metrics = evaluate_cached_predictions(config)
        print(f"metrics={metrics}")
        return 0
    if args.mode == "infer":
        run_dir = run_inference(config)
        print(f"run_dir={run_dir}")
        return 0
    if args.mode == "all":
        build_nyu_quad_manifest(config, "train")
        build_nyu_quad_manifest(config, "test")
        checkpoint = train_completion_model(config)
        config.paths.checkpoint = checkpoint
        evaluate_cached_predictions(config)
        run_dir = run_inference(config)
        print(f"run_dir={run_dir}")
        return 0
    raise ValueError(args.mode)


if __name__ == "__main__":
    sys.exit(main())
