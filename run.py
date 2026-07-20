from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wide_fov_supervision_v2.config import ensure_output_roots, make_default_config
from wide_fov_supervision_v2.pipeline import run_inference, validate_environment
from wide_fov_supervision_v2.train.cache import build_nyu_teacher_cache
from wide_fov_supervision_v2.train.evaluate import evaluate_cached_predictions
from wide_fov_supervision_v2.train.query_cache import build_nyu_query_sidecar_cache
from wide_fov_supervision_v2.train.trainer import train_refiner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="wide_fov_supervision_v2 adaptive ray/refiner pipeline")
    parser.add_argument("--mode", choices=["validate", "cache", "train", "evaluate", "infer", "all"], default="infer")
    parser.add_argument("--checkpoint", type=Path, default=None, help="refiner checkpoint path")
    parser.add_argument("--input-rgb", type=Path, default=None, help="override input rgb path")
    parser.add_argument(
        "--depth-source",
        choices=["da_v2", "external_npy"],
        default=None,
        help="inference D0 source: metric DA-V2 or a pixel-aligned external z-depth NPY",
    )
    parser.add_argument(
        "--depth-npy",
        type=Path,
        default=None,
        help="external_npy mode source-camera z-depth path (H,W float array in metres)",
    )
    parser.add_argument("--disable-direct", action="store_true")
    parser.add_argument("--disable-tangent", action="store_true")
    parser.add_argument("--disable-html", action="store_true")
    parser.add_argument("--disable-bev", action="store_true")
    parser.add_argument("--disable-dense-coverage", action="store_true")
    parser.add_argument("--max-train-items", type=int, default=None)
    parser.add_argument("--max-eval-items", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=None, help="debug용 query cap. 기본은 cap 없음")
    parser.add_argument("--epochs", type=int, default=None, help="debug/trial용 epoch override")
    parser.add_argument("--batch-size", type=int, default=None, help="debug/trial용 batch size override")
    parser.add_argument(
        "--dense-subdivision",
        type=int,
        default=None,
        help="source 2x2 cell dense BEV subdivision. 5 is fast, 7/9 recovers more cells.",
    )
    parser.add_argument("--dense-chunk-cells", type=int, default=None)
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
    if args.disable_dense_coverage:
        config.toggles.enable_dense_coverage_bev = False
    if args.max_train_items is not None:
        config.train.max_train_items = args.max_train_items
    if args.max_eval_items is not None:
        config.train.max_eval_items = args.max_eval_items
    if args.max_queries is not None:
        config.ray.max_queries_per_inference = args.max_queries
    if args.dense_subdivision is not None:
        config.ray.dense_coverage_subdivision = args.dense_subdivision
    if args.dense_chunk_cells is not None:
        config.ray.dense_coverage_chunk_cells = args.dense_chunk_cells
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
        checks = validate_environment(config)
        print(json.dumps(checks, indent=2, ensure_ascii=False))
        return 0
    if args.mode == "cache":
        build_nyu_teacher_cache(config, split="train")
        build_nyu_teacher_cache(config, split="test")
        build_nyu_query_sidecar_cache(config, split="train")
        build_nyu_query_sidecar_cache(config, split="test")
        return 0
    if args.mode == "train":
        ckpt = train_refiner(config)
        print(f"checkpoint={ckpt}")
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
        build_nyu_teacher_cache(config, split="train")
        build_nyu_teacher_cache(config, split="test")
        build_nyu_query_sidecar_cache(config, split="train")
        build_nyu_query_sidecar_cache(config, split="test")
        ckpt = train_refiner(config)
        config.paths.checkpoint = ckpt
        evaluate_cached_predictions(config)
        run_dir = run_inference(config)
        print(f"run_dir={run_dir}")
        return 0
    raise ValueError(args.mode)


if __name__ == "__main__":
    sys.exit(main())
