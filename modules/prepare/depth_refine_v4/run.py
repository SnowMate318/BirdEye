from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_PARENT = Path(__file__).resolve().parents[4]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from wide_fov_supervision_v2.modules.prepare.depth_refine_v4.config import make_v4_config
from wide_fov_supervision_v2.modules.prepare.depth_refine_v4.dataset import build_v4_cache
from wide_fov_supervision_v2.modules.prepare.depth_refine_v4.pipeline import (
    evaluate_refiner,
    run_inference,
    train_refiner,
    validate_environment,
)


def _path(value: str | None) -> Path | None:
    return Path(value).expanduser().resolve() if value else None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V4 Depth Anything V2 edge-conditioned depth refinement")
    parser.add_argument("--mode", choices=("validate", "cache", "train", "evaluate", "infer", "all"), required=True)
    parser.add_argument("--input-rgb", type=str, default=None)
    parser.add_argument(
        "--camera-config",
        choices=("fisheye", "pinhole_world"),
        default="fisheye",
        help="Camera/ray geometry used for inference and BEV restoration.",
    )
    parser.add_argument("--evaluation-depth", type=str, default=None)
    parser.add_argument("--depth0", type=str, default=None, help="Optional precomputed D0 npy. If omitted, DA-V2 is used.")
    parser.add_argument("--edge-run", type=str, default=None, help="Optional V2 inference run directory.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max-train-frames", type=int, default=None)
    parser.add_argument("--max-test-frames", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--skip-da-cache", action="store_true", help="Use blurred GT D0 proxy when building NYU cache.")
    parser.add_argument("--disable-amp", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = make_v4_config()
    if args.camera_config == "pinhole_world":
        config.base.camera = config.base.pinhole_camera
    if args.max_train_frames is not None:
        config.data.train_frames = args.max_train_frames
    if args.max_test_frames is not None:
        config.data.test_frames = args.max_test_frames
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

    input_rgb = _path(args.input_rgb)
    evaluation_depth = _path(args.evaluation_depth)
    depth0 = _path(args.depth0)
    edge_run = _path(args.edge_run)
    checkpoint = _path(args.checkpoint)

    if args.mode == "validate":
        print(json.dumps(validate_environment(config, input_rgb), indent=2, ensure_ascii=False))
        return 0
    if args.mode in ("cache", "all"):
        train_cache = build_v4_cache(config, "train")
        test_cache = build_v4_cache(config, "test")
        print(f"train cache: {train_cache}")
        print(f"test cache:  {test_cache}")
        if args.mode == "cache":
            return 0
    trained = None
    if args.mode in ("train", "all"):
        trained = train_refiner(config)
        print(f"best checkpoint: {trained}")
        if args.mode == "train":
            return 0
    if args.mode in ("evaluate", "all"):
        eval_dir = evaluate_refiner(config, checkpoint or trained)
        print(f"evaluation: {eval_dir}")
        if args.mode == "evaluate":
            return 0
    if args.mode in ("infer", "all"):
        infer_dir = run_inference(
            config,
            input_rgb=input_rgb,
            checkpoint=checkpoint or trained,
            evaluation_depth=evaluation_depth,
            depth0_path=depth0,
            edge_run=edge_run,
        )
        print(f"inference: {infer_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
