from __future__ import annotations

import html
import json
from pathlib import Path


CSS = """
body { margin: 0; font-family: "Segoe UI", "Malgun Gothic", sans-serif; background: #101216; color: #eceff4; }
main { max-width: 1240px; margin: 0 auto; padding: 28px 24px 48px; }
h1 { font-size: 28px; margin: 0 0 8px; }
h2 { font-size: 18px; margin: 28px 0 12px; color: #d8dee9; }
p, li { line-height: 1.55; color: #c8ced8; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; }
.card { border: 1px solid #2e3440; border-radius: 8px; padding: 12px; background: #171a21; }
.card img { width: 100%; background: #0b0d11; border-radius: 4px; }
.meta { white-space: pre-wrap; font: 12px/1.45 Consolas, monospace; color: #b8c0cc; overflow-x: auto; }
.pill { display: inline-block; padding: 3px 8px; border: 1px solid #4c566a; border-radius: 999px; margin: 2px 6px 2px 0; }
"""


IMAGE_LABELS = {
    "source_rgb.png": "입력 RGB",
    "quad_sampling_preview.png": "Adaptive query 위치 (초록: continuous, 빨강: edge)",
    "completion_rgb_preview.png": "추가 query RGB 예측",
    "completion_depth_preview.png": "추가 query z-depth 예측",
    "confidence_map.png": "Query confidence",
    "surface_gap_before_m.png": "추가 전 3D surface gap",
    "surface_gap_planned_after_m.png": "계획된 추가 후 3D surface gap",
    "bev_gap_before_cells.png": "추가 전 BEV gap (cell)",
    "bev_gap_planned_after_cells.png": "계획된 추가 후 BEV gap (cell)",
    "sampling_priority.png": "Adaptive sampling 우선순위",
    "sampling_eligible.png": "Adaptive candidate cell",
    "added_ray_density.png": "추가 query 밀도",
    "floor_surface_source_cell_mask.png": "Floor surface fill source cell",
    "front_hemisphere_coverage.png": "전방 180도 observed/unknown coverage",
    "depth_gt_absrel_error.png": "Source D0 GT AbsRel error",
    "completion_depth_gt_absrel_error.png": "Completion query GT AbsRel error",
    "normal_gt_angular_error.png": "Completion normal GT angular error",
    "bev_valid_before.png": "추가 query 적용 전 BEV coverage",
    "continuous_only/bev_rgb.png": "Continuous-only RGB BEV",
    "continuous_only/bev_valid.png": "Continuous-only BEV coverage",
    "continuous_only/newly_covered_bev_cells.png": "Continuous-only 신규 BEV cell",
    "continuous_only/floor_surface_rgb.png": "Continuous-only floor surface fill RGB",
    "continuous_only/floor_surface_valid.png": "Continuous-only floor surface fill coverage",
    "continuous_only/floor_surface_newly_covered_bev_cells.png": "Continuous-only floor surface 신규 BEV cell",
    "edge_confident/bev_rgb.png": "Edge-confident RGB BEV (기본 최종 결과)",
    "edge_confident/bev_valid.png": "Edge-confident BEV coverage",
    "edge_confident/newly_covered_bev_cells.png": "Edge-confident 신규 BEV cell",
    "edge_confident/floor_surface_rgb.png": "Edge-confident floor surface fill RGB",
    "edge_confident/floor_surface_valid.png": "Edge-confident floor surface fill coverage",
    "edge_confident/floor_surface_newly_covered_bev_cells.png": "Edge-confident floor surface 신규 BEV cell",
    "edge_confident/observed_top_occupancy.png": "관측된 top-facing non-floor surface",
    "edge_confident/top_probability_map.png": "Top-facing probability",
}


def generate_dashboard(run_dir: Path, metadata: dict, metrics: dict, image_files: list[str]) -> Path:
    """상대 경로 이미지와 JSON을 포함하는 한국어 결과 dashboard를 만든다."""

    cards = []
    for filename in image_files:
        if not (run_dir / filename).exists():
            continue
        label = IMAGE_LABELS.get(filename, filename)
        cards.append(
            f'<section class="card"><h2>{html.escape(label)}</h2>'
            f'<p>{html.escape(filename)}</p><img src="{html.escape(filename)}" alt="{html.escape(label)}"></section>'
        )
    document = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Convex Quad RGB-D Ray Completion 결과</title>
<style>{CSS}</style>
</head>
<body><main>
<h1>Convex Quad 기반 RGB-D Ray Completion · BEV 결과</h1>
<p>
어안 영상의 인접 2×2 pixel ray를 support로 사용하고, camera geometry가 만든 내부 query ray의
RGB·z-depth·valid·confidence를 completion 모델이 예측합니다. 모델은 ray 방향이나 개수를 직접
생성하지 않습니다.
</p>
<p>
<span class="pill">continuous_only: 연속 source cell + valid/confidence 통과</span>
<span class="pill">edge_confident: 모든 candidate + valid/confidence 통과</span>
<span class="pill">floor surface fill: 연속 바닥 cell을 BEV polygon으로 채움</span>
</p>
<p>
검정 observed_top_occupancy는 관측된 top-facing non-floor surface입니다. 흰색은 free가 아니라
non-top 또는 미관측 영역도 포함하므로 classic free/occupied grid가 아닙니다.
</p>
<h2>핵심 지표</h2><pre class="meta">{html.escape(json.dumps(metrics, indent=2, ensure_ascii=False))}</pre>
<h2>실행 metadata</h2><pre class="meta">{html.escape(json.dumps(metadata, indent=2, ensure_ascii=False))}</pre>
<h2>산출 이미지</h2><div class="grid">{''.join(cards)}</div>
</main></body></html>"""
    output = run_dir / "index.html"
    output.write_text(document, encoding="utf-8")
    return output
