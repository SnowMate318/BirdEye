from __future__ import annotations

from pathlib import Path
import html
import json


CSS = """
body { margin: 0; font-family: "Segoe UI", "Malgun Gothic", sans-serif; background: #101216; color: #eceff4; }
main { max-width: 1180px; margin: 0 auto; padding: 28px 24px 48px; }
h1 { font-size: 28px; margin: 0 0 8px; }
h2 { font-size: 18px; margin: 28px 0 12px; color: #d8dee9; }
p, li { line-height: 1.55; color: #c8ced8; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }
.card { border: 1px solid #2e3440; border-radius: 8px; padding: 12px; background: #171a21; }
.card img { width: 100%; background: #0b0d11; border-radius: 4px; image-rendering: auto; }
.meta { white-space: pre-wrap; font: 12px/1.45 Consolas, monospace; color: #b8c0cc; overflow-x: auto; }
.pill { display: inline-block; padding: 3px 8px; border: 1px solid #4c566a; border-radius: 999px; margin: 2px 6px 2px 0; color: #d8dee9; }
"""


IMAGE_LABELS = {
    "source_rgb.png": "입력 RGB",
    "ray_gap_before.png": "원본 ray 각도 간격 (진단용)",
    "surface_gap_before_m.png": "추가 전 3D 표면 간격",
    "surface_gap_planned_after_m.png": "계획된 추가 후 3D 표면 간격",
    "bev_gap_before_cells.png": "추가 전 BEV 간격 (cell 단위)",
    "bev_gap_planned_after_cells.png": "계획된 추가 후 BEV 간격",
    "sampling_priority.png": "Adaptive ray 선택 우선순위",
    "sampling_eligible.png": "Adaptive ray 생성 가능 cell",
    "planned_added_ray_density.png": "예산 적용 전 계획된 추가 query 분포",
    "adaptive_added_ray_density.png": "Adaptive Refiner query 분포",
    "dense_added_ray_density.png": "Dense source-cell BEV query 분포",
    "added_ray_density.png": "최종 추가 query ray 수",
    "front_hemisphere_coverage.png": "전방 180도 observed/unknown coverage",
    "bev_valid_before.png": "추가 ray 적용 전 BEV coverage",
    "bev_valid_after.png": "추가 ray 적용 후 BEV coverage",
    "newly_covered_bev_cells.png": "추가 ray로 새로 채워진 BEV cell",
    "bev_rgb.png": "최종 RGB BEV",
    "observed_top_occupancy.png": "관측 top-facing non-floor surface",
    "observed_support_occupancy.png": "최종 관측 BEV support coverage",
    "top_probability_map.png": "Top-facing probability",
}


def generate_dashboard(run_dir: Path, metadata: dict, metrics: dict, image_files: list[str]) -> Path:
    """self-contained 한국어 HTML dashboard를 생성한다."""

    cards = []
    for filename in image_files:
        path = run_dir / filename
        if path.exists():
            label = IMAGE_LABELS.get(filename, filename)
            cards.append(
                f'<section class="card"><h2>{html.escape(label)}</h2>'
                f'<p>{html.escape(filename)}</p><img src="{html.escape(filename)}" alt="{html.escape(label)}"></section>'
            )
    html_text = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>wide_fov_supervision_v2 결과</title>
<style>{CSS}</style>
</head>
<body>
<main>
<h1>wide_fov_supervision_v2 Ray 보완 · Depth Refinement · BEV 결과</h1>
<p>
선택한 depth source(DA-V2 또는 외부 z-depth)와 DSINE normal을 prior로 만든 뒤, D0와 camera ray로
복원한 인접 3D point 및 BEV footprint 간격이 큰 cell에 query ray를 추가했습니다.
각도 간격은 진단용으로만 사용하며, 학습 모델은 ray 방향이 아니라 각 query의 z-depth를 보정합니다.
전방 180° unknown ray는 별도 coverage 자료로만 저장되고 Refiner, loss, 3D point, BEV에서는 제외됩니다.
</p>
<p>
<span class="pill">검정 observed_top_occupancy: 관측된 top-facing non-floor surface</span>
<span class="pill">흰색: free / non-top / unobserved 포함</span>
</p>
<h2>핵심 지표</h2>
<pre class="meta">{html.escape(json.dumps(metrics, indent=2, ensure_ascii=False))}</pre>
<h2>실행 metadata</h2>
<pre class="meta">{html.escape(json.dumps(metadata, indent=2, ensure_ascii=False))}</pre>
<h2>산출 이미지</h2>
<div class="grid">
{''.join(cards)}
</div>
</main>
</body>
</html>
"""
    out = run_dir / "index.html"
    out.write_text(html_text, encoding="utf-8")
    return out
