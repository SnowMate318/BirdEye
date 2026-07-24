from __future__ import annotations

import html
import json
from pathlib import Path


CSS = """
body { margin: 0; font-family: "Segoe UI", "Malgun Gothic", sans-serif; background: #101216; color: #edf2f7; }
main { max-width: 1320px; margin: 0 auto; padding: 28px 24px 56px; }
h1 { margin: 0 0 8px; font-size: 28px; }
h2 { margin: 28px 0 12px; font-size: 20px; }
h3 { margin: 8px 0; font-size: 15px; }
p { color: #c7ced9; line-height: 1.55; }
.notice { padding: 12px; border-left: 4px solid #8fbcbb; background: #171b22; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(290px, 1fr)); gap: 14px; }
.card { padding: 12px; border: 1px solid #303642; border-radius: 8px; background: #171a21; }
.card img { display: block; width: 100%; border-radius: 4px; background: #07090c; }
pre { overflow-x: auto; padding: 12px; background: #0d0f13; color: #ccd5e1; font-size: 12px; }
"""


DEPTH_OUTPUTS = {
    "source_rgb.png": "입력 RGB 원본 복사본",
    "model_input_rgb.png": "모델 입력 RGB",
    "depth0_z.png": "Frozen DA-V2 D0",
    "depth_edge_diffusion_z.png": "비학습 edge-aware diffusion baseline",
    "depth_zero_condition_z.png": "V4 zero-condition ablation",
    "depth_final_z.png": "V4 refined depth D*",
    "delta_log_depth.png": "Zero-mean delta log-depth",
    "refinement_gate.png": "Refinement gate",
    "edge_condition.png": "V2 3D edge condition",
    "depth_error_d0.png": "D0 GT error",
    "depth_error_final.png": "D* GT error",
}


BEV_OUTPUTS = {
    "bev_d0.png": "D0 BEV RGB 지도",
    "bev_edge_diffusion.png": "Edge-aware diffusion BEV RGB 지도",
    "bev_zero_condition.png": "Zero-condition V4 BEV RGB 지도",
    "bev_final.png": "Full V4 D* BEV RGB 지도",
}


BEV_VALID_OUTPUTS = {
    "bev_d0_valid.png": "D0 BEV valid/coverage",
    "bev_edge_diffusion_valid.png": "Edge-aware diffusion BEV valid/coverage",
    "bev_zero_condition_valid.png": "Zero-condition V4 BEV valid/coverage",
    "bev_final_valid.png": "Full V4 D* BEV valid/coverage",
}


def _cards(run_dir: Path, labels: dict[str, str]) -> str:
    cards: list[str] = []
    for relative, label in labels.items():
        path = run_dir / relative
        if path.exists():
            cache_key = str(int(path.stat().st_mtime))
            cards.append(
                f'<article class="card"><h3>{html.escape(label)}</h3>'
                f'<p>{html.escape(relative)}</p>'
                f'<img src="{html.escape(relative)}?v={cache_key}" alt="{html.escape(label)}"></article>'
            )
    return "".join(cards)


def generate_report(run_dir: Path, metadata: dict, metrics: dict) -> Path:
    """Create a Korean HTML dashboard for V4 depth and BEV artifacts."""

    doc = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Depth Refine V4</title><style>{CSS}</style></head>
<body><main>
<h1>V4: V2 3D Edge-Conditioned Depth Anything V2 Refiner</h1>
<p class="notice">이 실험은 DA-V2의 전역 스케일을 새로 추정하지 않습니다. 모델은 입력 D0 주변의
zero-mean log-depth residual만 예측하고, V2 3D edge condition은 깊이가 끊기거나 이어져야 하는 위치를 알려주는
보조 신호로 사용됩니다. Evaluation depth는 입력이나 scale calibration에 쓰지 않고 지표 계산에만 사용합니다.</p>
<p class="notice">입력 파일: {html.escape(str(metadata.get("input_rgb", "")))}<br>
입력 SHA-256: {html.escape(str(metadata.get("input_rgb_sha256", "")))}<br>
Camera model: {html.escape(str(metadata.get("camera_model", "")))} /
K=({html.escape(str(metadata.get("camera_fx", "")))}, {html.escape(str(metadata.get("camera_fy", "")))},
{html.escape(str(metadata.get("camera_cx", "")))}, {html.escape(str(metadata.get("camera_cy", "")))})</p>
<h2>Metrics</h2><pre>{html.escape(json.dumps(metrics, indent=2, ensure_ascii=False))}</pre>
<h2>Metadata</h2><pre>{html.escape(json.dumps(metadata, indent=2, ensure_ascii=False))}</pre>
<h2>Depth / Edge Outputs</h2><div class="grid">{_cards(run_dir, DEPTH_OUTPUTS)}</div>
<h2>BEV RGB Maps</h2>
<p>각 depth 결과를 camera ray로 3D point로 복원한 뒤 world XY 평면에 splat한 정면 위 시점 지도입니다.</p>
<div class="grid">{_cards(run_dir, BEV_OUTPUTS)}</div>
<h2>BEV Valid / Coverage Maps</h2>
<p>흰색은 해당 BEV cell에 투영된 유효 3D point가 있음을 뜻합니다. free/occupied grid가 아니라 관측 coverage 비교용입니다.</p>
<div class="grid">{_cards(run_dir, BEV_VALID_OUTPUTS)}</div>
</main></body></html>"""
    out = run_dir / "index.html"
    out.write_text(doc, encoding="utf-8")
    return out
