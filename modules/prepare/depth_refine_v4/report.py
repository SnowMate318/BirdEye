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


LABELS = {
    "source_rgb.png": "입력 RGB",
    "depth0_z.png": "Frozen DA-V2 D0",
    "depth_edge_diffusion_z.png": "Nonlearned edge-aware diffusion baseline",
    "depth_zero_condition_z.png": "V4 zero-condition ablation",
    "depth_final_z.png": "V4 refined depth D*",
    "delta_log_depth.png": "Zero-mean delta log-depth",
    "refinement_gate.png": "Refinement gate",
    "edge_condition.png": "V2 3D edge condition",
    "depth_error_d0.png": "D0 GT error",
    "depth_error_final.png": "D* GT error",
    "bev_d0.png": "D0 BEV RGB",
    "bev_edge_diffusion.png": "Edge-aware diffusion BEV RGB",
    "bev_zero_condition.png": "Zero-condition V4 BEV RGB",
    "bev_final.png": "D* BEV RGB",
}


def generate_report(run_dir: Path, metadata: dict, metrics: dict) -> Path:
    cards: list[str] = []
    for relative, label in LABELS.items():
        path = run_dir / relative
        if path.exists():
            cards.append(
                f'<article class="card"><h3>{html.escape(label)}</h3>'
                f'<p>{html.escape(relative)}</p><img src="{html.escape(relative)}" alt="{html.escape(label)}"></article>'
            )
    doc = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Depth Refine V4</title><style>{CSS}</style></head>
<body><main>
<h1>V4: V2 3D Edge-Conditioned Depth Anything V2 Refiner</h1>
<p class="notice">이 실험은 DA-V2의 전역 스케일을 새로 추정하지 않습니다. 모델은 입력 D0 주변의
zero-mean log-depth residual만 예측하고, V2 3D edge condition은 깊이가 끊기거나 이어져야 하는 위치를 알려주는
보조 신호로 사용됩니다. Evaluation depth는 입력이나 scale calibration에 쓰지 않고 지표 계산에만 사용합니다.</p>
<h2>Metrics</h2><pre>{html.escape(json.dumps(metrics, indent=2, ensure_ascii=False))}</pre>
<h2>Metadata</h2><pre>{html.escape(json.dumps(metadata, indent=2, ensure_ascii=False))}</pre>
<h2>Outputs</h2><div class="grid">{"".join(cards)}</div>
</main></body></html>"""
    out = run_dir / "index.html"
    out.write_text(doc, encoding="utf-8")
    return out
