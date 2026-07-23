"""세 edge 모델의 영상·3D·BEV 산출물과 지표를 비교하는 한국어 HTML 생성기."""

from __future__ import annotations

import html
import json
from pathlib import Path


CSS = """
body { margin: 0; font-family: "Segoe UI", "Malgun Gothic", sans-serif; background: #101216; color: #eef1f6; }
main { max-width: 1440px; margin: 0 auto; padding: 28px 24px 56px; }
h1 { margin: 0 0 8px; font-size: 28px; }
h2 { margin: 28px 0 12px; font-size: 20px; }
h3 { margin: 10px 0 8px; font-size: 16px; }
p, li { color: #c8ced8; line-height: 1.55; }
.notice { padding: 12px; border-left: 4px solid #d08770; background: #1a1d24; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 14px; }
.card { padding: 12px; border: 1px solid #303642; border-radius: 8px; background: #171a21; }
.card img { display: block; width: 100%; border-radius: 4px; background: #080a0d; }
pre { overflow-x: auto; padding: 12px; background: #0d0f13; color: #cbd3df; font-size: 12px; }
.pill { display: inline-block; margin: 2px 5px 2px 0; padding: 3px 8px; border: 1px solid #526070; border-radius: 999px; }
"""


LABELS = {
    "source_rgb.png": "입력 어안 RGB",
    "edge_2d_prior.png": "RGB multi-cue 2D edge 보조 신호",
    "coarse_edge_probability.png": "2×2 셀 단위 coarse edge 확률",
    "edge_overlay.png": "완성된 subpixel edge overlay",
    "edge_3d_preview.png": "Camera XZ / World XY 3D edge 미리보기",
    "edge_probability.png": "Subpixel edge 확률",
    "edge_confidence.png": "Completion 신뢰도",
    "bev_keep_probability.png": "학습된 BEV 유지 확률",
    "edge_type.png": "3D edge 종류",
    "edge_depth_near_z.png": "Near/crease metric z-depth",
    "edge_depth_far_z.png": "Occlusion far metric z-depth",
    "edge_only/bev_edge_probability.png": "Edge-only BEV 신뢰도",
    "edge_only/bev_edge_polyline.png": "후처리된 BEV edge polyline",
    "edge_only/bev_edge_occupancy.png": "후처리된 BEV edge occupancy layer",
    "edge_only/bev_edge_projected_with_gt_depth.png": "모델 edge 위치 + GT depth BEV",
    "edge_only/bev_edge_projected_with_gt_depth_polyline.png": "모델 edge 위치 + GT depth polyline",
    "edge_only/bev_edge_projected_with_gt_depth_occupancy.png": "모델 edge 위치 + GT depth occupancy",
    "edge_only/bev_edge_near.png": "Near/crease 3D edge BEV",
    "edge_only/bev_edge_far.png": "Far background edge BEV",
    "fused/bev_rgb_with_edges.png": "기존 BEV와 edge를 융합한 복사본",
    "fused/bev_rgb_with_edge_polyline.png": "기존 BEV와 polyline edge 융합",
    "fused/bev_rgb_with_edge_occupancy.png": "기존 BEV와 occupancy edge 융합",
    "gt/edge_gt_overlay.png": "Simulator GT에서 계산한 edge",
    "gt/bev_edge_gt.png": "Simulator GT edge BEV",
}


def generate_edge_report(run_dir: Path, metadata: dict, metrics: dict, variants: list[str]) -> Path:
    """모든 variant의 3D edge completion 결과를 한 HTML에서 비교한다."""

    sections: list[str] = []
    for variant in variants:
        cards: list[str] = []
        for relative, label in LABELS.items():
            path = run_dir / variant / relative
            if not path.exists():
                continue
            source = f"{variant}/{relative}".replace("\\", "/")
            cards.append(
                f'<article class="card"><h3>{html.escape(label)}</h3>'
                f'<p>{html.escape(source)}</p><img src="{html.escape(source)}" alt="{html.escape(label)}"></article>'
            )
        sections.append(f'<h2>{html.escape(variant)}</h2><div class="grid">{"".join(cards)}</div>')
    document = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>3D Edge Completion 가설 검증</title><style>{CSS}</style></head>
<body><main>
<h1>Subpixel Ray 기반 3D Edge Completion 가설 검증</h1>
<p class="notice">완성된 ray는 카메라의 실측값이 아닙니다. 주변 2×2 셀의 RGB·ray 연속성과
NYU RGB-D에서 학습한 구조 prior로 예측한 값입니다. 신뢰도가 낮은 query는 생성하지 않고
unknown으로 남깁니다. Simulator depth는 모델 입력이나 scale 정렬에 사용하지 않고 이 보고서의 평가에만 사용합니다.</p>
<p><span class="pill">crease: 실제로 만나는 두 표면</span>
<span class="pill">occlusion: 전경/배경 경계</span>
<span class="pill">near edge를 기본 BEV에 반영</span></p>
<p class="notice">한계: NYU는 pinhole limited-FOV 데이터이며 raw depth 센서 노이즈가 있습니다.
또한 NYU 실내 장면과 Isaac warehouse fisheye 사이에는 domain gap이 있으므로, 생성된 edge와 confidence는
simulator GT 평가를 함께 확인해야 합니다.</p>
<h2>정량 지표</h2><pre>{html.escape(json.dumps(metrics, indent=2, ensure_ascii=False))}</pre>
<h2>실행 metadata</h2><pre>{html.escape(json.dumps(metadata, indent=2, ensure_ascii=False))}</pre>
{"".join(sections)}
</main></body></html>"""
    output = run_dir / "index.html"
    output.write_text(document, encoding="utf-8")
    return output
