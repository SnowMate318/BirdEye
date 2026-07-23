# Depth Refine V4

V4는 Depth Anything V2를 frozen backbone으로 사용하고, V2 3D edge 추론 결과를 condition으로 넣어
metric z-depth를 보정하는 실험 패키지입니다. 기존 completion/edge 파이프라인은 수정하지 않고
`outputs/depth_refine/v4` 아래에 cache, train, eval, inference 결과를 분리 저장합니다.

## 구조

```text
RGB
  -> frozen DA-V2
       -> D0
       -> 4-layer DINOv2 intermediate feature maps

RGB + D0 + camera rays
  -> frozen edge_estimate_v2 rgb_context teacher
       -> dense 3D edge condition

RGB + rays + D0 + DA features + V2 edge condition
  -> V4 residual refiner
       -> D* = D0 * exp(zero-mean delta_log_depth)
```

DA-V2 checkpoint는 그대로 사용하고 학습하지 않습니다. 학습되는 파라미터는
`EdgeConditionedDepthRefiner`뿐입니다. `evaluation-depth`는 입력이나 scale calibration에 쓰지 않고
지표 계산에만 사용합니다.

## 실행

PowerShell 기준:

```powershell
cd C:\projects\isaac_sim_test\wide_fov_supervision_v2
.\.venv\Scripts\Activate.ps1
```

환경 확인:

```powershell
python modules\prepare\depth_refine_v4\run.py --mode validate
```

NYU cache 생성:

```powershell
python modules\prepare\depth_refine_v4\run.py --mode cache
```

이 cache는 각 frame마다 DA-V2 feature와 frozen V2 edge teacher condition을 생성하므로 오래 걸립니다.
빠른 동작 확인만 할 때는 다음처럼 줄여 실행합니다.

```powershell
python modules\prepare\depth_refine_v4\run.py --mode all `
  --max-train-frames 1 `
  --max-test-frames 1 `
  --epochs 1 `
  --batch-size 1 `
  --num-workers 0 `
  --disable-amp
```

전체 학습:

```powershell
python modules\prepare\depth_refine_v4\run.py --mode train
```

평가:

```powershell
python modules\prepare\depth_refine_v4\run.py --mode evaluate
```

`rgb.png` 추론:

```powershell
python modules\prepare\depth_refine_v4\run.py --mode infer `
  --input-rgb rgb.png `
  --evaluation-depth compare\depth_z.npy
```

이미 계산해 둔 V2 edge run을 재사용할 때:

```powershell
python modules\prepare\depth_refine_v4\run.py --mode infer `
  --input-rgb rgb.png `
  --edge-run outputs\edge_estimate\v2\inference\<RUN> `
  --evaluation-depth compare\depth_z.npy
```

이미 계산해 둔 D0를 쓰고 싶을 때도 DA-V2 feature는 여전히 RGB에서 추출합니다.

```powershell
python modules\prepare\depth_refine_v4\run.py --mode infer `
  --input-rgb rgb.png `
  --depth0 outputs\inference\<RUN>\tangent\depth0_z.npy `
  --evaluation-depth compare\depth_z.npy
```

## 산출물

- `depth0_z.npy/png`: frozen DA-V2 기준 깊이
- `depth_edge_diffusion_z.npy/png`: 비학습 edge-aware diffusion baseline
- `depth_zero_condition_z.npy/png`: V2 edge condition을 0으로 둔 ablation
- `depth_final_z.npy/png`: full V4 보정 깊이
- `delta_log_depth.npy/png`: valid 영역 zero-mean residual
- `refinement_gate.npy/png`: residual gate
- `edge_condition.npy/png`: frozen V2 edge 기반 dense condition
- `bev_d0.png`, `bev_edge_diffusion.png`, `bev_zero_condition.png`, `bev_final.png`
- `metrics.json`, `metadata.json`, `index.html`

## 주의

V4는 전역 scale을 새로 추정하지 않습니다. DA-V2의 scale 오류는 별도 calibration 문제로 남겨두고,
이 패키지는 edge 주변의 번짐, 잘못된 layer, boundary consistency를 줄이는 방향만 학습합니다.

