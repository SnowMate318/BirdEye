# wide_fov_supervision_v2

어안 영상의 인접 `2×2` pixel ray가 넓게 벌어진 영역에 내부 query ray를 추가하고,
네 corner RGB-D만으로 query의 RGB·z-depth·valid·confidence를 복원하는 실험 코드입니다.

Ray 방향과 개수는 camera geometry와 adaptive sampler가 결정합니다. 학습 모델은 ray를
생성하지 않고, 각 추가 ray의 RGB-D 속성만 예측합니다.

## 실행 환경

PowerShell에서 프로젝트 폴더와 가상환경을 활성화합니다.

```powershell
cd C:\projects\isaac_sim_test\wide_fov_supervision_v2
.\.venv\Scripts\Activate.ps1
```

가상환경이 없다면 Python 3.10으로 새로 만듭니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 빠른 환경 검증

```powershell
python run.py --mode validate
```

Simulator z-depth 파일까지 함께 검사하려면:

```powershell
python run.py --mode validate `
  --depth-source external_npy `
  --depth-npy compare/depth_z.npy
```

## NYU convex-quad cache 생성

학습 전에 frame별 convex quadrilateral 좌표 manifest를 만듭니다.

```powershell
python run.py --mode cache
```

Manifest에는 corner 좌표, 연속 표면 여부, support 3D edge gap만 저장합니다. RGB-D
값은 복제하지 않고 학습 중 `nyu_depth_v2_labeled.mat`에서 직접 sampling합니다.

빠른 smoke cache는 다음처럼 실행합니다.

```powershell
python run.py --mode cache `
  --max-train-items 1 `
  --max-eval-items 1
```

## 모델 학습

```powershell
python run.py --mode train
```

Checkpoint는 아래에 저장됩니다.

```text
outputs/train/YYYY_MM_DD_HH_MM_SS/checkpoints/last.pt
```

1-frame, 1-epoch smoke 학습:

```powershell
python run.py --mode train `
  --max-train-items 1 `
  --epochs 1 `
  --batch-size 2
```

새 모델은 이전 pose-conditioned Refiner checkpoint와 호환되지 않습니다. 잘못된
checkpoint를 지정하면 schema 불일치 오류가 명시적으로 발생합니다.

## NYU 평가

```powershell
python run.py --mode evaluate `
  --checkpoint outputs/train/YYYY_MM_DD_HH_MM_SS/checkpoints/last.pt
```

`metrics.json`에서 completion과 four-corner bilinear baseline의 RGB MAE/PSNR,
depth AbsRel/RMSE, valid/confidence precision·recall을 같은 query 위치에서 비교합니다.

## DA-V2 depth로 추론

```powershell
python run.py --mode infer `
  --checkpoint outputs/train/YYYY_MM_DD_HH_MM_SS/checkpoints/last.pt
```

기본값은 direct와 tangent backbone을 모두 실행하고 tangent D0를 adaptive query
guidance와 최종 BEV의 support depth로 사용합니다.

## Simulator 또는 외부 z-depth로 추론

```powershell
python run.py --mode infer `
  --depth-source external_npy `
  --depth-npy compare/depth_z.npy `
  --checkpoint outputs/train/YYYY_MM_DD_HH_MM_SS/checkpoints/last.pt
```

`depth_z.npy`는 RGB와 같은 `(H,W)` shape의 source-camera z-depth이며 단위는 metre여야
합니다. 외부 metric depth를 넣으면 completion 출력도 같은 metric scale을 따릅니다.

빠른 query/BEV 연결 확인:

```powershell
python run.py --mode infer `
  --depth-source external_npy `
  --depth-npy compare/depth_z.npy `
  --checkpoint outputs/train/YYYY_MM_DD_HH_MM_SS/checkpoints/last.pt `
  --disable-tangent `
  --max-queries 200
```

## 전체 실행

`cache → train → evaluate → infer`를 순서대로 실행합니다.

```powershell
python run.py --mode all
```

전체 NYU cache와 20 epoch 학습이 포함되므로 먼저 smoke 명령으로 환경을 확인하는 것이
좋습니다.

## 주요 옵션

```text
--disable-direct       direct backbone 비활성화
--disable-tangent      tangent backbone 비활성화
--disable-completion   학습 모델 대신 bilinear depth/RGB baseline 사용
--disable-bev          BEV 산출물 비활성화
--disable-html         index.html 생성 비활성화
--max-queries N        inference 추가 query 예산 제한
--epochs N             학습 epoch override
--batch-size N         학습/평가 batch size override
```

Stage별 기본 on/off와 threshold는 `config.py`의 `StageToggles`,
`CompletionConfig`, `RaySamplerConfig`에서 바꿀 수 있습니다.

## 모델 입력과 출력

`modules/quad_completion/model.py`의 공개 인터페이스는 다음과 같습니다.

```python
result = model(
    support_ray_dir,       # (B,4,3)
    support_rgb,           # (B,4,3), 0..1
    support_depth_z,       # (B,4)
    support_valid,         # (B,4)
    query_ray_dir,         # (B,Q,3)
    query_relative_uv,     # (B,Q,2)
    query_mask,            # (B,Q)
)
```

출력은 `rgb`, `depth_z`, `valid_logit`, `confidence_logit`,
`delta_log_depth`입니다. 네 support의 valid median depth로 정규화하고 bilinear
RGB/log-depth를 base로 사용하므로 공통 support scale에 대해 scale-equivariant합니다.
단, DA-V2 자체의 잘못된 전역 scale을 새로 알아내는 모델은 아닙니다.

## Inference 산출물

결과는 `outputs/inference/YYYY_MM_DD_HH_MM_SS/`에 저장됩니다. 가장 먼저
`index.html`을 열면 됩니다.

```text
quad_completion_queries.npz
query_rgb_pred.npy
query_depth_pred_z.npy
query_valid_probability.npy
query_confidence_probability.npy
source_cell_continuous.npy

continuous_only/query_points_camera.npy
continuous_only/query_points_world.npy
continuous_only/bev_rgb.png
continuous_only/bev_valid.png
continuous_only/newly_covered_bev_cells.png

edge_confident/query_points_camera.npy
edge_confident/query_points_world.npy
edge_confident/bev_rgb.png
edge_confident/bev_valid.png
edge_confident/newly_covered_bev_cells.png

quad_sampling_preview.png
completion_rgb_preview.png
completion_depth_preview.png
confidence_map.png
metrics.json
metadata.json
index.html
```

`continuous_only`는 depth-continuous source cell 중 valid/confidence threshold를 통과한
query만 사용합니다. `edge_confident`는 모든 adaptive candidate를 대상으로 같은
threshold를 적용하며 기본 최종 BEV입니다.

`observed_top_occupancy.png`의 검정색은 관측된 top-facing non-floor surface입니다.
Classic free/occupied grid가 아니며 흰색에는 free, non-top, 미관측 상태가 함께 포함됩니다.

## 테스트

```powershell
python -m pytest -q
```

Convex/self-intersection/Jacobian 검사, corner mapping, scale-equivariance, padding mask,
confidence target, NaN-safe gradient, sampler 결정성, checkpoint schema를 검증합니다.

## Git 제외 항목

`.gitignore`는 `.venv/`, `outputs/`, `compare/`, NYU 원본, checkpoint, `*.npy`,
`*.npz`, point cloud 등 대용량 로컬 산출물을 제외합니다. Push 전에는 다음으로
확인합니다.

```powershell
git status --short
```
