# 3D Edge Completion 가설 검증 실험

이 패키지는 기존 `QuadRayCompletionModel`과 기본 `run.py`를 변경하지 않는 독립 실험입니다.
NYU RGB-D에서 만든 pseudo edge를 사용해 `rgb_local`, `rgb_context` 모델을 학습하고,
fisheye 영상의 실제 픽셀 사이에 있는 subpixel edge ray와 prior-relative z-depth를 예측합니다.

## 실행 위치

아래 명령은 `wide_fov_supervision_v2` 폴더에서 가상환경을 활성화한 뒤 실행합니다.

```powershell
.\.venv\Scripts\Activate.ps1
python modules/prepare/edge_estimate_v2/run.py --mode validate
```

## 전체 학습 및 실험

`all`은 NYU train/test cache 생성, `rgb_local`/`rgb_context` 20 epoch 순차 학습, NYU 평가,
`rgb.png` 추론과 한국어 HTML 보고서 생성을 순서대로 수행합니다.

```powershell
python modules/prepare/edge_estimate_v2/run.py --mode all `
  --input-rgb rgb.png `
  --prior-depth compare/depth_z.npy `
  --evaluation-depth compare/depth_z.npy `
  --base-bev-run outputs/inference/<RUN>
```

DA-V2 cache를 포함한 전체 NYU 실행은 시간이 오래 걸리고 저장 공간을 많이 사용합니다.
중단 후 다시 같은 설정으로 실행하면 이미 생성된 patch와 DA prior cache를 재사용합니다.

현재 모델은 RGB multi-cue 2D edge prior를 `support_edge_2d` 보조 입력으로 사용하고,
query depth는 `D = D0 * exp(delta_log_depth)` 형태로 prior depth를 보정합니다. 이 입력/출력 구조 때문에
이전 edge-estimate checkpoint와 cache는 호환되지 않으며, 먼저 `--mode cache`와 `--mode train`을 다시 실행해야 합니다.

## 단계별 실행

```powershell
# 1. NYU 795/654 frame cache 생성
python modules/prepare/edge_estimate_v2/run.py --mode cache

# 2. DA context를 제외하고 두 모델을 별도 checkpoint로 학습
python modules/prepare/edge_estimate_v2/run.py --mode train --variant rgb_local
python modules/prepare/edge_estimate_v2/run.py --mode train --variant rgb_context

# 3. 최신 best.pt 두 개를 자동 선택해 비교 평가
python modules/prepare/edge_estimate_v2/run.py --mode evaluate --variant all

# 4. 현재 fisheye RGB 추론, prior-depth 기반 투영, simulator depth 평가
python modules/prepare/edge_estimate_v2/run.py --mode infer --variant all `
  --input-rgb rgb.png `
  --prior-depth compare/depth_z.npy `
  --evaluation-depth compare/depth_z.npy `
  --base-bev-run outputs/inference/<RUN>
```

특정 checkpoint를 사용하려면 다음처럼 variant별 경로를 지정합니다.

```powershell
python modules/prepare/edge_estimate_v2/run.py --mode infer --variant rgb_context `
  --checkpoint outputs/edge_estimate/v2/train/rgb_context/<RUN>/checkpoints/best.pt `
  --input-rgb rgb.png `
  --prior-depth compare/depth_z.npy `
  --evaluation-depth compare/depth_z.npy
```

## 빠른 동작 확인

다음은 전체 학습이 아니라 입출력 경로만 확인하는 smoke 설정입니다.

```powershell
python modules/prepare/edge_estimate_v2/run.py --mode cache --skip-da-cache `
  --max-train-frames 1 --max-test-frames 1

python modules/prepare/edge_estimate_v2/run.py --mode train --variant rgb_local --skip-da-cache `
  --max-train-frames 1 --max-test-frames 1 --epochs 1 --batch-size 4 --num-workers 0
```

## 결과 위치와 의미

모든 생성물은 Git ignore 대상인 `outputs/edge_estimate/v2` 아래에 저장됩니다.

- `cache/<hash>`: NYU lattice patch 및 선택적인 DA relative-depth prior
- `train/<variant>/<timestamp>`: epoch checkpoint, `best.pt`, `last.pt`, history
- `eval/<timestamp>`: 세 variant 정량 비교와 paired bootstrap 결과
- `inference/<timestamp>`: subpixel query, near/far depth, camera/world polyline, edge-only/fused BEV, `index.html`

`completed=True`인 query는 카메라에서 직접 관측한 값이 아니라 모델이 완성한 ray입니다.
confidence가 threshold보다 낮은 query는 `unknown=True`로 남고 3D/BEV에 들어가지 않습니다.
`--evaluation-depth`는 평가 GT입니다. `--prior-depth`는 모델 입력 prior입니다.
둘에 같은 `compare/depth_z.npy`를 넣으면 GT depth를 prior로 사용하는 진단 run이라는 의미가 됩니다.

## 프로젝트 V2 실행 명령 모음

다음 명령은 `wide_fov_supervision_v2` 폴더에서 실행합니다. V2의 cache,
checkpoint와 결과는 모두 `outputs/edge_estimate/v2`에서 자동으로 검색하거나 저장합니다.

```powershell
# 환경과 V2 cache 경로 확인
python modules\prepare\edge_estimate_v2\run.py --mode validate

# 기존 전체 학습 rgb_local checkpoint를 자동으로 찾아 fisheye 입력 추론
python modules\prepare\edge_estimate_v2\run.py --mode infer `
  --variant rgb_local `
  --input-rgb rgb.png `
  --prior-depth compare\depth_z.npy `
  --evaluation-depth compare\depth_z.npy

# pinhole 비교 입력 추론
python modules\prepare\edge_estimate_v2\run.py --mode infer `
  --variant rgb_local `
  --input-rgb pinhole.png `
  --prior-depth compare\pinhole_depth.npy `
  --evaluation-depth compare\pinhole_depth.npy

# V2 cache를 처음부터 다시 생성하고 rgb_local을 전체 학습
python modules\prepare\edge_estimate_v2\run.py --mode cache --skip-da-cache
python modules\prepare\edge_estimate_v2\run.py --mode train --variant rgb_local

# V2 rgb_local 정량 평가
python modules\prepare\edge_estimate_v2\run.py --mode evaluate --variant rgb_local
```

자동 검색 대신 특정 V2 checkpoint를 고정하려면 다음처럼 지정합니다.

```powershell
python modules\prepare\edge_estimate_v2\run.py --mode infer `
  --variant rgb_local `
  --checkpoint outputs\edge_estimate\v2\train\rgb_local\2026_07_23_15_21_57\checkpoints\best.pt `
  --input-rgb rgb.png `
  --prior-depth compare\depth_z.npy `
  --evaluation-depth compare\depth_z.npy
```
