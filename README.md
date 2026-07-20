# wide_fov_supervision_v2

어안 영상에서 ray 간격이 sparse해지는 영역을 보완하고, Depth Anything V2 / DSINE prior와 ray-aware refiner를 이용해 z-depth, normal, BEV 결과를 만드는 실험 코드입니다.

이 README는 실행 방법 중심으로 정리합니다. 데이터셋, checkpoint, cache, inference 결과는 `.gitignore`에 의해 저장소에 포함하지 않습니다.

## 1. 폴더 이동

PowerShell 기준으로 실행합니다.

```powershell
cd C:\projects\isaac_sim_test\wide_fov_supervision_v2
```

## 2. 가상환경 준비

이미 `.venv`가 있으면 활성화만 하면 됩니다.

```powershell
.\.venv\Scripts\Activate.ps1
```

새로 만들 때는 Python 3.10 환경에서 다음처럼 설치합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt`는 CUDA 12.8용 PyTorch 2.7.0을 기준으로 되어 있습니다.

## 3. 실행 전 검증

경로, checkpoint, fisheye ray round-trip을 빠르게 확인합니다.

```powershell
python run.py --mode validate
```

정상이라면 `input_rgb_exists`, `depth_anything_vitl_ckpt_exists`, `dsine_ckpt_exists` 등이 `true`로 나옵니다.

## 4. 기본 추론

기본값은 DA-V2 기반 depth source를 사용합니다.

```powershell
python run.py --mode infer
```

결과는 timestamp 폴더에 저장됩니다.

```text
outputs/inference/YYYY_MM_DD_HH_MM_SS/
```

가장 먼저 열어볼 파일은 다음입니다.

```text
outputs/inference/YYYY_MM_DD_HH_MM_SS/index.html
```

## 5. 외부 depth_z.npy로 추론

Isaac 등에서 만든 z-depth를 임시로 비교하고 싶을 때 사용합니다.

```powershell
python run.py --mode infer `
  --depth-source external_npy `
  --depth-npy compare/depth_z.npy
```

`compare/`는 실험용 로컬 산출물이므로 Git에는 포함하지 않습니다.

## 6. 빠른 디버그 추론

direct branch를 끄고 tangent branch만 확인하려면:

```powershell
python run.py --mode infer `
  --disable-direct
```

HTML 생성을 끄려면:

```powershell
python run.py --mode infer `
  --disable-html
```

BEV 생성을 끄려면:

```powershell
python run.py --mode infer `
  --disable-bev
```

dense BEV coverage 보완을 끄려면:

```powershell
python run.py --mode infer `
  --disable-dense-coverage
```

dense source-cell subdivision을 바꾸려면:

```powershell
python run.py --mode infer `
  --dense-subdivision 7
```

기본값은 `5`입니다. 값을 키우면 coverage는 늘 수 있지만 실행 시간이 길어집니다.

## 7. NYU 학습 cache 생성

정식 학습 전에는 teacher cache와 query sidecar cache가 필요합니다.

```powershell
python run.py --mode cache
```

이 단계는 다음을 수행합니다.

```text
NYU RGB-D 로드
virtual fisheye 생성
DA-V2 D0 생성
DSINE N0 생성
teacher cache 저장
query sidecar 저장
```

cache는 아래에 생성되며 Git에는 포함하지 않습니다.

```text
outputs/cache/
```

## 8. Refiner 학습

cache 생성이 끝난 뒤 학습합니다.

```powershell
python run.py --mode train
```

학습된 checkpoint는 다음 위치에 저장됩니다.

```text
outputs/train/YYYY_MM_DD_HH_MM_SS/checkpoints/last.pt
```

짧은 smoke 학습만 돌리고 싶으면:

```powershell
python run.py --mode train `
  --max-train-items 1 `
  --epochs 1 `
  --batch-size 1
```

정식 학습에서는 `--max-train-items`를 빼고 기본 설정을 사용합니다.

## 9. 학습 checkpoint로 추론

학습된 refiner를 inference에 반영하려면 `--checkpoint`를 명시해야 합니다.

```powershell
python run.py --mode infer `
  --checkpoint outputs/train/YYYY_MM_DD_HH_MM_SS/checkpoints/last.pt
```

checkpoint를 지정하지 않으면 refiner는 학습 파라미터를 로드하지 않습니다. 이 경우 결과는 주로 foundation model D0, analytic ray, adaptive/dense BEV postprocess에 의해 결정됩니다.

## 10. 전체 파이프라인 한 번에 실행

cache, train, evaluate, infer를 모두 실행합니다.

```powershell
python run.py --mode all
```

시간이 오래 걸릴 수 있으므로 처음에는 `validate`, `cache`, `train`, `infer`를 단계별로 확인하는 것을 권장합니다.

## 11. 주요 산출물 의미

inference run 폴더에서 자주 보는 파일은 다음입니다.

```text
source_rgb.png
  입력 RGB입니다.

tangent/depth0_z.png
  tangent branch에서 만든 source z-depth D0입니다.

tangent/normal0.png
  DSINE 기반 source normal N0입니다.

added_ray_density.png
  최종 추가 query ray 분포입니다.

adaptive_added_ray_density.png
  Ray-aware Refiner에 들어간 adaptive query 분포입니다.

dense_added_ray_density.png
  BEV coverage 보완용 dense source-cell query 분포입니다.

bev_valid_before.png
  추가 ray 적용 전 BEV support coverage입니다.

bev_valid_after.png
  추가 ray 적용 후 BEV support coverage입니다.

newly_covered_bev_cells.png
  추가 ray로 새롭게 채워진 BEV cell입니다.

bev_rgb.png
  최종 RGB BEV입니다.

observed_top_occupancy.png
  관측된 top-facing non-floor surface입니다.
  classic free/occupied grid가 아닙니다.

observed_support_occupancy.png
  top-facing 여부와 무관하게 최종 ray가 관측/보완한 BEV support입니다.
  coverage 변화를 직접 보고 싶을 때 이 파일을 확인합니다.

top_probability_map.png
  top-facing 판단 점수 지도입니다.

metrics.json
  query 수, BEV coverage 변화, checkpoint 로드 여부 등 정량 지표입니다.

metadata.json
  실행 설정과 각 산출물 의미를 기록합니다.

index.html
  위 결과를 한국어 dashboard로 묶은 파일입니다.
```

## 12. Git에 포함하지 않는 항목

다음은 `.gitignore`로 제외합니다.

```text
.venv/
outputs/
compare/
rgb.png
data/
dataset/
datasets/**/raw/
datasets/**/processed/
checkpoints/
weights/
*.pt, *.pth, *.ckpt
*.npy, *.npz
*.mat, *.h5, *.hdf5
```

이미 Git에 추가된 대용량 파일은 `.gitignore`만으로 제거되지 않습니다. 파일은 유지하고 Git 추적만 끊으려면 다음처럼 실행합니다.

```powershell
git rm --cached -r outputs
git rm --cached -r .venv
git rm --cached -r compare
git rm --cached rgb.png
```

필요한 항목만 선택해서 실행하세요.

## 13. GitHub push 예시

저장소가 아직 초기화되지 않았다면:

```powershell
git init
git remote add origin https://github.com/SnowMate318/BirdEye.git
git add .
git commit -m "Add wide_fov_supervision_v2 pipeline"
git branch -M main
git push -u origin main
```

이미 remote가 있다면:

```powershell
git add .
git commit -m "Update wide_fov_supervision_v2 pipeline"
git push
```

push 전에 `git status --short`로 `outputs/`, `.venv/`, `*.pt`, `*.npy`, `*.npz`가 올라가지 않는지 확인하세요.

## 14. 테스트

단위 테스트는 다음처럼 실행합니다.

```powershell
python -m pytest tests -q
```

프로젝트 root인 `wide_fov_supervision_v2` 안에서 실행하는 것을 기준으로 합니다.
