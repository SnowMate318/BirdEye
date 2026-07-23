# Edge estimate 실험 버전 관리 규칙

각 프로젝트 실험 버전은 모델 코드, cache schema, checkpoint를 서로 공유하지 않는다.
checkpoint 내부의 `edge_estimate_*_v3` 같은 문자열은 모델 파일 schema이며,
이 문서의 프로젝트 버전 `V2`, `V3`와는 별개의 값이다.

## 디렉터리 규칙

```text
modules/prepare/edge_estimate_v2/   # 프로젝트 V2의 고정 코드
modules/prepare/edge_estimate/      # 현재 개발 중인 버전

outputs/edge_estimate/v2/cache/
outputs/edge_estimate/v2/train/
outputs/edge_estimate/v2/eval/
outputs/edge_estimate/v2/inference/
```

다음 버전을 고정할 때는 `edge_estimate_v4`, `edge_estimate_v5`처럼 새 패키지로
복사하고, 해당 패키지의 `EdgeEstimateConfig.output_root`도 각각 `v4`, `v5`로
지정한다. 버전 패키지끼리 model, dataset, losses, pipeline을 import하지 않는다.

## V2 실행

```powershell
python modules\prepare\edge_estimate_v2\run.py --mode validate
python modules\prepare\edge_estimate_v2\run.py --mode cache
python modules\prepare\edge_estimate_v2\run.py --mode train --variant all
python modules\prepare\edge_estimate_v2\run.py --mode infer --variant all
```

V2의 cache와 checkpoint는 항상 `outputs/edge_estimate/v2`에서 자동 검색한다.
