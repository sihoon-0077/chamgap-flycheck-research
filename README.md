# 참값 × FlyCheck

스마트팜 센서가 이상해 보일 때 **무엇부터 확인할지** 고르는 작은 의사결정기(FlyCheck)와, 그것을 시험하기 위한 3구역 실험대의 설계·코드·실험 기록입니다. 캡스톤 디자인 프로젝트입니다.

- 결과·부품 문서 사이트: https://chamgap-flycheck-site.vercel.app (`/results`, `/parts`)
- 사이트 저장소: https://github.com/sihoon-0077/chamgap-flycheck-site

> **읽기 전에.** 여기 있는 성능 수치는 모두 보정되지 않은 가상 시뮬레이터 결과입니다(`domain=SIM_ONLY_UNCALIBRATED`). 실제 현장 성능이 아닙니다. 난수 연결 모델은 `connectome=DEMO_RANDOM_NOT_BIOLOGICAL`로 표시하며, 실제 초파리 연결에 대한 증거가 아닙니다.

## 구성

| 경로 | 내용 |
|---|---|
| `CHAMGAP_FlyCheck_FULL_SYSTEM_MASTER_v4.1 (1).md` | 전체 설계서 v4.1. 부록 A에 코드 원본과 SHA-256이 들어 있습니다 |
| `extract_master.py` | 설계서 부록 A에서 코드를 복원하는 추출기 (A.1절과 같은 코드) |
| `chamgap_v4/` | 부록 A에서 복원한 코드. 52개 파일 모두 설계서에 적힌 SHA-256과 일치합니다 |
| `chamgap_v4/experiments/` | 이 저장소에서 추가한 실험 코드. 부록 코드는 수정하지 않고, 바꿀 동작은 실행 중에만 적용합니다 |
| `chamgap_v4/results/` | 실험 결과 (요약 CSV, 그래프, 사건별 원자료, manifest) |
| `research.md` | 날짜·시각별 연구 기록: 무엇을 했고, 결과와 결정이 무엇이었는지 |
| `slides/` | 진행 상황 발표 자료 |

## 실험

| 실험 | 실행 코드 | 결과 |
|---|---|---|
| 16.3절 기본 파이프라인 (고정·MLP·fly 정책 평가) | `python -m chamgap.*` (설계서 16.3절) | `chamgap_v4/runs/`는 올리지 않음. 수치는 `research.md` |
| 5.4절 학습곡선 (난수 연결 데모, 보상 v1) | `experiments/learning_curve.py` → `analyze_learning_curve.py` | `results/learning_curve*` |
| connectome_v1 (실제 hemibrain PN→KC 연결, 보상 v2 + 규칙 A) | `experiments/connectome_curve.py` → `analyze_connectome_curve.py` | `results/connectome_v1/` |

- connectome_v1의 가설, 판정 규칙, 변경 기록은 실험 전에 `chamgap_v4/experiments/PREREG_connectome_v1.md`에 고정했습니다. 결과 해석은 그 판정 문장으로만 합니다.
- 실험 설정 v2(보상과 규칙 A)는 `experiments/sim_v2.py`에 있습니다.

## 다시 실행하기

```bash
cd chamgap_v4
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1   macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-integration.txt
python -m pytest -q        # 46 passed
```

학습곡선 실험을 다시 돌리려면 16.3절 [2] 단계로 데모 연결(`data/demo_mask.npy`)을 먼저 만든 뒤 아래를 실행합니다.

```bash
python -m chamgap.connectome demo --out data/demo_mask.npy --pn 32 --kc 128
python -m experiments.learning_curve && python -m experiments.analyze_learning_curve
```

같은 시드로 다시 실행했을 때 사건별 결과가 완전히 같은지 확인한 기록이 `research.md`에 있습니다.

## 올리지 않은 것

| 제외 항목 | 이유와 대체 방법 |
|---|---|
| `.env` | neuPrint 토큰이 들어 있습니다. 직접 쓰려면 `.env.example`을 복사해 본인 토큰을 넣으세요 |
| `chamgap_v4/data/`, `chamgap_v4/runs/` | 학습 데이터, 연결 행렬, 모델 체크포인트. 위 스크립트로 다시 만들 수 있습니다. 실제 연결 행렬은 `experiments/build_real_connectome.py`와 본인의 neuPrint 토큰으로 다시 받습니다 |
| `site/` | 별도 저장소입니다 (Vercel 배포) |

## 데이터 출처

실제 연결 실험은 Janelia FlyEM hemibrain v1.2.1 데이터를 neuPrint(https://neuprint.janelia.org)로 조회해 사용했습니다. 이 데이터는 CC BY 4.0 라이선스입니다.

> Scheffer, L. K. et al. (2020). A connectome and analysis of the adult *Drosophila* central brain. *eLife* 9:e57443. https://doi.org/10.7554/eLife.57443

PN·KC 선택 규칙은 `experiments/build_real_connectome.py`에 있으며, 팀 검토 전이라 `review_status=PENDING_TEAM_REVIEW`로 표시합니다.

## 라이선스

이 저장소의 코드와 문서에는 아직 라이선스 파일이 없습니다. 팀이 정한 뒤 추가합니다. hemibrain에서 나온 데이터와 결과는 위 CC BY 4.0 출처 표기를 따릅니다.
