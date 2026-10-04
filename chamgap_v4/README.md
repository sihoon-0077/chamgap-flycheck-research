# CHAMGAP × FlyCheck v4 — 연구 코어와 통합 참조 구현

전체 설명은 `CHAMGAP_FlyCheck_FULL_SYSTEM_MASTER_v4.md`에 있다.
본 폴더는 완성된 제어 제품이 아니다. 실제 센서/펌프 동작 없이 실행할 수 있는
시뮬레이션, 학습, 추천 API, 영속 dry-run 원장과 센서 초기 점검 참고 코드다.

## 빠른 실행

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-integration.txt
python -m pytest -q
python -m integration.demo
python -m chamgap.connectome demo --out data/demo_mask.npy --pn 32 --kc 128
python -m chamgap.collect --episodes 240 --replicas 3 --seed 17 --out data/sim_train.npz
python -m chamgap.train --kind mlp --data data/sim_train.npz --epochs 30 --out runs/mlp.pt
python -m chamgap.train --kind fly --matrix data/demo_mask.npy --data data/sim_train.npz --epochs 30 --out runs/fly.pt
python -m chamgap.evaluate --episodes 90 --out runs/fixed_eval.json
python -m chamgap.evaluate --checkpoint runs/fly.pt --episodes 90 --out runs/fly_eval.json
```

## 출처와 준비 상태

- 데모 projection은 `DEMO_RANDOM_NOT_BIOLOGICAL`이며 실제 초파리 연결 데이터가 아니다.
- 시뮬레이터는 실물에 아직 적합하지 않은 작은 모델이다.
- `backend/api.py`는 추천 전용이며, 펌프를 작동시키지 않는다.
- `integration/store.py`는 SQLite 원자적 예산 예약·중복 검사 **dry-run**이다.
- `integration/collector.py`는 read-only MQTT collector 참고 구현이다. 실제 broker 연결은 별도 검수한다.
- `infra/schema_v4.sql`, `infra/*.example`은 PostgreSQL/Mosquitto 배포 설계 예시다.
- `firmware_reference/src/sensor_bringup.ino`는 펌프 출력을 항상 LOW로 두는 센서 점검 예시다.
  실제 Arduino 빌드와 ESP32 업로드는 아직 하지 않았다.
- 완성 live executor, 기기 영속 원장, MQTT production firmware, 프런트엔드,
  실제 커넥톰 추출·실물 평가·기구 하중 검수는 팀이 수행할 후속 작업이다.

실제 데이터 추출은 검토된 뉴런 선택 ID와 계정·라이선스 검토 후 `tools.fetch_connectome`을 사용한다.
토큰은 소스/논문/MD에 넣지 않는다. 모델 가중치는 자신이 생성하고 검수한 파일만 불러온다.
검증 결과는 `docs/VERIFICATION_V4.json`, 이전 제공물의 결과는 `docs/SMOKE_TEST_REPORT.json`로 구분한다.
