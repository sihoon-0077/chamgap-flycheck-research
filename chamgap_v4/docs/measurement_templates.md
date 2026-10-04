# 데이터 양식

## 독립 저울로 펌프 검수: pump_trials.csv

```csv
pump_id,duration_s,delivered_g
```

최소 3개 운전시간 × 2회는 코드 실행의 시작 조건일 뿐 충분한 보정/통계 검증을 뜻하지 않는다.

## 정상 반응: normal_response.csv

```csv
session_id,verified_normal,soil_before,delivered_estimate_ml,elapsed_s,temperature_c,humidity_pct,soil_after
```

verified_normal은 검수 근거에 따라 1/0. 운영 모델 입력에 넣지 않는다.
soil_before/after는 0~1 상대 지표이며 VWC로 보정하지 않았다면 체적함수율로 표시하지 않는다.

## 실측 점검 원기록

```csv
session_id,step_id,zone_id,policy_version,observation_path,available_actions,selected_action,requested_at,completed_at,command_ml,measured_delivery_ml,decision,decision_reason,domain
```

추가 측정값은 별도 원시 테이블과 ID로 연결한다. 고장 정답·기준 계측은 QA 계정에 분리하고 시험 후 결합한다.
실제로 실행하지 않은 행동의 결과는 미관측이며 0점 정답이 아니다. 학습 시 target_mask=false로 표시한다.
