"""Export site/parts/data/parts.json: what we buy, what each part does, how it connects.

Commercial fields (qty, unit price, total, vendor, url) are read from
docs/procurement_snapshot_20260929.csv, never retyped. Role / connection / check text
comes from the master document (sections 2.1, 19, 20, 29). Wiring is a REVIEW PROPOSAL:
final pins are fixed only after the actual boards are inspected (20.2).

Run from chamgap_v4/:
  python -m experiments.export_site_parts --out ../site/parts/data
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

SOURCE = Path('docs/procurement_snapshot_20260929.csv')
CATEGORIES = [
    {'id': 'E', 'name': '센서·제어·계량', 'doc_subtotal': 351771},
    {'id': 'P', 'name': '전원·보호', 'doc_subtotal': 49960},
    {'id': 'C', 'name': '배선·함체', 'doc_subtotal': 39810},
    {'id': 'W', 'name': '물길·재배', 'doc_subtotal': 63350},
]
BUDGET = {'base': 504891, 'spares': 107489, 'base_plus_spares': 612380, 'limit': 1100000,
          'phase1': 156977, 'phase2': 347914}
SPARES = {'E01': 1, 'E02': 1, 'E03': 2, 'E04': 1, 'E05': 1, 'E06': 1, 'E07': 1, 'P03': 1}   # 19.2

# id -> display content. Sources: 2.1, 19.1/19.5, 20.1-20.5, 29.2-29.6.
CONTENT = {
    'E01': dict(name='ESP32 개발보드', model='ESP32 DevKitC-32 · USB-C · CH340C', icon='esp32',
                role='구역마다 1대씩 두는 구역 노드입니다. 센서 값을 모아 MQTT(telemetry.v4)로 노트북에 보내고, 명령을 받아 펌프를 켭니다. 서버가 멈춰도 펌프가 멈추도록 부팅 시 OFF, 하드 타임아웃, 정지 입력 확인을 보드에서 직접 합니다.',
                connections=['USB-C ← 5V USB 어댑터(P02) + 케이블(P03)', 'GPIO21 SDA / GPIO22 SCL ↔ 온습도 A·B (I2C)',
                             'GPIO32 / GPIO33 ← 수분 A·B (아날로그, ADC1)', 'GPIO26 DT / GPIO27 SCK ↔ HX711',
                             'GPIO25 → DFR0457 제어 신호 (펌프 명령)', 'GPIO34 ← 비상정지 보조 접점, GPIO35 ← 누수 감지'],
                checks=['핀 번호는 클래식 ESP32 기준 제안입니다. 받은 보드의 실크와 칩 문서로 확정합니다', '3.3V GPIO에 5V 신호를 직접 연결하지 않습니다',
                        'GPIO34·35는 입력 전용이고 내부 풀업이 없어 외부 10kΩ이 필요합니다'],
                harness=['H01–H13'], phase1=1),
    'E02': dict(name='온습도 센서', model='Sensirion SHT31-D · I2C', icon='sht31',
                role='구역마다 대상(A)·비교(B) 2개를 나란히 두고 온습도를 잽니다. 두 값이 어긋나면 센서 이상을 의심하는 근거가 됩니다.',
                connections=['VIN / GND ← ESP32 3V3 / GND (H01, H03)', 'SDA / SCL → GPIO21 / GPIO22, 두 개가 같은 I2C 버스 (H02, H04)',
                             'I2C 주소: A 0x44, B 0x45'],
                checks=['모듈의 ADDR 설정으로 주소를 다르게 할 수 있는지 확인합니다. 안 되면 멀티플렉서나 버스 분리가 필요합니다', '풀업이 3.3V로 연결되는지 측정합니다',
                        '전장 박스 안이 아니라 박스 밖의 같은 통풍 조건 지그에 설치합니다 (보드 발열을 재지 않도록)', '방수 등급이 확인되지 않았습니다. 분무나 침수 금지'],
                harness=['H01', 'H02', 'H03', 'H04'], phase1=2),
    'E03': dict(name='정전용량 토양수분 센서', model='DFRobot SEN0193 · 케이블 포함', icon='soil',
                role='같은 화분에 대상(A)·비교(B) 2개를 꽂아 수분을 잽니다. 두 값의 비교와, 급수 뒤 수분이 오르는 반응을 기록합니다.',
                connections=['전원 / GND ← ESP32 3V3 / GND', '아날로그 출력: A → GPIO32, B → GPIO33 (H05, H06)'],
                checks=['포함 케이블의 선 배열과 극성을 확인합니다', '상부 PCB 경계선 위로는 흙에 묻지 않게 합니다',
                        'Wi-Fi와 함께 쓰므로 ADC1 핀(GPIO32·33)을 씁니다. 원시 ADC와 출력 전압의 대응을 보정합니다'],
                harness=['H05', 'H06'], phase1=2),
    'E04': dict(name='12V 연동 도우징 펌프', model='peristaltic dosing pump 12VDC', icon='pump',
                role='구역마다 따로 정해진 양만 시험 급수합니다. 초기 제안 한도는 1회 10mL, 세션 누적 20mL, 최대 2회, 연속 운전 60초입니다.',
                connections=['12V 분기 퓨즈 → DFR0457 출력 → 펌프 (+/−)', '펌프 단자 양단에 1N4007 역병렬 (P05)',
                             '물길: 물통 → 실리콘 튜브 → 펌프 → 고정 출수구 → 화분'],
                checks=['기어모터·워터펌프가 아니라 연동(peristaltic) 펌프입니다', '펌프 헤드의 튜브 외경(4.7mm)은 W01(4.5mm)과 달라 헤드 교체용으로 바로 쓰지 않습니다',
                        '퓨즈 용량은 어댑터 정격이 아니라 펌프 실측 전류로 정합니다'],
                harness=['전력 경로 29.4'], phase1=1),
    'E05': dict(name='MOSFET 전력 구동 모듈', model='DFRobot DFR0457', icon='mosfet',
                role='ESP32의 작은 3.3V 신호로 12V 펌프 전원을 켜고 끄는 스위치입니다.',
                connections=['제어: GPIO25 / GND → 신호 / GND (H10), 신호 기준 접지를 ESP32와 공유', '전력 입력: 12V 분기 퓨즈 → DFR0457',
                             '전력 출력: DFR0457 → 펌프'],
                checks=['단자 이름은 제조사 안내로 확인합니다. 일반 릴레이 보드와 같은 배선이라고 가정하지 않습니다', 'LOW에서 OFF인지, HIGH 극성을 확인합니다',
                        'GPIO25에 외부 10kΩ 풀다운을 달아 부팅 중 펌프가 켜지지 않게 합니다', '제어 측 VCC는 5V가 아니라 ESP32 3V3을 씁니다'],
                harness=['H10'], phase1=1),
    'E06': dict(name='4선식 로드셀 10kg', model='PP-A410 · Full Bridge', icon='loadcell',
                role='화분과 물받이를 올린 계량 받침의 무게 증가를 재서, 물이 실제로 도착했다는 보조 증거를 얻습니다.',
                connections=['4선 E+ / E− / A+ / A− → HX711 (H09)', '한쪽 끝은 고정 베이스, 다른 쪽 끝은 하중을 받는 상판에 고정'],
                checks=['약 10g 증가가 잡음·크리프·선 장력보다 충분히 큰지 실측합니다', '선 색상은 데이터시트를 우선합니다',
                        '중앙 변형부를 볼트나 판으로 눌러 막지 않습니다', '10kg 정격이 측정 정밀도를 보장하지는 않습니다'],
                harness=['H09'], phase1=1),
    'E07': dict(name='로드셀 신호 변환 모듈', model='HX711', icon='hx711',
                role='로드셀의 아주 작은 전압 변화를 증폭해 디지털 측정값으로 바꿉니다.',
                connections=['E± / A± ← 로드셀 4선 (H09)', 'DT → GPIO26, SCK ← GPIO27 (H08)', 'VCC / GND ← 검수된 로직 전원 (H07)'],
                checks=['출력 HIGH가 3.3V GPIO에 안전한지 실측합니다', '신호선은 짧게, 모터선과 떨어뜨립니다'],
                harness=['H07', 'H08'], phase1=1),
    'P01': dict(name='12V 5A 어댑터', model='프라임디렉트 · 코드일체형 · 5.5×2.1mm', icon='adapter',
                role='펌프 3대의 구동 전원입니다. AC 단자가 노출되지 않는 완제품 어댑터를 씁니다.',
                connections=['(+) → DC 잭 단자(P04) → 주 퓨즈 → 비상정지 차단 경로 → 분배 단자대 → 구역별 분기 퓨즈', '(−) → 검수된 공통 귀환점'],
                checks=['어댑터의 5A 표기가 분기나 스위치의 차단 능력을 뜻하지는 않습니다'], harness=['전력 경로 29.4'], phase1=1),
    'P02': dict(name='5V USB 어댑터', model='마하링크 오비투 LX050200 · 5V 2A', icon='usbpower',
                role='ESP32 3대에 각각 독립된 5V 전원을 공급합니다. 펌프 전원과 분리됩니다.',
                connections=['→ USB 케이블(P03) → ESP32 USB-C (H13)'], checks=['펌프용 12V 어댑터와 섞어 쓰지 않습니다'], harness=['H13'], phase1=1),
    'P03': dict(name='USB 데이터 케이블', model='USB-A → USB-C · USB2.0 · 약 0.9m', icon='usbcable',
                role='펌웨어 업로드와 ESP32 전원 공급에 씁니다.',
                connections=['5V USB 어댑터(P02) 또는 노트북 → ESP32 USB-C'],
                checks=['충전 전용 케이블과 섞이지 않게 표시합니다', '모터 전원을 물리적으로 끊은 상태에서 업로드합니다'], harness=['H13'], phase1=1),
    'P04': dict(name='DC 잭 → 나사단자 어댑터', model='5.5×2.1mm Female', icon='dcjack',
                role='12V 어댑터 선을 자르지 않고 저전압 분배부에 연결합니다.',
                connections=['12V 어댑터(P01) 플러그 → 나사단자 → 주 퓨즈'], checks=['+/− 단자 표시를 실측으로 확인합니다'], harness=['전력 경로 29.4'], phase1=1),
    'P05': dict(name='정류 다이오드', model='1N4007 · 20개 묶음', icon='diode',
                role='펌프를 끌 때 생기는 역기전력에서 회로를 보호합니다.',
                connections=['펌프 단자 양단에 역병렬로 연결합니다. 줄무늬(캐소드) 쪽이 + 단자입니다'],
                checks=['펌프와 직렬로 달지 않습니다', '모든 펌프나 PWM에 자동으로 적합하지는 않습니다. 정격을 검토합니다'], harness=['전력 경로 29.4'], phase1=1),
    'P06': dict(name='비상정지 스위치 (후보)', model='KGE-H4R1R · 22mm · 1a1b · PP-B057', icon='estop',
                role='소프트웨어와 관계없이 사람이 눌러 펌프 에너지를 끊는 물리 정지 장치입니다.',
                connections=['주 접점: 12V 주 퓨즈 다음의 차단 경로', '보조 접점: GPIO34 입력 (외부 10kΩ 풀업) → 보드가 정지 상태를 확인 (H11)'],
                checks=['이 스위치가 DC를 직접 차단할 수 있는지는 미확정입니다 (19.4)', '에너지 차단 접점과 상태 확인 접점을 한 접점으로 겸용하지 않습니다'],
                harness=['H11'], phase1=1),
    'P07': dict(name='인라인 퓨즈 홀더 (후보)', model='BR-FHAM1 · 자동차 퓨즈 홀더', icon='fuse',
                role='과전류를 막습니다. 주 전원에 1개, 구역별 펌프 분기에 1개씩 모두 4개를 둡니다.',
                connections=['주 전원 1: DC 잭 단자 다음', '분기 3: 분배 단자대 → 각 구역 DFR0457'],
                checks=['퓨즈 자체의 용량은 펌프 실측 전류로 정하며 아직 사지 않았습니다 (19.4)'], harness=['전력 경로 29.4'], phase1=2),
    'C01': dict(name='점퍼선 암-암', model='PP-A398 · 40핀 20cm', icon='jumper',
                role='모듈 핀끼리 이어 초기 연결을 시험합니다.', connections=['센서 모듈 ↔ 센서 모듈 / 헤더'],
                checks=['펌프 전류가 점퍼선을 지나가게 최종 배선하지 않습니다'], harness=[], phase1=1),
    'C02': dict(name='점퍼선 암-수', model='PP-A397 · 40핀 20cm', icon='jumper',
                role='센서, 개발보드, 브레드보드 사이 신호를 잇습니다.', connections=['센서 모듈 핀 ↔ 브레드보드'],
                checks=['신호선만 씁니다. 12V 전력선으로 쓰지 않습니다'], harness=[], phase1=1),
    'C03': dict(name='점퍼선 수-수', model='PP-A396 · 40핀 20cm', icon='jumper',
                role='브레드보드의 전원·신호 연결과 회로 점검에 씁니다.', connections=['브레드보드 ↔ 브레드보드'],
                checks=['신호선만 씁니다'], harness=[], phase1=1),
    'C04': dict(name='열수축 튜브', model='PP-K022 · 1~14mm · 164pcs', icon='heatshrink',
                role='납땜한 접속부를 절연하고 배선을 마감합니다.', connections=['만능기판·하네스의 납땜부'], checks=[], harness=[], phase1=1),
    'C05': dict(name='브레드보드 830점', model='PP-A384 · MB-102', icon='breadboard',
                role='구역별 센서 회로를 납땜 전에 먼저 꽂아서 시험합니다.', connections=['ESP32와 센서 신호 (sensor-only 단계)'],
                checks=['펌프 전류를 브레드보드로 흘리지 않습니다'], harness=[], phase1=1),
    'C06': dict(name='에폭시 단면 만능기판', model='PP-D014 · 2.54mm · 17×36홀', icon='perfboard',
                role='브레드보드에서 검증한 센서 연결 회로를 납땜해 고정합니다.', connections=['센서 커넥터·풀업/풀다운 저항·ESP32 헤더'],
                checks=['커넥터마다 하네스 ID를 양쪽에 붙입니다', '12V와 3.3V에 같은 모양 커넥터를 쓰지 않습니다'], harness=[], phase1=1),
    'C07': dict(name='전장 컨트롤 박스', model='BOXCO BC-AGS-152010 · 15×20×10cm', icon='box',
                role='구역별로 보드, 계량 모듈, 배선을 보호하고 고정합니다.', connections=['프레임 후면 상부 캐리어에 3개를 나란히 설치 (29.2)'],
                checks=['온습도 센서는 박스 안에 넣지 않습니다', '글랜드, 뚜껑 여유, 케이블 당김 보호를 확인합니다'], harness=[], phase1=1),
    'W01': dict(name='실리콘 튜브 10m', model='내경 2.5 / 외경 4.5mm', icon='tube',
                role='물통에서 펌프, 펌프에서 출수구까지 구역별 급수 연장선을 만듭니다.', connections=['물통 → 펌프 입구', '펌프 출구 → 고정 출수구'],
                checks=['펌프 헤드 교체용이 아닙니다 (외경 다름)', '튜브가 화분이나 계량판을 누르지 않게 고정합니다'], harness=[], phase1=1),
    'W02': dict(name='PE 광구 물통 5L', model='광구병 5L', icon='tank',
                role='펌프 3대가 함께 쓰는 물 저장조입니다.', connections=['튜브 3개가 각 구역 펌프로 따로 나감'],
                checks=['뚜껑을 덮어 씁니다', '펌프 이후 물길은 합치지 않습니다'], harness=[], phase1=1),
    'W03': dict(name='사각 화분 18cm', model='두이츠 · 윗지름 18cm', icon='pot',
                role='3구역에 같은 규격의 용기를 씁니다.', connections=['계량 받침 상판 위에 물받이와 함께 놓음'],
                checks=['배수되는 물도 같은 계량 상판에서 받도록 구성합니다'], harness=[], phase1=1),
    'W04': dict(name='마이그린 배양토 10L', model='10L 포장', icon='soil_bag',
                role='3구역에 같은 배지를 채우고, 반복 실험 때 초기 조건을 다시 만듭니다.', connections=['화분 3개'], checks=[], harness=[], phase1=1),
}

PENDING = [  # 19.4: not bought or not priced yet (price is "미확정", not zero)
    {'name': '10kΩ 저항', 'icon': 'resistor', 'spec': '구역당 3개 (GPIO25 풀다운 1, GPIO34·35 풀업 2), 3구역 + 예비 = 12개 이상, 1/4W 금속피막',
     'needs': '1차 구매에 반드시 포함', 'required_first': True},
    {'name': '누수 감지 센서', 'icon': 'leak', 'spec': 'GPIO35 입력. 판매 모델 미확정', 'needs': '회로·출력 로직·단선 대응 검토', 'required_first': False},
    {'name': 'DC 물리 차단 경로', 'icon': 'estop', 'spec': '스위치/접촉기 등 정격 검토 후 선정', 'needs': 'P06이 직접 차단할 수 있는지 확인', 'required_first': False},
    {'name': '퓨즈·전력선·단자', 'icon': 'fuse', 'spec': '주전원 1, 분기 3, 예비 포함', 'needs': '기동 전류·정격·선경·접점 검토', 'required_first': False},
    {'name': '2020 프레임', 'icon': 'frame', 'spec': '120cm×4, 76cm×6, 36cm×8, 총 12.24m', 'needs': '홈·체결·절단·배송·보강 포함 견적', 'required_first': False},
    {'name': '계량 지그', 'icon': 'loadcell', 'spec': '22×26cm 상·하판 각 3세트', 'needs': '로드셀 고정 홀·변형 여유·스토퍼', 'required_first': False},
    {'name': '물받이·누수 트레이', 'icon': 'pot', 'spec': '화분별 3 + 전체 하부 1', 'needs': '계량판 간섭·배수 포집·용량', 'required_first': False},
    {'name': '피팅·클램프', 'icon': 'tube', 'spec': '내경 2.5mm 연결구 최소 6, 클램프 3', 'needs': '헤드 튜브와 연장 튜브 결합 실물 검수', 'required_first': False},
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    rows = list(csv.DictReader(SOURCE.open(encoding='utf-8')))
    if {r['id'] for r in rows} != set(CONTENT):
        raise ValueError(f'CSV ids and CONTENT ids differ: {sorted({r["id"] for r in rows} ^ set(CONTENT))}')
    items = []
    for r in rows:
        qty, unit, total = int(r['qty']), int(r['price']), int(r['total'])
        if qty * unit != total:
            raise ValueError(f"{r['id']}: qty x unit != total")
        items.append({'id': r['id'], 'category': r['id'][0], **CONTENT[r['id']], 'qty': qty,
                      'unit_price': unit, 'total': total, 'spare': SPARES.get(r['id'], 0),
                      'vendor': r['vendor'], 'url': r['url'], 'source_description': r['description']})
    for c in CATEGORIES:
        got = sum(i['total'] for i in items if i['category'] == c['id'])
        if got != c['doc_subtotal']:
            raise ValueError(f"category {c['id']}: CSV sum {got} != document subtotal {c['doc_subtotal']}")
    if sum(i['total'] for i in items) != BUDGET['base']:
        raise ValueError('CSV total != document base total')
    spares = sum(i['spare'] * i['unit_price'] for i in items)
    if spares != BUDGET['spares']:
        raise ValueError(f'spare total {spares} != document {BUDGET["spares"]}')
    doc = {'schema': 'chamgap.parts/v1',
           'generated_at': datetime.now().astimezone().isoformat(timespec='seconds'),
           'status': 'REVIEW_PROPOSAL_NOT_INSPECTED',
           'status_note': '검토용 제안 · 실물 검수 전. 핀 번호는 클래식 ESP32 DevKit 기준이며 보드 수령 후 확정 (설계서 20.2).',
           'price_date': '2026-09-29',
           'price_note': '구매요청서의 2026-09-29 조사값. 오늘 판매가가 아니며 배송비 제외 (설계서 19.1).',
           'source': {'file': SOURCE.as_posix(), 'sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                      'document': 'CHAMGAP_FlyCheck_FULL_SYSTEM_MASTER v4.1 — 2.1, 19, 20, 29장'},
           'categories': CATEGORIES, 'budget': BUDGET, 'items': items, 'pending': PENDING}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'parts.json').write_text(json.dumps(doc, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f"parts.json: {len(items)} items, total {sum(i['total'] for i in items):,} KRW, "
          f"spares {spares:,}, pending {len(PENDING)} -> {a.out}")


if __name__ == '__main__':
    main()
