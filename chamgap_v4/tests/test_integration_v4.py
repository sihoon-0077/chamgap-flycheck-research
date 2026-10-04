from datetime import datetime,timedelta,timezone
import sqlite3
import pytest
from pydantic import ValidationError
from integration.contracts import Telemetry,CheckRequest,TrustedState
from integration.store import Ledger,Rejected

NOW=datetime(2026,10,1,tzinfo=timezone.utc)

def request(**kw):
    data=dict(request_id='r1',session_id='s1',zone_id='Z1',node_id='N1',target_boot_id='b1',
              config_version='SAFE-1',action='PULSE_TEST',issued_at=NOW,
              expires_at=NOW+timedelta(seconds=10),dose_ul=10000,duration_ms=10000)
    data.update(kw);return CheckRequest(**data)
def state(**kw):
    data=dict(zone_id='Z1',node_id='N1',boot_id='b1',config_version='SAFE-1',received_at=NOW,
              clock_synced=True,connected=True,scale_ok=True,estop_clear=True,leak_clear=True,
              manual_armed=True,cooldown_done=True,fresh_wet_evidence=False,channel='soil')
    data.update(kw);return TrustedState(**data)
def telemetry(**kw):
    data=dict(mode='simulation',zone_id='Z1',node_id='N1',boot_id='b1',seq=1,measured_at=NOW,
              uptime_ms=1000,clock_synced=True,fw_version='demo',cal_version='none',
              readings={},quality={},interlocks={})
    data.update(kw);return Telemetry(**data)
@pytest.fixture
def ledger(tmp_path):
    l=Ledger(str(tmp_path/'test.sqlite'));l.register_session('s1','Z1','N1','b1')
    yield l
    l.close()

def test_durable_dry_run_and_duplicate(ledger):
    assert ledger.reserve_dry_run(request(),state(),NOW)['motor_enabled'] is False
    assert ledger.reserve_dry_run(request(),state(),NOW)['status']=='DUPLICATE'
    assert ledger.db.execute('SELECT reserved_ul,pulses FROM sessions').fetchone()==(10000,1)
def test_conflicting_duplicate(ledger):
    ledger.reserve_dry_run(request(),state(),NOW)
    with pytest.raises(Rejected,match='idempotency'):ledger.reserve_dry_run(request(dose_ul=9000),state(),NOW)
def test_budget_two_pulses(ledger):
    ledger.reserve_dry_run(request(),state(),NOW)
    ledger.reserve_dry_run(request(request_id='r2'),state(),NOW)
    with pytest.raises(Rejected,match='budget'):ledger.reserve_dry_run(request(request_id='r3'),state(),NOW)
def test_reopen_keeps_budget(tmp_path):
    p=str(tmp_path/'persistent.sqlite'); l=Ledger(p);l.register_session('s1','Z1','N1','b1')
    l.reserve_dry_run(request(),state(),NOW);l.close();l=Ledger(p)
    assert l.db.execute('SELECT reserved_ul FROM sessions').fetchone()[0]==10000;l.close()
@pytest.mark.parametrize('kw',[{'scale_ok':False},{'estop_clear':False},{'leak_clear':False},
                             {'manual_armed':False},{'cooldown_done':False},
                             {'fresh_wet_evidence':True},{'channel':'temperature'}, {'clock_synced':False}])
def test_interlocks_fail_closed(ledger,kw):
    with pytest.raises(Rejected):ledger.reserve_dry_run(request(),state(**kw),NOW)
    assert ledger.db.execute('SELECT reserved_ul FROM sessions').fetchone()[0]==0

def test_stale(ledger):
    with pytest.raises(Rejected,match='stale'):ledger.reserve_dry_run(request(),state(received_at=NOW-timedelta(seconds=6)),NOW)
def test_boot(ledger):
    with pytest.raises(Rejected,match='boot'):ledger.reserve_dry_run(request(),state(boot_id='b2'),NOW)
def test_expiry(ledger):
    with pytest.raises(Rejected):ledger.reserve_dry_run(request(),state(),NOW+timedelta(seconds=10))
def test_existing_session_cannot_reset(ledger):
    with pytest.raises(sqlite3.IntegrityError):ledger.register_session('s1','Z1','N1','b1')
def test_pulse_requires_bounded_int():
    with pytest.raises(ValidationError):request(dose_ul=10001)
    with pytest.raises(ValidationError):request(dose_ul=3.2)
def test_nonpulse_has_no_motor_parameters():
    with pytest.raises(ValidationError):request(action='COMPARE')
def test_unknown_truth_field():
    with pytest.raises(ValidationError):telemetry(fault_type='stuck')
def test_flag_requires_reading():
    with pytest.raises(ValidationError):telemetry(quality={'soil_a_valid':True})
def test_nan_denied():
    with pytest.raises(ValidationError):telemetry(readings={'pot_mass_g':float('nan')})
def test_telemetry_dedupe(ledger):
    t=telemetry();topic='chamgap/v4/Z1/N1/telemetry'
    assert ledger.ingest(t,topic,NOW)=='STORED'
    assert ledger.ingest(t,topic,NOW)=='DUPLICATE'
def test_topic_mismatch(ledger):
    with pytest.raises(ValueError):ledger.ingest(telemetry(),'chamgap/v4/Z2/N2/telemetry',NOW)
def test_changed_payload_same_sequence(ledger):
    topic='chamgap/v4/Z1/N1/telemetry';ledger.ingest(telemetry(),topic,NOW)
    with pytest.raises(Rejected):ledger.ingest(telemetry(uptime_ms=2000),topic,NOW)
def test_future_telemetry(ledger):
    with pytest.raises(Rejected):ledger.ingest(telemetry(measured_at=NOW+timedelta(seconds=6)),'chamgap/v4/Z1/N1/telemetry',NOW)
def test_aware_timestamp():
    with pytest.raises(ValidationError):telemetry(measured_at=NOW.replace(tzinfo=None))
def test_reboot_new_sequence(ledger):
    topic='chamgap/v4/Z1/N1/telemetry'
    assert ledger.ingest(telemetry(),topic,NOW)=='STORED'
    assert ledger.ingest(telemetry(boot_id='b2'),topic,NOW)=='STORED'
