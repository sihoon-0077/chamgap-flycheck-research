"""Run with python -m integration.demo. Synthetic input, no device connection."""
from datetime import datetime,timedelta,timezone
from pathlib import Path
import json,tempfile
from .contracts import CheckRequest,TrustedState
from .store import Ledger

def main():
    now=datetime.now(timezone.utc)
    state=TrustedState(zone_id='Z1',node_id='N1',boot_id='boot-demo',config_version='SAFE-DEMO',
                       received_at=now,clock_synced=True,connected=True,scale_ok=True,
                       estop_clear=True,leak_clear=True,manual_armed=True,cooldown_done=True,
                       fresh_wet_evidence=False,channel='soil')
    req=CheckRequest(request_id='request-demo',session_id='session-demo',zone_id='Z1',node_id='N1',
                     target_boot_id='boot-demo',config_version='SAFE-DEMO',action='PULSE_TEST',
                     issued_at=now,expires_at=now+timedelta(seconds=10),dose_ul=10000,duration_ms=8000)
    with tempfile.TemporaryDirectory() as d:
        ledger=Ledger(str(Path(d)/'ledger.sqlite'))
        ledger.register_session('session-demo','Z1','N1','boot-demo')
        print(json.dumps({'domain':'SYNTHETIC_DRY_RUN_ONLY','first':ledger.reserve_dry_run(req,state,now),
                          'second':ledger.reserve_dry_run(req,state,now)},indent=2))
        ledger.close()
if __name__=='__main__': main()
