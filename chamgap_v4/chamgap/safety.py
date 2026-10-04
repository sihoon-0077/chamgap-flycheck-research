from __future__ import annotations
from dataclasses import dataclass
import time
import sqlite3
from .contracts import Action, action_mask, F

@dataclass(frozen=True)
class CheckRequest:
    request_id: str
    action: int
    expires_unix_s: float
    requested_ml: float = 0.
    requested_ms: int = 0

class DryRunExecutor:
    """Contract / duplicate-test implementation. Never energizes a real pin.

    Production adds authenticated authority, current device evidence,
    per-session device-enforced budgets, durable ack/result lifecycle and
    hardware watchdog / estop. This class is NOT that complete safety system.
    """
    def __init__(self, ledger_path=':memory:'):
        self.db = sqlite3.connect(ledger_path)
        self.db.execute('CREATE TABLE IF NOT EXISTS requests(id TEXT PRIMARY KEY, status TEXT)')
        self.db.commit()

    def accept(self, request: CheckRequest, observation, now=None):
        now = time.time() if now is None else now
        if not request.request_id:
            return {'status':'REJECTED','reason':'missing_id'}
        if self.db.execute('SELECT 1 FROM requests WHERE id=?',(request.request_id,)).fetchone():
            return {'status':'DUPLICATE','reason':'already_seen_no_reexecution'}
        if now >= request.expires_unix_s:
            return {'status':'REJECTED','reason':'expired'}
        try:
            a = Action(request.action)
        except ValueError:
            return {'status':'REJECTED','reason':'unknown_action'}
        if not action_mask(observation)[a]:
            return {'status':'REJECTED','reason':'observable_interlock'}
        if a == Action.PULSE_TEST:
            if not (0 < request.requested_ml <= 10. and 0 < request.requested_ms <= 60000):
                return {'status':'REJECTED','reason':'dose_or_time_limit'}
            if request.requested_ml > observation[F['remaining_water_scaled']]*20.:
                return {'status':'REJECTED','reason':'session_water_limit'}
        # Atomic durable insert before action acceptance; no actual motor call.
        try:
            self.db.execute('INSERT INTO requests VALUES (?,?)',(request.request_id,'DRY_RUN'))
            self.db.commit()
        except sqlite3.IntegrityError:
            return {'status':'DUPLICATE','reason':'race_no_reexecution'}
        return {'status':'DRY_RUN_ACCEPTED','action':a.name,'motor_enabled':False}
