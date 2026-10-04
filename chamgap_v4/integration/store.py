"""Durable dry-run ledger. This module has NO MQTT publication or GPIO calls.
A production live executor needs authenticated state, outbox delivery, node-side
limits, persistent device accounting, certified wiring and real acceptance tests.
"""
from __future__ import annotations
import hashlib,json,sqlite3
from datetime import datetime,timezone
from .contracts import CheckRequest,TrustedState,Telemetry

class Rejected(ValueError): pass

def canonical_hash(value:dict)->str:
    raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()

class Ledger:
    def __init__(self,path:str):
        self.db=sqlite3.connect(path,timeout=10,isolation_level=None)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS telemetry(
          node TEXT, boot TEXT, seq INTEGER, mode TEXT, measured TEXT,
          received TEXT, payload_hash TEXT, body TEXT,
          PRIMARY KEY(node,boot,seq,mode));
        CREATE TABLE IF NOT EXISTS sessions(
          session TEXT PRIMARY KEY, zone TEXT, node TEXT, boot TEXT,
          maximum_ul INTEGER CHECK(maximum_ul>0), reserved_ul INTEGER DEFAULT 0,
          pulses INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS requests(
          request_id TEXT PRIMARY KEY, session TEXT, payload_hash TEXT,
          status TEXT, body TEXT, created TEXT);
        ''')

    def close(self): self.db.close()

    def ingest(self, message:Telemetry, topic:str, received_at:datetime)->str:
        message.require_topic(topic)
        if received_at.tzinfo is None: raise Rejected('receiver clock must be aware')
        # Raw out-of-order packets can be retained, but future event times are rejected here.
        if (message.measured_at-received_at).total_seconds()>5:
            raise Rejected('future_measurement')
        body=message.model_dump(mode='json'); h=canonical_hash(body)
        key=(message.node_id,message.boot_id,message.seq,message.mode)
        self.db.execute('BEGIN IMMEDIATE')
        try:
            old=self.db.execute('SELECT payload_hash FROM telemetry WHERE node=? AND boot=? AND seq=? AND mode=?',key).fetchone()
            if old:
                if old[0]!=h: raise Rejected('same_sequence_different_payload')
                self.db.execute('COMMIT'); return 'DUPLICATE'
            self.db.execute('INSERT INTO telemetry VALUES (?,?,?,?,?,?,?,?)',
                            (*key,message.measured_at.isoformat(),received_at.isoformat(),h,json.dumps(body)))
            self.db.execute('COMMIT'); return 'STORED'
        except Exception:
            self.db.execute('ROLLBACK'); raise

    def register_session(self,session:str,zone:str,node:str,boot:str,maximum_ul:int=20000):
        if not 0 < maximum_ul <= 20000: raise Rejected('unapproved_session_budget')
        # Administrative setup only. Same ID NEVER resets the budget.
        self.db.execute('INSERT INTO sessions(session,zone,node,boot,maximum_ul) VALUES (?,?,?,?,?)',
                        (session,zone,node,boot,maximum_ul))

    def reserve_dry_run(self,req:CheckRequest,state:TrustedState,now:datetime)->dict:
        if now.tzinfo is None or state.received_at.tzinfo is None:
            raise Rejected('aware clocks required')
        body=req.model_dump(mode='json'); h=canonical_hash(body)
        self.db.execute('BEGIN IMMEDIATE')
        try:
            old=self.db.execute('SELECT payload_hash,status FROM requests WHERE request_id=?',(req.request_id,)).fetchone()
            if old:
                if old[0]!=h: raise Rejected('idempotency_key_conflict')
                self.db.execute('COMMIT')
                return {'status':'DUPLICATE','previous_status':old[1],'motor_enabled':False}
            if not req.issued_at <= now < req.expires_at: raise Rejected('expired_or_future_request')
            if (req.zone_id,req.node_id,req.target_boot_id,req.config_version)!=(state.zone_id,state.node_id,state.boot_id,state.config_version):
                raise Rejected('identity_boot_or_config_mismatch')
            age=(now-state.received_at).total_seconds()
            if not state.connected or not 0 <= age <= 5: raise Rejected('stale_device')
            row=self.db.execute('SELECT zone,node,boot,maximum_ul,reserved_ul,pulses FROM sessions WHERE session=?',(req.session_id,)).fetchone()
            if not row or row[:3]!=(req.zone_id,req.node_id,req.target_boot_id): raise Rejected('session_mismatch')
            if req.action=='PULSE_TEST':
                conditions=(state.clock_synced,state.scale_ok,state.estop_clear,state.leak_clear,
                            state.manual_armed,state.cooldown_done,not state.fresh_wet_evidence,state.channel=='soil')
                if not all(conditions): raise Rejected('interlock_denied')
                if row[4]+req.dose_ul>row[3] or row[5]>=2: raise Rejected('session_budget')
                self.db.execute('UPDATE sessions SET reserved_ul=reserved_ul+?,pulses=pulses+1 WHERE session=?',(req.dose_ul,req.session_id))
            self.db.execute('INSERT INTO requests VALUES (?,?,?,?,?,?)',
                            (req.request_id,req.session_id,h,'DRY_RUN_RESERVED',json.dumps(body),now.isoformat()))
            self.db.execute('COMMIT')
            return {'status':'DRY_RUN_RESERVED','action':req.action,'motor_enabled':False}
        except Exception:
            self.db.execute('ROLLBACK'); raise
