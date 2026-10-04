"""Reference wire contracts. No claim of hardware or identity validation.
The receiver must separately authenticate the device and bind its topic.
"""
from __future__ import annotations
import re
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ID = r'^[A-Za-z0-9_-]{1,64}$'

class StrictMessage(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)

class Readings(StrictMessage):
    temperature_a_c: float | None = None
    temperature_b_c: float | None = None
    humidity_a_rh_pct: float | None = None
    humidity_b_rh_pct: float | None = None
    soil_a_adc_raw: int | None = Field(default=None, ge=0, le=4095)
    soil_b_adc_raw: int | None = Field(default=None, ge=0, le=4095)
    soil_a_index: float | None = None
    soil_b_index: float | None = None
    pot_mass_raw: int | None = None
    pot_mass_g: float | None = None

class Quality(StrictMessage):
    air_a_valid: bool = False
    air_b_valid: bool = False
    soil_a_valid: bool = False
    soil_b_valid: bool = False
    mass_valid: bool = False
    scale_stable: bool = False

class Interlocks(StrictMessage):
    # null = unknown, NEVER silently interpreted as safe.
    estop_asserted: bool | None = None
    leak_detected: bool | None = None
    motor_power_enabled: bool | None = None
    manual_armed: bool = False

class Telemetry(StrictMessage):
    schema_version: Literal['telemetry.v4'] = 'telemetry.v4'
    mode: Literal['simulation','replay','shadow','live_supervised']
    zone_id: str = Field(pattern=r'^Z[1-3]$')
    node_id: str = Field(pattern=r'^N[1-3]$')
    boot_id: str = Field(pattern=ID)
    seq: int = Field(ge=0, le=2**32-1, strict=True)
    measured_at: datetime
    uptime_ms: int = Field(ge=0, le=2**64-1, strict=True)
    clock_synced: bool
    fw_version: str = Field(min_length=1,max_length=64)
    cal_version: str = Field(min_length=1,max_length=64)
    readings: Readings
    quality: Quality
    interlocks: Interlocks

    @field_validator('measured_at')
    @classmethod
    def aware_time(cls, value):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('measured_at must contain timezone')
        return value

    @model_validator(mode='after')
    def valid_flags_need_data(self):
        q,r=self.quality,self.readings
        pairs=[(q.air_a_valid,[r.temperature_a_c,r.humidity_a_rh_pct]),
               (q.air_b_valid,[r.temperature_b_c,r.humidity_b_rh_pct]),
               (q.soil_a_valid,[r.soil_a_adc_raw,r.soil_a_index]),
               (q.soil_b_valid,[r.soil_b_adc_raw,r.soil_b_index]),
               (q.mass_valid,[r.pot_mass_g])]
        if any(flag and any(x is None for x in values) for flag,values in pairs):
            raise ValueError('valid flag requires corresponding readings')
        if q.scale_stable and not q.mass_valid:
            raise ValueError('stable scale requires valid mass')
        return self

    def require_topic(self, topic:str) -> None:
        expected=f'chamgap/v4/{self.zone_id}/{self.node_id}/telemetry'
        if topic!=expected:
            raise ValueError('payload identity does not match topic')

class CheckRequest(StrictMessage):
    request_id: str = Field(pattern=ID)
    session_id: str = Field(pattern=ID)
    zone_id: str = Field(pattern=r'^Z[1-3]$')
    node_id: str = Field(pattern=r'^N[1-3]$')
    target_boot_id: str = Field(pattern=ID)
    action: Literal['RESAMPLE','COMPARE','PULSE_TEST','REQUEST_INSPECTION']
    config_version: str = Field(min_length=1,max_length=64)
    issued_at: datetime
    expires_at: datetime
    dose_ul: int = Field(default=0,ge=0,strict=True)
    duration_ms: int = Field(default=0,ge=0,strict=True)

    @field_validator('issued_at','expires_at')
    @classmethod
    def aware_time(cls,value):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('command times must include timezone')
        return value

    @model_validator(mode='after')
    def bounded_request(self):
        if not 0 < (self.expires_at-self.issued_at).total_seconds() <= 30:
            raise ValueError('validity must be >0 and <=30 seconds')
        if self.action=='PULSE_TEST':
            if not (0 < self.dose_ul <= 10000 and 0 < self.duration_ms <= 60000):
                raise ValueError('unapproved pulse volume/duration')
        elif self.dose_ul or self.duration_ms:
            raise ValueError('non-pulse actions must have zero motor parameters')
        return self

class TrustedState(StrictMessage):
    """Internal executor input, NOT a request body accepted from the browser."""
    zone_id: str
    node_id: str
    boot_id: str
    config_version: str
    received_at: datetime
    clock_synced: bool = False
    connected: bool = False
    scale_ok: bool = False
    estop_clear: bool = False
    leak_clear: bool = False
    manual_armed: bool = False
    cooldown_done: bool = False
    fresh_wet_evidence: bool = True
    channel: Literal['soil','temperature','humidity']
