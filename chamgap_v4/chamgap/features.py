"""Shared real/replay observation builder. Inputs are measured engineering units.
No ground-truth labels are accepted here. This is an input contract, not a
replacement for node calibration, freshness checks or authenticated safety state.
"""
from dataclasses import dataclass
import numpy as np
from .contracts import D,F,check_observation

@dataclass(frozen=True)
class Reading:
    timestamp_s: float
    value: float

@dataclass(frozen=True)
class Evidence:
    channel: str
    target_window: tuple[Reading, ...]
    comparison: Reading | None = None
    command_ml: float = 0.
    delivered_estimate_ml: float = 0.
    delivery_valid: bool = False
    response: float = 0.
    response_valid: bool = False
    ambient_temperature_c: float = 24.5
    ambient_humidity_pct: float = 63.
    elapsed_s: float = 0.
    resample_count: int = 0
    compare_count: int = 0
    pulse_count: int = 0
    remaining_water_ml: float = 20.
    permit_pulse: bool = False
    scale_ok: bool = False
    estop: bool = True
    leak: bool = True
    cooldown_done: bool = False


def build_observation(e: Evidence, now_s: float) -> np.ndarray:
    channel_index={'soil':0,'temperature':1,'humidity':2}
    if e.channel not in channel_index or not e.target_window:
        raise ValueError('channel and a nonempty measured target window required')
    readings=sorted(e.target_window,key=lambda r:r.timestamp_s)
    if readings[-1].timestamp_s>now_s or (e.comparison and e.comparison.timestamp_s>now_s):
        raise ValueError('future measurement must not enter a pre-action observation')
    scale={'soil':1.,'temperature':50.,'humidity':100.}[e.channel]
    values=np.array([r.value/scale for r in readings],dtype=float)
    x=np.zeros(D,dtype=np.float32); x[channel_index[e.channel]]=1.
    x[F['target_scaled']]=values[-1]
    x[F['target_delta']]=values[-1]-values[0]
    x[F['target_std']]=values.std()
    x[F['target_age_scaled']]=(now_s-readings[-1].timestamp_s)/600.
    if e.comparison is not None:
        peer=e.comparison.value/scale
        x[F['comparison_scaled']]=peer; x[F['comparison_valid']]=1.
        x[F['comparison_age_scaled']]=(now_s-e.comparison.timestamp_s)/600.
        x[F['pair_abs_difference']]=abs(values[-1]-peer)
    mapping={
      'last_command_ml_scaled':e.command_ml/20.,
      'delivered_ml_scaled':e.delivered_estimate_ml/20.,
      'delivery_valid':e.delivery_valid,'response_scaled':e.response/scale,
      'response_valid':e.response_valid,'air_temperature_scaled':e.ambient_temperature_c/50.,
      'air_humidity_scaled':e.ambient_humidity_pct/100.,'elapsed_scaled':e.elapsed_s/1200.,
      'resample_count':e.resample_count,'compare_count':e.compare_count,'pulse_count':e.pulse_count,
      'remaining_water_scaled':e.remaining_water_ml/20.,'permit_pulse':e.permit_pulse,
      'scale_ok':e.scale_ok,'estop':e.estop,'leak':e.leak,'cooldown_done':e.cooldown_done}
    for k,v in mapping.items(): x[F[k]]=v
    return check_observation(x)
