from __future__ import annotations
from enum import IntEnum
import numpy as np

class Action(IntEnum):
    RESAMPLE = 0
    COMPARE = 1
    PULSE_TEST = 2
    REQUEST_INSPECTION = 3

# All fields are observable; no fault labels or hidden environmental state.
FEATURES = (
    "channel_soil", "channel_temperature", "channel_humidity",
    "target_scaled", "target_delta", "target_std", "target_age_scaled",
    "comparison_scaled", "comparison_valid", "comparison_age_scaled",
    "pair_abs_difference", "last_command_ml_scaled", "delivered_ml_scaled",
    "delivery_valid", "response_scaled", "response_valid",
    "air_temperature_scaled", "air_humidity_scaled", "elapsed_scaled",
    "resample_count", "compare_count", "pulse_count", "remaining_water_scaled",
    "permit_pulse", "scale_ok", "estop", "leak", "cooldown_done",
)
D = len(FEATURES)
F = {name: i for i, name in enumerate(FEATURES)}


def check_observation(obs: np.ndarray) -> np.ndarray:
    a = np.asarray(obs, dtype=np.float32)
    if a.shape != (D,) or not np.isfinite(a).all():
        raise ValueError(f"observation must be finite shape {(D,)}, got {a.shape}")
    return a


def action_mask(obs: np.ndarray) -> np.ndarray:
    """Permissions use ONLY deployable measurements / operator interlocks."""
    x = check_observation(obs)
    fresh_wet = (x[F['comparison_valid']] > .5 and
                 x[F['comparison_age_scaled']] <= .05 and
                 x[F['comparison_scaled']] >= .82)
    pulse = (x[F['channel_soil']] > .5 and
             x[F['permit_pulse']] > .5 and x[F['scale_ok']] > .5 and
             x[F['estop']] < .5 and x[F['leak']] < .5 and
             x[F['cooldown_done']] > .5 and not fresh_wet and
             x[F['remaining_water_scaled']] >= .5 and
             x[F['pulse_count']] < 2)
    return np.array([x[F['resample_count']] < 2,
                     x[F['compare_count']] < 2,
                     pulse, True], dtype=bool)
