from __future__ import annotations
import numpy as np
from .contracts import Action, F, action_mask


def fixed_policy(obs: np.ndarray) -> int:
    """Reasonable fixed baseline; cached measurements are never recharged."""
    mask = action_mask(obs)
    if obs[F['resample_count']] < 1 and mask[0]:
        return int(Action.RESAMPLE)
    if obs[F['compare_count']] < 1 and mask[1]:
        return int(Action.COMPARE)
    if mask[2] and obs[F['pulse_count']] < 1:
        return int(Action.PULSE_TEST)
    if mask[1]:
        return int(Action.COMPARE)
    return int(Action.REQUEST_INSPECTION)
