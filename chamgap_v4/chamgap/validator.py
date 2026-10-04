from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .contracts import F, check_observation

@dataclass(frozen=True)
class Diagnosis:
    label: str
    reason: str


def diagnose(obs: np.ndarray) -> Diagnosis:
    """Deliberately simple research baseline; same function for every policy.

    Does not read hidden state. Thresholds are SIMULATOR defaults, not a
    calibrated or certified agronomic diagnostic algorithm.
    """
    x = check_observation(obs)
    target = x[F['target_scaled']]
    if not (-.1 <= target <= 1.2):
        return Diagnosis('SENSOR', 'measurement_outside_configured_range')
    if (x[F['delivery_valid']] > .5 and
        x[F['last_command_ml_scaled']] > .05):
        ratio = x[F['delivered_ml_scaled']] / x[F['last_command_ml_scaled']]
        if ratio < .25:
            # Reports a delivery-path problem, not an exact blockage location.
            return Diagnosis('SUPPLY', 'command_without_expected_mass_increase')
    fresh = (x[F['comparison_valid']] > .5 and
             x[F['comparison_age_scaled']] <= .05 and
             x[F['target_age_scaled']] <= .05)
    if fresh:
        diff = x[F['pair_abs_difference']]
        # A pair difference alone cannot identify which sensor is faulty.
        # SIM baseline requires a second kind of evidence, or repeat history.
        if (diff > .10 and x[F['response_valid']] > .5 and
            x[F['delivery_valid']] > .5 and
            x[F['delivered_ml_scaled']] > .05 and
            abs(x[F['response_scaled']]) < .015):
            return Diagnosis('SENSOR', 'comparison_and_no_response_after_delivery')
        if (diff > .16 and x[F['compare_count']] >= 2 and
            x[F['resample_count']] >= 1):
            # This is a measurement disagreement, not absolute truth recovery.
            return Diagnosis('SENSOR', 'persistent_measurement_disagreement')
        if diff < .045:
            if x[F['channel_soil']] > .5 and target < .42:
                # If no recent delivery evidence, supply health is still unknown.
                if (x[F['delivery_valid']] > .5 and
                    x[F['delivered_ml_scaled']] > .05):
                    return Diagnosis('DRY', 'low_concordant_values_with_delivery')
                return Diagnosis('UNKNOWN', 'dry_readings_supply_not_checked')
            return Diagnosis('NORMAL', 'fresh_comparison_agrees')
    return Diagnosis('UNKNOWN', 'insufficient_or_ambiguous_evidence')
