"""Runtime configuration v2 for the connectome_v1 experiment. Appendix files stay unmodified.

apply() must run in every process that collects, trains or evaluates (also spawned workers):
  reward v2  SimConfig(time_cost_per_s=.001, water_cost_per_ml=.08). Correct +5, wrong -10,
             inspection 1.2 and timeout 1.5 are unchanged (team decision 2026-10-04).
  rule A     REQUEST_INSPECTION is masked until at least one check (resample, compare or
             pulse) has been done. Exceptions: estop or leak active, or no check is possible.

The masked function is rebound in every module that imported action_mask by name, because
`from .contracts import action_mask` copies the reference at import time.
"""
from __future__ import annotations
import importlib
from dataclasses import asdict

import numpy as np

from chamgap import contracts
from chamgap.contracts import F
from chamgap.simulator import SimConfig, SmartFarmCore

REWARD_V2 = {'time_cost_per_s': .001, 'water_cost_per_ml': .08}
MASK_MODULES = ('chamgap.contracts', 'chamgap.simulator', 'chamgap.collect',
                'chamgap.evaluate', 'chamgap.policies')
RULE_A = ('REQUEST_INSPECTION masked while resample_count + compare_count + pulse_count == 0, '
          'unless estop or leak is active or no check action is allowed')
ORIGINAL_MASK = contracts.action_mask
ORIGINAL_INIT = SmartFarmCore.__init__
_applied = False


def config_v2() -> SimConfig:
    return SimConfig(**REWARD_V2)


def action_mask_v2(obs):
    m = ORIGINAL_MASK(obs)                       # validates shape and finiteness
    x = np.asarray(obs, dtype=np.float32)
    untouched = x[F['resample_count']] + x[F['compare_count']] + x[F['pulse_count']] == 0
    emergency = x[F['estop']] > .5 or x[F['leak']] > .5
    if untouched and not emergency and m[:3].any():
        m[3] = False
    return m


def _init_v2(self, config=None):
    ORIGINAL_INIT(self, config if config is not None else config_v2())


def describe() -> dict:
    return {'reward': asdict(config_v2()), 'rule_a': RULE_A, 'patched_modules': list(MASK_MODULES)}


def apply() -> dict:
    """Idempotent; returns what was applied, for the manifest."""
    global _applied
    if not _applied:
        SmartFarmCore.__init__ = _init_v2
        for name in MASK_MODULES:
            module = importlib.import_module(name)
            if getattr(module, 'action_mask', None) not in (ORIGINAL_MASK, action_mask_v2):
                raise RuntimeError(f'{name}.action_mask is neither the appendix nor the v2 mask')
            module.action_mask = action_mask_v2
        _applied = True
    if SmartFarmCore().cfg.water_cost_per_ml != REWARD_V2['water_cost_per_ml']:
        raise RuntimeError('reward v2 not active')
    return describe()
