from __future__ import annotations
import copy
from dataclasses import dataclass
import numpy as np
from .contracts import Action, D, F, action_mask
from .validator import diagnose

@dataclass
class SimConfig:
    version: str = 'toy-physical-v0.1-unfitted'
    max_steps: int = 5
    pulse_ml: float = 10.0
    max_water_ml: float = 20.0
    resample_s: float = 20.0
    compare_s: float = 10.0
    pulse_observe_s: float = 180.0
    # Planning defaults ONLY; fit to development measurements before real use.
    correct_reward: float = 5.0
    wrong_reward: float = -10.0
    inspection_cost: float = 1.2
    timeout_cost: float = 1.5
    time_cost_per_s: float = .003
    water_cost_per_ml: float = .025


class SmartFarmCore:
    """A small numerical test environment, not a validated digital twin.

    API matches reset/step tuples. Gymnasium adapter is in gym_adapter.py.
    The scoring oracle is private to the simulator/evaluator, not the policy.
    Each decision episode concerns one zone and one suspect sensor channel.
    Three physical zones are orchestrated outside this core.
    """
    SCENARIOS = ('normal', 'dry', 'stuck', 'bias', 'supply', 'ambiguous')

    def __init__(self, config: SimConfig | None = None):
        self.cfg = config or SimConfig()
        if self.cfg.pulse_ml != 10. or self.cfg.max_water_ml != 20.:
            raise ValueError('v0.1 feature/mask contract uses 10 mL pulse, 20 mL budget')
        self.rng = np.random.default_rng(0)
        self.finished = True

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.rng = np.random.default_rng(seed)
        op = options or {}
        self.channel = int(op.get('channel', self.rng.integers(3)))
        scenario = op.get('scenario', self.rng.choice(self.SCENARIOS))
        if self.channel != 0 and scenario in ('dry', 'supply'):
            scenario = 'normal'
        self.scenario = scenario
        self.true_water = float(self.rng.uniform(500, 1000))  # mL in soil volume
        self.soil_volume = float(self.rng.uniform(1800, 3500))  # mL bulk medium
        self.theta_sat = .65
        self.theta_fc = .36
        if self.channel == 0 and scenario == 'normal':
            self.true_water = float(self.rng.uniform(.30, .44)) * self.soil_volume
        if scenario in ('dry', 'supply'):
            self.true_water = float(self.rng.uniform(.12, .24)) * self.soil_volume
        self.local_theta = self.true_water / self.soil_volume
        self.temp_c = float(self.rng.uniform(20, 29))
        self.rh_pct = float(self.rng.uniform(45, 80))
        self.tau_s = float(self.rng.uniform(40, 420))
        self.evap_ml_s = float(self.rng.uniform(.0002, .003))
        self.noise = float(self.rng.uniform(.001, .02))
        self.efficiency = float(self.rng.uniform(.90, 1.04))
        self.blockage = float(self.rng.uniform(.78, 1.0)) if scenario == 'supply' else 0.
        self.target_bias = float(self.rng.uniform(-.26, -.13)) if scenario == 'bias' else 0.
        self.peer_bias = float(self.rng.normal(0, .008))
        if scenario == 'ambiguous':
            # A sensor pair may be inconsistent for an unidentifiable reason.
            self.peer_bias = float(self.rng.choice([-1, 1]) * self.rng.uniform(.10, .22))
            self.tau_s = float(self.rng.uniform(600, 1200))
        self.hold = float(self.rng.uniform(.22, .40))
        self.counts = np.zeros(4, dtype=int)
        self.elapsed_s = 0.
        self.used_ml = 0.
        self.last_command = 0.
        self.last_delivery = 0.
        self.last_response = 0.
        self.delivery_valid = False
        self.response_valid = False
        self.target_history = [self._target_sample() for _ in range(6)]
        self.target = self.target_history[-1]
        # A comparison reading may already be cached: never hide available data.
        self.peer_valid = bool(self.rng.random() < .5)
        self.peer = self._peer_sample() if self.peer_valid else 0.
        self.peer_age = float(self.rng.uniform(0, 900)) if self.peer_valid else 0.
        self.target_age = 0.
        self.permit = bool(op.get('permit_pulse', True))
        self.scale_ok = bool(op.get('scale_ok', True))
        self.estop = bool(op.get('estop', False))
        self.leak = bool(op.get('leak', False))
        self.finished = False
        self.step_count = 0
        return self._obs(), {'sim_version': self.cfg.version}

    def _physical_signal(self):
        if self.channel == 0:
            return self.local_theta / self.theta_sat  # simulated relative index
        return self.temp_c / 50. if self.channel == 1 else self.rh_pct / 100.

    def _target_sample(self):
        if self.scenario == 'stuck':
            return self.hold
        return float(np.clip(self._physical_signal() + self.target_bias +
                             self.rng.normal(0, self.noise), -.10, 1.10))

    def _peer_sample(self):
        return float(np.clip(self._physical_signal() + self.peer_bias +
                             self.rng.normal(0, self.noise), -.10, 1.10))

    def _advance(self, seconds: float, water_ml: float = 0.):
        self.true_water += water_ml
        # Mass balance; local probe response is a separate lag state.
        chunks = max(1, int(np.ceil(seconds / 10.)))
        dt = seconds / chunks
        for _ in range(chunks):
            excess = max(0., self.true_water - self.theta_fc * self.soil_volume)
            drainage = excess * (1. - np.exp(-dt / 1500.))
            self.true_water = np.clip(self.true_water - self.evap_ml_s * dt - drainage,
                                      0., self.theta_sat * self.soil_volume)
            bulk = self.true_water / self.soil_volume
            self.local_theta += (1. - np.exp(-dt / self.tau_s)) * (bulk - self.local_theta)
            # Correlated gentle ambient evolution; not greenhouse CFD.
            self.temp_c += float(self.rng.normal(0, .003 * np.sqrt(dt)))
            self.rh_pct += float(self.rng.normal(0, .01 * np.sqrt(dt)))
        self.elapsed_s += seconds
        self.target_age += seconds
        if self.peer_valid:
            self.peer_age += seconds

    def _new_target(self, count: int = 1):
        for _ in range(count):
            self.target_history.append(self._target_sample())
        self.target_history = self.target_history[-10:]
        self.target = self.target_history[-1]
        self.target_age = 0.

    def _obs(self):
        x = np.zeros(D, dtype=np.float32)
        x[self.channel] = 1.
        v = {
            'target_scaled': self.target,
            'target_delta': self.target_history[-1] - self.target_history[0],
            'target_std': float(np.std(self.target_history)),
            'target_age_scaled': self.target_age / 600.,
            'comparison_scaled': self.peer if self.peer_valid else 0.,
            'comparison_valid': float(self.peer_valid),
            'comparison_age_scaled': self.peer_age / 600. if self.peer_valid else 0.,
            'pair_abs_difference': abs(self.target - self.peer) if self.peer_valid else 0.,
            'last_command_ml_scaled': self.last_command / 20.,
            'delivered_ml_scaled': self.last_delivery / 20.,
            'delivery_valid': float(self.delivery_valid),
            'response_scaled': self.last_response,
            'response_valid': float(self.response_valid),
            # Starter uses fixed nominal room context, not hidden true readings.
            # Replace with logged auxiliary sensor readings in the calibrated version.
            'air_temperature_scaled': self.context_temp,
            'air_humidity_scaled': self.context_rh,
            'elapsed_scaled': self.elapsed_s / 1200.,
            'resample_count': float(self.counts[0]),
            'compare_count': float(self.counts[1]),
            'pulse_count': float(self.counts[2]),
            'remaining_water_scaled': (self.cfg.max_water_ml-self.used_ml)/20.,
            'permit_pulse': float(self.permit), 'scale_ok': float(self.scale_ok),
            'estop': float(self.estop), 'leak': float(self.leak), 'cooldown_done': 1.,
        }
        for k, val in v.items():
            x[F[k]] = val
        return x

    @property
    def context_temp(self):
        # Initial nominal room readings, not target ground-truth temperature.
        return .49

    @property
    def context_rh(self):
        return .63

    def clone(self, future_seed: int | None = None):
        other = copy.deepcopy(self)
        if future_seed is not None:
            other.rng = np.random.default_rng(future_seed)
        return other

    def ground_truth_for_evaluation(self):
        return {'normal': 'NORMAL', 'dry': 'DRY', 'stuck': 'SENSOR',
                'bias': 'SENSOR', 'supply': 'SUPPLY', 'ambiguous': 'UNKNOWN'}[self.scenario]

    def step(self, action: int):
        if self.finished:
            raise RuntimeError('episode finished; call reset')
        a = Action(action)
        if not action_mask(self._obs())[a]:
            raise ValueError(f'action {a.name} rejected by observable-state mask')
        before_t = self.elapsed_s
        before_w = self.used_ml
        self.counts[a] += 1
        self.step_count += 1
        diagnosis = 'UNKNOWN'
        reason = 'inspection_requested'
        terminated = a == Action.REQUEST_INSPECTION
        if a == Action.RESAMPLE:
            self._advance(self.cfg.resample_s)
            self._new_target(5)
        elif a == Action.COMPARE:
            self._advance(self.cfg.compare_s)
            self._new_target()
            self.peer = self._peer_sample()
            self.peer_valid = True
            self.peer_age = 0.
        elif a == Action.PULSE_TEST:
            # 10 mL is a simulator command, NOT a universal plant-safe dose.
            cmd = self.cfg.pulse_ml
            actual = max(0., cmd*self.efficiency*(1.-self.blockage))
            old_target = self.target
            self.last_command = cmd
            self._advance(self.cfg.pulse_observe_s, actual)
            self._new_target()
            self.last_delivery = float(actual + self.rng.normal(0, .35))
            self.delivery_valid = True
            self.last_response = self.target - old_target
            self.response_valid = True
            self.used_ml += cmd  # cap requested volume conservatively
        x = self._obs()
        if not terminated:
            d = diagnose(x)
            diagnosis, reason = d.label, d.reason
            terminated = diagnosis != 'UNKNOWN'
        dt = self.elapsed_s-before_t
        dw = self.used_ml-before_w
        reward = -self.cfg.time_cost_per_s*dt - self.cfg.water_cost_per_ml*dw
        if a == Action.REQUEST_INSPECTION:
            reward -= self.cfg.inspection_cost  # NOT an automatic correct diagnosis
        elif terminated:
            reward += (self.cfg.correct_reward if diagnosis == self.ground_truth_for_evaluation()
                       else self.cfg.wrong_reward)
        truncated = (not terminated and self.step_count >= self.cfg.max_steps)
        if truncated:
            reward -= self.cfg.timeout_cost
        self.finished = terminated or truncated
        info = {'diagnosis': diagnosis, 'reason': reason, 'elapsed_s': dt,
                'command_ml': dw, 'measured_delivery_ml': self.last_delivery,
                'human_request': a == Action.REQUEST_INSPECTION,
                'sim_version': self.cfg.version}
        return x, float(reward), bool(terminated), bool(truncated), info
