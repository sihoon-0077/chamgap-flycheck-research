from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .contracts import FEATURES, action_mask
from .simulator import SmartFarmCore
from .validator import diagnose
from .policies import fixed_policy


def rollout_value(env: SmartFarmCore, first_action: int) -> float:
    """Score chosen first check, then frozen baseline until end.

    This is a supervised policy-improvement target, not optimal Q-learning.
    Simulator reward may use truth; observation/model never does.
    """
    obs, reward, term, trunc, _ = env.step(first_action)
    total = reward
    while not (term or trunc):
        obs, reward, term, trunc, _ = env.step(fixed_policy(obs))
        total += reward
    return float(total)


def collect(output: str, episodes: int = 500, replicas: int = 3, seed: int = 17):
    if episodes < 10 or replicas < 1:
        raise ValueError('use episodes >= 10, replicas >= 1')
    rng = np.random.default_rng(seed)
    xs, ys, target_masks, groups, masks = [], [], [], [], []
    for group in range(episodes):
        env = SmartFarmCore()
        obs, _ = env.reset(seed=int(rng.integers(1, 2**31-1)))
        # Prefixes cover later decision points; ALL descendants share group.
        for depth in range(env.cfg.max_steps):
            if diagnose(obs).label != 'UNKNOWN':
                break
            allowed = action_mask(obs)
            target = np.zeros(4, dtype=np.float32)
            for a in np.flatnonzero(allowed):
                values = []
                for r in range(replicas):
                    # Same cloned physical state, different future noise.
                    # Common seed is NOT perfect common-random-number coupling:
                    # actions consume random numbers differently.
                    future_seed = int(seed + group*1009 + depth*101 + r)
                    values.append(rollout_value(env.clone(future_seed), int(a)))
                target[a] = np.mean(values)
            xs.append(obs.copy()); ys.append(target)
            target_masks.append(allowed.copy()); masks.append(allowed.copy())
            groups.append(group)
            # Exploration only in simulation; no physical pump is connected.
            a = int(rng.choice(np.flatnonzero(allowed)))
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
    if not xs:
        raise RuntimeError('no unresolved states collected')
    path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=np.asarray(xs, dtype=np.float32),
                        y=np.asarray(ys, dtype=np.float32),
                        target_mask=np.asarray(target_masks, dtype=bool),
                        allowed_mask=np.asarray(masks, dtype=bool),
                        group=np.asarray(groups, dtype=np.int64))
    meta = {'domain': 'SIM_ONLY_UNCALIBRATED', 'sim_version': env.cfg.version,
            'features': list(FEATURES), 'episodes_requested': episodes,
            'groups_saved': len(set(groups)), 'rows': len(xs), 'replicas': replicas,
            'seed': seed, 'target': 'return_after_first_action_then_fixed_continuation',
            'truth_in_features': False}
    path.with_suffix('.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(json.dumps(meta, indent=2))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='data/sim_train.npz')
    p.add_argument('--episodes', type=int, default=500)
    p.add_argument('--replicas', type=int, default=3)
    p.add_argument('--seed', type=int, default=17)
    a = p.parse_args(); collect(a.out, a.episodes, a.replicas, a.seed)
