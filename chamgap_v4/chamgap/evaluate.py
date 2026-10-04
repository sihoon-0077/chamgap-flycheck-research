from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .contracts import action_mask
from .policies import fixed_policy
from .simulator import SmartFarmCore
from .validator import diagnose
from .train import load_model
from .model import choose_action


def evaluate(checkpoint=None, episodes=120, seed=900001, output='runs/eval.json'):
    torch.set_num_threads(1)
    policy = load_model(checkpoint) if checkpoint else None
    rows = []
    for i in range(episodes):
        env = SmartFarmCore()
        obs, _ = env.reset(seed=seed+i)
        expected = env.ground_truth_for_evaluation()  # evaluator only
        d = diagnose(obs)
        label = d.label
        total = 0.; actions = []; requested = False; timeout = False
        initial_resolved = label != 'UNKNOWN'
        while label == 'UNKNOWN' and not env.finished:
            mask = action_mask(obs)
            a = (choose_action(policy, obs, mask)[0] if policy else fixed_policy(obs))
            obs, reward, term, trunc, info = env.step(a)
            total += reward; actions.append(int(a)); label = info['diagnosis']
            requested |= info['human_request']; timeout |= trunc
            if term or trunc:
                break
        # UNKNOWN/human is abstention, NEVER counted as automatic success.
        auto = label != 'UNKNOWN'
        rows.append({'episode':i,'expected':expected,'predicted':label,
                     'auto_correct':bool(auto and label==expected),
                     'auto_wrong':bool(auto and label!=expected),
                     'initial_resolved':initial_resolved,
                     'human':requested,'timeout':timeout,'checks':len(actions),
                     'actions':actions,'elapsed_s':env.elapsed_s,
                     'commanded_water_ml':env.used_ml,'return':total})
    summary = {k:float(np.mean([r[k] for r in rows])) for k in
               ('auto_correct','auto_wrong','human','timeout','initial_resolved',
                'checks','elapsed_s','commanded_water_ml','return')}
    summary['automatic_coverage'] = summary['auto_correct']+summary['auto_wrong']
    summary['conditional_auto_accuracy'] = (
        summary['auto_correct']/summary['automatic_coverage'] if summary['automatic_coverage'] else None)
    # Bootstrap independent episodes, not repeated sensor timestamps.
    rng = np.random.default_rng(seed+999)
    arr = np.array([r['auto_wrong'] for r in rows],dtype=float)
    boot = np.array([rng.choice(arr,size=len(arr),replace=True).mean() for _ in range(1000)])
    summary['auto_wrong_bootstrap95'] = np.quantile(boot,[.025,.975]).tolist()
    payload = {'domain':'SIM_ONLY_UNCALIBRATED_NOT_REAL_PERFORMANCE',
               'policy':str(checkpoint or 'fixed'), 'episodes':episodes,
               'seed_start':seed,'summary':summary,'episodes_detail':rows}
    p=Path(output); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    return payload

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint'); p.add_argument('--episodes',type=int,default=120)
    p.add_argument('--seed',type=int,default=900001); p.add_argument('--out',default='runs/eval.json')
    a=p.parse_args(); evaluate(a.checkpoint,a.episodes,a.seed,a.out)
