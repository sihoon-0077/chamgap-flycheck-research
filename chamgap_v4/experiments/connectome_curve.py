"""connectome_v1 learning curve: real hemibrain PN->KC mask vs rewired, random fan-in and MLP.

Pre-registration: experiments/PREREG_connectome_v1.md (its SHA-256 goes into the manifest).
Runtime configuration: experiments/sim_v2.py (reward v2 + rule A), applied in every process.
Appendix files are not modified. domain=SIM_ONLY_UNCALIBRATED for every row.

Stages (run from chamgap_v4/, in this order):
  python -m experiments.connectome_curve --stage smoke   # 2 sizes x 2 seeds, timing + reproducibility
  python -m experiments.connectome_curve --stage tune    # learning rate per model, validation set
  python -m experiments.connectome_curve --stage main    # the pre-registered experiment
  python -m experiments.connectome_curve --stage extend  # amendment 1: 80 fresh seeds at n=200
"""
from __future__ import annotations
import os
# One BLAS/OpenMP thread per process, set before numpy/torch load (workers inherit it).
# Without this, 10 workers x multi-threaded BLAS oversubscribed the 8-core laptop CPU.
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '1')
import argparse
import hashlib
import importlib
import itertools
import json
import platform
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from chamgap.collect import collect
from chamgap.connectome import load_mask, rewire, save_mask
from chamgap.contracts import D, F
from chamgap.simulator import SmartFarmCore
from experiments import sim_v2
from experiments.learning_curve import (BATCH, REPLICAS, TARGET_STEPS, load_any, make_value_model,
                                        patched, quiet, step_budget, write_csv)

train_mod = importlib.import_module('chamgap.train')
eval_mod = importlib.import_module('chamgap.evaluate')

DOMAIN = 'SIM_ONLY_UNCALIBRATED'
REAL_MASK = Path('data/connectome/hemibrain_v121_caR_olf51.npy')
PREREG = Path('experiments/PREREG_connectome_v1.md')
TEST = (900001, 1000)          # (first seed, episodes): the same test events for every condition
VALIDATION = (800001, 500)     # tuning only
SMALL, LARGE = (50, 100, 200), (500, 1000, 2000)
LRS = (.001, .003, .01)
TUNE_N = 200
APPENDIX_LR = .003             # what chamgap.train passes to Adam; we override it explicitly


def pick_seeds(count=33, start=(17, 29, 43, 61, 79)):
    """Seeds whose pairwise gaps avoid collect()'s future-seed lattice seed + 1009g + 101d + r."""
    lattice = {(101 * dd + dr) % 1009 for dd in range(-4, 5) for dr in range(-2, 3)}
    chosen, c, misses = list(start), start[-1] + 1, 0
    while len(chosen) < count:
        if all((c - s) % 1009 not in lattice for s in chosen):
            chosen.append(c)
            misses = 0
        else:
            misses += 1
            if misses > 1009:   # every residue is blocked: no further seed can ever qualify
                raise ValueError(f'only {len(chosen)} lattice-safe seeds exist from this start')
        c += 1
    return chosen


ALL_SEEDS = pick_seeds()
DATA_SEEDS, TUNE_SEEDS = ALL_SEEDS[:30], ALL_SEEDS[30:]
LARGE_SEEDS = DATA_SEEDS[:10]
# Amendment 1 (PREREG section 7): 80 fresh seeds at n=200. pick_seeds() cannot go much past ~35
# seeds (each seed blocks ~45 of the 1009 residues), so these are spaced wider than a whole
# pool's rollout-seed span instead, which makes every span disjoint by construction.
EXTEND_N = 200
EXTEND_SEEDS = tuple(3_000_000 + 250_000 * k for k in range(80))


def rollout_span(seed, pool):
    """Inclusive range of collect()'s rollout seeds seed + 1009g + 101d + r (d <= 4, r <= 2)."""
    return seed, seed + 1009 * (pool - 1) + 101 * 4 + 2


def check_extend_spans():
    """New spans must be disjoint from each other and lie above every earlier span.
    (Earlier spans overlap each other by design; the residue lattice keeps their seeds apart.)"""
    used = ([rollout_span(s, max(LARGE) if s in LARGE_SEEDS else max(SMALL)) for s in DATA_SEEDS]
            + [rollout_span(s, TUNE_N) for s in TUNE_SEEDS])
    new = sorted(rollout_span(s, EXTEND_N) for s in EXTEND_SEEDS)
    if new[0][0] <= max(end for _, end in used):
        raise RuntimeError('extension spans overlap earlier pools')
    if any(a[1] >= b[0] for a, b in zip(new, new[1:])) or new[-1][1] >= 2**31:
        raise RuntimeError('extension spans overlap each other or exceed the seed range')

MODELS = {
    'mlp_matched': {'kind': 'mlp', 'id': 'P1', 'connectome': 'NONE'},
    'fly_rewired': {'kind': 'fly', 'id': 'P2', 'connectome': 'REWIRED_FROM_HEMIBRAIN_NOT_BIOLOGICAL'},
    'fly_real': {'kind': 'fly', 'id': 'P3'},   # label read from the mask's provenance
    'fly_random_fanin': {'kind': 'fly', 'id': 'R', 'connectome': 'RANDOM_FANIN_MATCHED_NOT_BIOLOGICAL'},
}
EPISODE_FIELDS = ('stage', 'model', 'n_episodes', 'seed', 'lr', 'episode', 'expected', 'predicted',
                  'initial_resolved', 'auto_correct', 'auto_wrong', 'human', 'timeout', 'checks',
                  'first_action', 'elapsed_s', 'water_ml', 'return', 'domain', 'connectome')


def dirs(stage):
    tag = '' if stage == 'main' else f'_{stage}'
    return (Path(f'data/learning_curve_v2{tag}'), Path(f'runs/learning_curve_v2{tag}'),
            Path('results/connectome_v1') / ('' if stage == 'main' else stage))


def real_mask():
    m, meta = load_mask(REAL_MASK)
    if meta.get('source_kind') != 'neuprint-hemibrain-rule-selected-pn-kc':
        raise SystemExit('real mask provenance is not the connectome_v1 rule-selected hemibrain mask')
    return m, meta


def matched_hidden(mask):
    """MLP width whose trainable parameters (33h + 4) match the fly model (29P + 4K + 4)."""
    k, p = mask.shape
    fly_params = (D + 1) * p + 4 * k + 4
    return round((fly_params - 4) / (D + 5)), fly_params


def random_fanin(mask, seed):
    """Keep every KC's input count; draw its glomeruli uniformly without replacement."""
    rng = np.random.default_rng(seed)
    out = np.zeros_like(mask, dtype=np.float32)
    for k, deg in enumerate((mask > 0).sum(1)):
        out[k, rng.choice(mask.shape[1], size=int(deg), replace=False)] = 1.
    assert np.array_equal(out.sum(1), (mask > 0).sum(1))
    return out


def prepare_seed(job):
    """Pool of `pool` episodes (size-N datasets = first N groups) plus this seed's control masks."""
    stage, seed, pool, sizes = job
    data_dir = dirs(stage)[0]
    data_dir.mkdir(parents=True, exist_ok=True)
    reset_seeds = []
    original_reset = SmartFarmCore.reset

    def logging_reset(self, *, seed=None, options=None):
        reset_seeds.append(seed)
        return original_reset(self, seed=seed, options=options)

    path = data_dir / f'pool_seed{seed}.npz'
    with patched(SmartFarmCore, 'reset', logging_reset), quiet():
        collect(str(path), pool, REPLICAS, seed)
    with np.load(path, allow_pickle=False) as d:
        arrays = {k: d[k].copy() for k in d.files}
    subsets = {}
    for n in sizes:
        keep = arrays['group'] < n
        np.savez_compressed(data_dir / f'seed{seed}_n{n}.npz', **{k: v[keep] for k, v in arrays.items()})
        subsets[n] = {'rows': int(keep.sum()), 'groups': int(len(np.unique(arrays['group'][keep])))}
    with tempfile.TemporaryDirectory() as tmp, quiet():
        direct = Path(tmp) / 'direct.npz'
        collect(str(direct), min(sizes), REPLICAS, seed)
        with np.load(direct) as a, np.load(data_dir / f'seed{seed}_n{min(sizes)}.npz') as b:
            prefix_ok = all(np.array_equal(a[k], b[k]) for k in a.files)
    if not prefix_ok:
        raise RuntimeError(f'collect prefix property failed for seed {seed}')
    # Rule A at depth 0: INSPECT never has a target before the first check.
    x, tm = arrays['x'], arrays['target_mask']
    first = x[:, F['resample_count']] + x[:, F['compare_count']] + x[:, F['pulse_count']] == 0
    if tm[first, 3].any():
        raise RuntimeError('rule A not active during collection')
    real, meta = real_mask()
    rewired, info = rewire(real, seed)
    save_mask(data_dir / f'rewired_seed{seed}.npy', rewired, 'degree-preserving-rewired',
              {'parent_sha256': meta['sha256'], 'parent_source_kind': meta['source_kind'], 'seed': seed, **info})
    save_mask(data_dir / f'random_fanin_seed{seed}.npy', random_fanin(real, seed), 'random-fanin-matched',
              {'parent_sha256': meta['sha256'], 'seed': seed,
               'rule': 'per-KC input count kept, glomeruli drawn uniformly without replacement'})
    return {'seed': seed, 'pool': pool, 'reset_seeds': reset_seeds, 'subsets': subsets,
            'prefix_check': prefix_ok, 'rewire': info}


def evaluate_on(checkpoint, start, count):
    with tempfile.TemporaryDirectory() as tmp, patched(eval_mod, 'load_model', load_any), quiet():
        payload = eval_mod.evaluate(None if checkpoint is None else str(checkpoint), count, start,
                                    str(Path(tmp) / 'eval.json'))
    return payload['episodes_detail']


def verify_detail(detail):
    """Every evaluated episode obeys rule A and reward v2."""
    c = sim_v2.config_v2()
    for r in detail:
        if r['actions'] and r['actions'][0] == 3:
            raise RuntimeError(f'rule A violated in episode {r["episode"]}')
        if r['initial_resolved']:
            expected = 0.
        else:
            expected = (c.correct_reward * r['auto_correct'] + c.wrong_reward * r['auto_wrong']
                        - c.inspection_cost * r['human'] - c.timeout_cost * r['timeout']
                        - c.time_cost_per_s * r['elapsed_s'] - c.water_cost_per_ml * r['commanded_water_ml'])
        if abs(expected - r['return']) > 1e-9:
            raise RuntimeError(f'reward v2 mismatch in episode {r["episode"]}')


def episode_rows(stage, model, n, seed, lr, connectome, detail):
    return [{'stage': stage, 'model': model, 'n_episodes': n, 'seed': seed, 'lr': lr,
             'episode': r['episode'], 'expected': r['expected'], 'predicted': r['predicted'],
             'initial_resolved': int(r['initial_resolved']), 'auto_correct': int(r['auto_correct']),
             'auto_wrong': int(r['auto_wrong']), 'human': int(r['human']), 'timeout': int(r['timeout']),
             'checks': r['checks'], 'first_action': r['actions'][0] if r['actions'] else -1,
             'elapsed_s': r['elapsed_s'], 'water_ml': r['commanded_water_ml'], 'return': r['return'],
             'domain': DOMAIN, 'connectome': connectome} for r in detail]


def adam_with(chosen_lr):
    """Stand-in for torch.optim.Adam that swaps chamgap.train's hard-coded lr for chosen_lr."""
    original = torch.optim.Adam

    def make(params, lr=None, **kwargs):
        if lr != APPENDIX_LR:
            raise RuntimeError(f'chamgap.train passed lr={lr}; expected {APPENDIX_LR}')
        return original(params, lr=chosen_lr, **kwargs)
    return make


def run_job(job):
    stage, model, n, seed, lr, eval_set = job
    data_dir, run_dir, _ = dirs(stage)
    spec = MODELS[model]
    real, meta = real_mask()
    hidden, _ = matched_hidden(real)
    matrix = {'fly_real': REAL_MASK, 'fly_rewired': data_dir / f'rewired_seed{seed}.npy',
              'fly_random_fanin': data_dir / f'random_fanin_seed{seed}.npy'}.get(model)
    connectome = meta['connectome_label'] if model == 'fly_real' else spec['connectome']
    data = data_dir / f'seed{seed}_n{n}.npz'
    epochs, steps, train_rows, train_groups = step_budget(data, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = run_dir / f'{model}_n{n}_s{seed}_lr{lr}.pt'

    def factory(kind, adjacency=None, top_fraction=.05):
        return make_value_model(kind, adjacency, top_fraction, hidden)

    t0 = time.perf_counter()
    with patched(train_mod, 'ValueModel', factory), patched(torch.optim, 'Adam', adam_with(lr)), quiet():
        train_mod.train(str(data), str(ckpt), spec['kind'], None if matrix is None else str(matrix),
                        False, epochs, seed, 'cpu')
    t1 = time.perf_counter()
    info = json.loads(ckpt.with_suffix('.json').read_text(encoding='utf-8'))
    if info['train_groups'] != train_groups:
        raise RuntimeError('step budget was computed on a different split than train() used')
    detail = evaluate_on(ckpt, *(TEST if eval_set == 'test' else VALIDATION))
    verify_detail(detail)
    t2 = time.perf_counter()
    projection = info['projection'] or {}
    run = {'stage': stage, 'model': model, 'policy_id': spec['id'], 'n_episodes': n, 'seed': seed,
           'lr': lr, 'eval_set': eval_set, 'params': info['trainable_parameters'],
           'mean_return': float(np.mean([r['return'] for r in detail])),
           'matrix_sha256': projection.get('sha256', ''),
           'rewire_edge_overlap': projection.get('edge_overlap_fraction', ''),
           'epochs': epochs, 'optimizer_steps': steps, 'train_rows': train_rows,
           'best_val_huber': info['validation_huber'], 'internal_test_huber': info['test_huber'],
           'train_s': round(t1 - t0, 2), 'eval_s': round(t2 - t1, 2),
           'domain': DOMAIN, 'connectome': connectome}
    return run, episode_rows(stage, model, n, seed, lr, connectome, detail)


def baseline_job(stage):
    detail = evaluate_on(None, *TEST)
    verify_detail(detail)
    return episode_rows(stage, 'fixed_P0', 0, 0, 0., 'NONE', detail)


def code_sha256():
    files = sorted(Path('chamgap').glob('*.py')) + [Path('experiments') / f for f in (
        'connectome_curve.py', 'sim_v2.py', 'learning_curve.py', 'build_real_connectome.py')]
    return {f.as_posix(): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


def _init_worker():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sim_v2.apply()


def plan(stage, lr_choice):
    """(pool jobs, training jobs) for a stage."""
    if stage == 'smoke':
        seeds, sizes = DATA_SEEDS[:2], (50, 200)
        pools = [(stage, s, max(sizes), sizes) for s in seeds]
        jobs = [(stage, m, n, s, APPENDIX_LR, 'test') for m in MODELS for n in sizes for s in seeds]
    elif stage == 'tune':
        pools = [(stage, s, TUNE_N, (50, TUNE_N)) for s in TUNE_SEEDS]
        jobs = [(stage, m, TUNE_N, s, lr, 'validation') for m in MODELS for lr in LRS for s in TUNE_SEEDS]
    elif stage == 'extend':
        check_extend_spans()
        pools = [(stage, s, EXTEND_N, (50, EXTEND_N)) for s in EXTEND_SEEDS]   # n=50 only for the prefix check
        jobs = [(stage, m, EXTEND_N, s, lr_choice[m], 'test') for m in MODELS for s in EXTEND_SEEDS]
    else:
        pools = [(stage, s, max(LARGE) if s in LARGE_SEEDS else max(SMALL),
                  SMALL + LARGE if s in LARGE_SEEDS else SMALL) for s in DATA_SEEDS]
        jobs = [(stage, m, n, s, lr_choice[m], 'test') for m in MODELS
                for n, s in itertools.chain(itertools.product(SMALL, DATA_SEEDS),
                                            itertools.product(LARGE, LARGE_SEEDS))]
    return pools, jobs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=('smoke', 'tune', 'main', 'extend'), required=True)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    if not Path('chamgap').is_dir():
        raise SystemExit('run from chamgap_v4/')
    applied = sim_v2.apply()
    real, meta = real_mask()
    hidden, fly_params = matched_hidden(real)
    _, _, out = dirs(a.stage)
    out.mkdir(parents=True, exist_ok=True)
    lr_choice = None
    if a.stage in ('main', 'extend'):
        lr_choice = json.loads((dirs('tune')[2] / 'tuning_choice.json').read_text(encoding='utf-8'))['lr']
    started = time.time()
    code = code_sha256()
    prereg_sha = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    pools, jobs = plan(a.stage, lr_choice)
    print(f'stage={a.stage}: {len(pools)} pools, {len(jobs)} training runs, real mask {real.shape}, '
          f'fly params {fly_params}, matched MLP hidden {hidden}', flush=True)

    with ProcessPoolExecutor(min(a.workers, len(pools)), initializer=_init_worker) as ex:
        prepared = list(ex.map(prepare_seed, pools))
    held_out = set(range(TEST[0], TEST[0] + TEST[1])) | set(range(VALIDATION[0], VALIDATION[0] + VALIDATION[1]))
    train_seeds = {s for item in prepared for s in item['reset_seeds']}
    overlap = sorted(train_seeds & held_out)
    if overlap:
        raise RuntimeError(f'training episodes reuse test/validation seeds: {overlap[:5]}')
    print(f'prepared pools in {time.time() - started:.0f}s; train vs test/validation seed overlap = 0', flush=True)

    runs, episodes = [], []
    with ProcessPoolExecutor(a.workers, initializer=_init_worker) as ex:
        baseline = ex.submit(baseline_job, a.stage) if a.stage in ('smoke', 'main') else None
        futures = {ex.submit(run_job, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            run, rows = fut.result()
            runs.append(run)
            episodes.extend(rows)
            print(f"[{i:3d}/{len(jobs)}] {run['model']:<17} n={run['n_episodes']:<5} seed={run['seed']:<4} "
                  f"lr={run['lr']:<6} return={run['mean_return']:.3f} ({run['train_s'] + run['eval_s']:.0f}s)",
                  flush=True)
        if baseline is not None:
            episodes.extend(baseline.result())

    key = lambda r: (r['model'], r['n_episodes'], r['seed'], r['lr'])
    runs.sort(key=key)
    episodes.sort(key=lambda r: (*key(r), r['episode']))
    write_csv(out / 'runs.csv', runs, list(runs[0]))
    write_csv(out / 'episodes.csv.gz', episodes, EPISODE_FIELDS, compress=True)
    manifest = {
        'stage': a.stage, 'domain': DOMAIN,
        'claim_scope': 'learning-curve behaviour in an uncalibrated toy simulator only',
        'prereg': {'path': PREREG.as_posix(), 'sha256': prereg_sha},
        'created_unix': started, 'wall_seconds': round(time.time() - started, 1),
        'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
        'code_sha256': code, 'sim_v2': applied,
        'real_mask': {'path': REAL_MASK.as_posix(), 'sha256': meta['sha256'], 'shape_kc_pn': meta['shape_kc_pn'],
                      'edges': meta['edges'], 'label': meta['connectome_label'],
                      'review_status': meta['review_status'], 'license_url': meta['license_url']},
        'design': {'models': {m: {k: v for k, v in s.items()} for m, s in MODELS.items()},
                   'matched_mlp_hidden': hidden, 'fly_params': fly_params,
                   'target_optimizer_steps': TARGET_STEPS, 'batch': BATCH, 'replicas': REPLICAS,
                   'test': list(TEST), 'validation': list(VALIDATION), 'lr_grid': LRS,
                   'lr_choice': lr_choice, 'jobs': len(jobs)},
        'checks': {'train_vs_heldout_seed_overlap': len(overlap), 'train_reset_seeds_logged': len(train_seeds),
                   'prefix_property': {str(x['seed']): x['prefix_check'] for x in prepared},
                   'rule_a_and_reward_v2_verified_per_episode': True},
        'pools': {str(x['seed']): {'pool': x['pool'], 'subsets': x['subsets'], 'rewire': x['rewire']}
                  for x in prepared},
    }
    if a.stage == 'tune':
        by = {}
        for r in runs:
            by.setdefault((r['model'], r['lr']), []).append(r['mean_return'])
        means = {m: {str(lr): float(np.mean(by[(m, lr)])) for lr in LRS} for m in MODELS}
        choice = {}
        for m in MODELS:
            best = max(means[m].values())
            tied = [lr for lr in LRS if np.isclose(means[m][str(lr)], best)]
            choice[m] = APPENDIX_LR if APPENDIX_LR in tied else tied[0]
        (out / 'tuning_choice.json').write_text(json.dumps(
            {'rule': 'highest mean validation return over tuning seeds; tie -> 0.003',
             'validation_mean_return': means, 'lr': choice}, indent=2), encoding='utf-8')
        manifest['tuning'] = {'validation_mean_return': means, 'lr': choice}
        print(json.dumps(manifest['tuning'], indent=2), flush=True)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'wrote {out} in {time.time() - started:.0f}s', flush=True)


if __name__ == '__main__':
    main()
