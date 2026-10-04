"""Learning-curve experiment (design doc 5.4) on the UNMODIFIED appendix code.

domain=SIM_ONLY_UNCALIBRATED, connectome=DEMO_RANDOM_NOT_BIOLOGICAL.
The fly models use the random demo mask or degree-preserving rewires of it, so
nothing here is evidence about the real Drosophila PN->KC connectome.

Stage 1 (this file) trains and closed-loop evaluates every (model, n_episodes,
seed) and stores PER-EPISODE outcomes. Aggregation scopes (all events, events
whose truth is not UNKNOWN) are chosen later in analyze_learning_curve.py.

Run from the repository root:  python -m experiments.learning_curve
"""
from __future__ import annotations
import argparse
import contextlib
import csv
import gzip
import hashlib
import importlib
import io
import itertools
import json
import math
import platform
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from torch import nn

from chamgap.collect import collect
from chamgap.connectome import load_mask, rewire, save_mask
from chamgap.contracts import D, FEATURES
from chamgap.model import ValueModel
from chamgap.simulator import SmartFarmCore

# Submodules, not the same-named functions they define.
train_mod = importlib.import_module('chamgap.train')
eval_mod = importlib.import_module('chamgap.evaluate')
ORIGINAL_LOAD = train_mod.load_model

DOMAIN = 'SIM_ONLY_UNCALIBRATED'
CONNECTOME = 'DEMO_RANDOM_NOT_BIOLOGICAL'
SIZES = (50, 100, 200, 500, 1000, 2000)
# Pairwise gaps avoid collect()'s future-seed lattice (seed + 1009*g + 101*d + r).
SEEDS = (17, 29, 43, 61, 79)
REPLICAS = 3                                    # same as 16.3
TEST_SEED_START, TEST_EPISODES = 900001, 1000   # one fixed test set for every condition
TARGET_STEPS = 1500                             # optimizer updates per run, for every data size
BATCH = 64                                      # chamgap.train uses minibatches of 64
DEMO_MASK = Path('data/demo_mask.npy')
DATA_DIR = Path('data/learning_curve')
RUN_DIR = Path('runs/learning_curve')

# MLP trainable params = 33*hidden + 4: h64 -> 2,116; h44 -> 1,456 (fly: 1,444).
MODELS = {
    'mlp_h64': {'kind': 'mlp', 'hidden': 64,
                'structure': 'dense MLP 28-64-4, no connectome'},
    'mlp_h44_matched': {'kind': 'mlp', 'hidden': 44,
                        'structure': 'dense MLP 28-44-4, no connectome, params matched to fly'},
    'fly_random': {'kind': 'fly', 'structure': 'demo random PN-KC mask (mask seed 11)'},
    'fly_rewired': {'kind': 'fly', 'structure': 'degree-preserving rewire of the demo mask '
                                                '(rewire seed = run seed)'},
}
EPISODE_FIELDS = ('model', 'n_episodes', 'seed', 'episode', 'expected', 'predicted',
                  'initial_resolved', 'auto_correct', 'auto_wrong', 'human', 'timeout',
                  'checks', 'elapsed_s', 'water_ml', 'domain', 'connectome')


@contextlib.contextmanager
def patched(owner, name, value):
    original = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, original)


def quiet():
    # collect/train/evaluate print JSON summaries; the runner prints its own progress.
    return contextlib.redirect_stdout(io.StringIO())


def make_value_model(kind, adjacency=None, top_fraction=.05, hidden=64):
    m = ValueModel(kind, adjacency, top_fraction)
    if kind == 'mlp' and hidden != 64:
        m.net = nn.Sequential(nn.Linear(D, hidden), nn.ReLU(), nn.Linear(hidden, 4))
    return m


def load_any(path):
    """chamgap.train.load_model, extended to MLPs whose hidden width is not 64."""
    ck = torch.load(path, map_location='cpu', weights_only=True)
    hidden = ck['state_dict']['net.0.weight'].shape[0] if ck['kind'] == 'mlp' else 64
    if hidden == 64:
        return ORIGINAL_LOAD(path)
    if ck['features'] != list(FEATURES):
        raise ValueError('feature schema mismatch')
    m = make_value_model('mlp', None, ck['top_fraction'], hidden)
    m.load_state_dict(ck['state_dict'])
    m.eval()
    return m


def step_budget(data_path, seed):
    """Reproduce chamgap.train's group split to turn TARGET_STEPS into epochs."""
    with np.load(data_path, allow_pickle=False) as d:
        groups = d['group'].copy()
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    rng.shuffle(unique)
    train_groups = unique[:max(1, int(len(unique) * .7))]
    train_rows = int(np.isin(groups, train_groups).sum())
    batches = math.ceil(train_rows / BATCH)
    epochs = math.ceil(TARGET_STEPS / batches)
    return epochs, epochs * batches, train_rows, train_groups.tolist()


def prepare_seed(data_seed):
    """Pool of max(SIZES) episodes; size-N datasets are its first N episode groups."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pool = DATA_DIR / f'pool_seed{data_seed}.npz'
    reset_seeds = []
    original_reset = SmartFarmCore.reset

    def logging_reset(self, *, seed=None, options=None):
        reset_seeds.append(seed)
        return original_reset(self, seed=seed, options=options)

    with patched(SmartFarmCore, 'reset', logging_reset), quiet():
        collect(str(pool), max(SIZES), REPLICAS, data_seed)
    with np.load(pool, allow_pickle=False) as d:
        arrays = {k: d[k].copy() for k in d.files}
    subsets = {}
    for n in SIZES:
        keep = arrays['group'] < n
        path = DATA_DIR / f'seed{data_seed}_n{n}.npz'
        np.savez_compressed(path, **{k: v[keep] for k, v in arrays.items()})
        subsets[n] = {'rows': int(keep.sum()),
                      'groups': int(len(np.unique(arrays['group'][keep])))}
    # The prefix design is only valid if collect(N) == first N groups of collect(max).
    with tempfile.TemporaryDirectory() as tmp, quiet():
        direct = Path(tmp) / 'direct.npz'
        collect(str(direct), min(SIZES), REPLICAS, data_seed)
        with np.load(direct) as a, np.load(DATA_DIR / f'seed{data_seed}_n{min(SIZES)}.npz') as b:
            prefix_ok = all(np.array_equal(a[k], b[k]) for k in a.files)
    if not prefix_ok:
        raise RuntimeError(f'collect prefix property failed for seed {data_seed}')
    mask, meta = load_mask(DEMO_MASK)
    rewired, info = rewire(mask, data_seed)
    save_mask(DATA_DIR / f'rewired_seed{data_seed}.npy', rewired, 'degree-preserving-rewired',
              {'parent_sha256': meta['sha256'], 'parent_source_kind': meta['source_kind'],
               'seed': data_seed, **info})
    return {'seed': data_seed, 'reset_seeds': reset_seeds, 'subsets': subsets,
            'prefix_check': prefix_ok, 'rewire': info}


def evaluate_policy(checkpoint):
    """Run the appendix evaluator unchanged; keep only its per-episode rows."""
    with tempfile.TemporaryDirectory() as tmp, patched(eval_mod, 'load_model', load_any), quiet():
        payload = eval_mod.evaluate(None if checkpoint is None else str(checkpoint),
                                    TEST_EPISODES, TEST_SEED_START, str(Path(tmp) / 'eval.json'))
    return payload['episodes_detail']


def episode_rows(model, n, seed, detail):
    return [{'model': model, 'n_episodes': n, 'seed': seed, 'episode': r['episode'],
             'expected': r['expected'], 'predicted': r['predicted'],
             'initial_resolved': int(r['initial_resolved']), 'auto_correct': int(r['auto_correct']),
             'auto_wrong': int(r['auto_wrong']), 'human': int(r['human']),
             'timeout': int(r['timeout']), 'checks': r['checks'], 'elapsed_s': r['elapsed_s'],
             'water_ml': r['commanded_water_ml'], 'domain': DOMAIN, 'connectome': CONNECTOME}
            for r in detail]


def run_job(job):
    model, n, seed = job
    spec = MODELS[model]
    data = DATA_DIR / f'seed{seed}_n{n}.npz'
    matrix = {'fly_random': DEMO_MASK,
              'fly_rewired': DATA_DIR / f'rewired_seed{seed}.npy'}.get(model)
    epochs, steps, train_rows, train_groups = step_budget(data, seed)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = RUN_DIR / f'{model}_n{n}_s{seed}.pt'
    hidden = spec.get('hidden', 64)

    def factory(kind, adjacency=None, top_fraction=.05):
        return make_value_model(kind, adjacency, top_fraction, hidden)

    t0 = time.perf_counter()
    with patched(train_mod, 'ValueModel', factory), quiet():
        train_mod.train(str(data), str(ckpt), spec['kind'], None if matrix is None else str(matrix),
                        False, epochs, seed, 'cpu')
    t1 = time.perf_counter()
    meta = json.loads(ckpt.with_suffix('.json').read_text(encoding='utf-8'))
    if meta['train_groups'] != train_groups:
        raise RuntimeError('step budget was computed on a different split than train() used')
    detail = evaluate_policy(ckpt)
    t2 = time.perf_counter()
    projection = meta['projection'] or {}
    run = {'model': model, 'n_episodes': n, 'seed': seed,
           'params': meta['trainable_parameters'], 'structure': spec['structure'],
           'matrix_sha256': projection.get('sha256', ''),
           'rewire_successful_swaps': projection.get('successful_swaps', ''),
           'rewire_edge_overlap': projection.get('edge_overlap_fraction', ''),
           'epochs': epochs, 'optimizer_steps': steps, 'train_rows': train_rows,
           'best_val_huber': meta['validation_huber'], 'internal_test_huber': meta['test_huber'],
           'train_s': round(t1 - t0, 2), 'eval_s': round(t2 - t1, 2),
           'domain': DOMAIN, 'connectome': CONNECTOME}
    return run, episode_rows(model, n, seed, detail)


def code_sha256():
    """Per-file SHA-256 of the simulator core and this runner, paths relative to the repo root."""
    runner = Path(__file__).resolve().relative_to(Path.cwd().resolve())
    files = sorted(Path('chamgap').glob('*.py')) + [runner]
    return {f.as_posix(): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


def write_csv(path, rows, fields, compress=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if compress else open
    with opener(path, 'wt', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    global SIZES, SEEDS
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', default='results', type=Path)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--sizes', type=int, nargs='+', default=list(SIZES))
    p.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    p.add_argument('--models', nargs='+', default=list(MODELS), choices=list(MODELS))
    a = p.parse_args()
    if not (Path('chamgap').is_dir() and DEMO_MASK.exists()):
        raise SystemExit('run from chamgap_v4/ after: python -m chamgap.connectome demo ...')
    if load_mask(DEMO_MASK)[1].get('source_kind') != CONNECTOME:
        raise SystemExit('demo mask provenance is not DEMO_RANDOM_NOT_BIOLOGICAL')
    SIZES, SEEDS = tuple(sorted(a.sizes)), tuple(a.seeds)
    started = time.time()
    code = code_sha256()   # hashed BEFORE running: the code that produced these results

    with ProcessPoolExecutor(min(a.workers, len(SEEDS)), initializer=_init_worker,
                             initargs=(SIZES,)) as ex:
        prepared = list(ex.map(prepare_seed, SEEDS))
    test_seeds = set(range(TEST_SEED_START, TEST_SEED_START + TEST_EPISODES))
    train_seeds = {s for item in prepared for s in item['reset_seeds']}
    overlap = sorted(train_seeds & test_seeds)
    if overlap:
        raise RuntimeError(f'training episodes reuse test seeds: {overlap[:5]}')
    print(f'prepared {len(SEEDS)} seed pools; train/test seed overlap = 0', flush=True)

    jobs = list(itertools.product(a.models, SIZES, SEEDS))
    runs, episodes = [], []
    with ProcessPoolExecutor(a.workers, initializer=_init_worker, initargs=(SIZES,)) as ex:
        baseline = ex.submit(evaluate_policy, None)
        futures = {ex.submit(run_job, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            run, rows = fut.result()
            runs.append(run); episodes.extend(rows)
            ac = np.mean([r['auto_correct'] for r in rows])
            print(f"[{i:3d}/{len(jobs)}] {run['model']:<16} n={run['n_episodes']:<5} "
                  f"seed={run['seed']:<3} params={run['params']:<5} steps={run['optimizer_steps']:<5} "
                  f"auto_correct={ac:.3f} ({run['train_s']+run['eval_s']:.0f}s)", flush=True)
        episodes.extend(episode_rows('fixed_P0', 0, 0, baseline.result()))

    order = {m: i for i, m in enumerate(MODELS)}
    runs.sort(key=lambda r: (order[r['model']], r['n_episodes'], r['seed']))
    episodes.sort(key=lambda r: (order.get(r['model'], -1), r['n_episodes'], r['seed'], r['episode']))
    out = a.out_dir
    write_csv(out / 'learning_curve_runs.csv', runs, list(runs[0]))
    write_csv(out / 'learning_curve_episodes.csv.gz', episodes, EPISODE_FIELDS, compress=True)
    manifest = {
        'domain': DOMAIN, 'connectome': CONNECTOME,
        'claim_scope': 'pipeline/learning-curve behaviour in an uncalibrated toy simulator only',
        'created_unix': started, 'wall_seconds': round(time.time() - started, 1),
        'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
        'code_sha256': code,
        'design': {'sizes': SIZES, 'seeds': SEEDS, 'replicas': REPLICAS,
                   'n_episodes_meaning': 'episodes requested from collect(); groups resolved at '
                                         'reset are dropped; train() then splits groups 70/15/15',
                   'target_optimizer_steps': TARGET_STEPS, 'batch': BATCH,
                   'checkpoint_selection': 'chamgap.train best validation epoch',
                   'test_seeds': [TEST_SEED_START, TEST_SEED_START + TEST_EPISODES - 1],
                   'models': {m: {k: v for k, v in s.items()} for m, s in MODELS.items()
                              if m in a.models}},
        'checks': {'train_test_seed_overlap': len(overlap),
                   'train_reset_seeds_logged': len(train_seeds),
                   'prefix_property': {p['seed']: p['prefix_check'] for p in prepared}},
        'datasets': {p['seed']: p['subsets'] for p in prepared},
        'rewire': {p['seed']: p['rewire'] for p in prepared},
    }
    (out / 'learning_curve_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'wrote {out}/learning_curve_runs.csv, learning_curve_episodes.csv.gz, '
          f'learning_curve_manifest.json in {time.time() - started:.0f}s', flush=True)


def _init_worker(sizes):
    # Spawned workers re-import this module; carry the CLI-selected sizes over.
    global SIZES
    SIZES = tuple(sizes)


if __name__ == '__main__':
    main()
