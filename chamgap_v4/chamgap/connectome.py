from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def random_mask(kc: int = 128, pn: int = 32, fanin: int = 6, seed: int = 11):
    if not 1 <= fanin <= pn or kc < 1:
        raise ValueError('invalid projection dimensions')
    r = np.random.default_rng(seed)
    m = np.zeros((kc, pn), dtype=np.float32)
    for k in range(kc):
        m[k, r.choice(pn, size=fanin, replace=False)] = 1.
    return m


def rewire(m: np.ndarray, seed: int, swaps_per_edge: int = 10):
    """Bipartite double-edge swaps preserve PN and KC degree EXACTLY.

    Does not claim uniform sampling over all degree-constrained graphs.
    Weight-distribution preservation is a separate experiment; this is binary.
    """
    r = np.random.default_rng(seed)
    out = (m > 0).astype(np.float32)
    edges = np.argwhere(out > 0)
    requested = len(edges)*swaps_per_edge
    done = 0
    for _ in range(max(1, requested*30)):
        if done >= requested or len(edges) < 2:
            break
        i, j = r.choice(len(edges), size=2, replace=False)
        u, a = edges[i]; v, b = edges[j]
        if u == v or a == b or out[u,b] or out[v,a]:
            continue
        out[u,a] = out[v,b] = 0.
        out[u,b] = out[v,a] = 1.
        edges[i] = (u,b); edges[j] = (v,a)
        done += 1
    assert np.array_equal(out.sum(0), (m > 0).sum(0))
    assert np.array_equal(out.sum(1), (m > 0).sum(1))
    return out, {'requested_swaps': requested, 'successful_swaps': done,
                 'edge_overlap_fraction': float(((m > 0) & (out > 0)).sum()/max(1,len(edges)))}


def save_mask(path, m, source_kind, details=None):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, m.astype(np.float32), allow_pickle=False)
    meta = {'source_kind': source_kind, 'shape_kc_pn': list(m.shape),
            'edges': int((m > 0).sum()), 'sha256': sha256(p), **(details or {})}
    p.with_suffix('.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')


def load_mask(path, require_real=False):
    p = Path(path)
    m = np.load(p, allow_pickle=False)
    meta = json.loads(p.with_suffix('.json').read_text(encoding='utf-8'))
    if sha256(p) != meta['sha256']:
        raise ValueError('projection file hash mismatch')
    if m.ndim != 2 or not np.isfinite(m).all() or not (m >= 0).all():
        raise ValueError('invalid connection matrix')
    if np.any(m.sum(1) == 0):
        raise ValueError('zero-input KC rows must be reviewed / removed')
    if require_real and meta.get('source_kind') != 'neuprint-reviewed-pn-kc':
        raise ValueError('NOT real PN-KC connectome; cannot label this model real')
    return m.astype(np.float32), meta

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='op', required=True)
    d = sub.add_parser('demo')
    d.add_argument('--out', default='data/demo_mask.npy')
    d.add_argument('--kc', type=int, default=128); d.add_argument('--pn', type=int, default=32)
    d.add_argument('--seed', type=int, default=11)
    w = sub.add_parser('rewire')
    w.add_argument('--input', required=True); w.add_argument('--out', required=True)
    w.add_argument('--seed', type=int, default=31)
    a = p.parse_args()
    if a.op == 'demo':
        save_mask(a.out, random_mask(a.kc, a.pn, seed=a.seed), 'DEMO_RANDOM_NOT_BIOLOGICAL',
                  {'seed': a.seed})
    else:
        m, meta = load_mask(a.input)
        rewired, info = rewire(m, a.seed)
        save_mask(a.out, rewired, 'degree-preserving-rewired',
                  {'parent_sha256': meta['sha256'], 'parent_source_kind': meta['source_kind'],
                   'seed': a.seed, **info})
