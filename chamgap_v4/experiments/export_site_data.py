"""Export the data files the static /results page reads (design doc 41.4).

  <out>/results.json        per-policy summaries. Every policy entry keeps the field names
                            chamgap.evaluate writes (domain, policy, episodes, seed_start,
                            summary), so the page needs no conversion code.
  <out>/learning_curve.csv  the 5.4 learning curve, one row per (model, n_episodes, seed, scope)

Both files carry generated_at, code_sha256 and domain. The HTML never changes: re-run this
script and replace the two files to update the page.

code_sha256 = SHA-256 of the lines "<file sha256>  <path>\\n" sorted by path, i.e.
  sha256sum -t <files> | LC_ALL=C sort -k2 | sha256sum
(-t keeps the two-space text format on Windows; LC_ALL=C sorts bytewise like Python.)
The per-file hashes are stored next to it so anyone can recompute it.

Run from chamgap_v4/:
  python -m experiments.export_site_data --out ../site/results/data \\
      --master "../CHAMGAP_FlyCheck_FULL_SYSTEM_MASTER_v4.1 (1).md"
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from . import analyze_learning_curve as lc

EVALS = ('runs/fixed_eval.json', 'runs/mlp_eval.json', 'runs/fly_eval.json')
LABELS = {'fixed': '고정 정책 P0', 'mlp': 'MLP', 'fly': 'fly형'}
SUMMARY_KEYS = {'auto_correct', 'auto_wrong', 'human', 'timeout', 'initial_resolved', 'checks',
                'elapsed_s', 'commanded_water_ml', 'return', 'automatic_coverage',
                'conditional_auto_accuracy', 'auto_wrong_bootstrap95'}
THIS = Path('experiments/export_site_data.py')
ANALYZER = Path('experiments/analyze_learning_curve.py')
HASH_METHOD = ('sha256 of "<file sha256>  <path>\\n" lines sorted bytewise by path; '
               'shell: sha256sum -t <code_files> | LC_ALL=C sort -k2 | sha256sum')


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_sha256(files):
    return hashlib.sha256(''.join(f'{h}  {p}\n' for p, h in sorted(files.items())).encode()).hexdigest()


def current_hashes(paths):
    return {Path(p).as_posix(): file_sha256(p) for p in paths}


def iso(ts):
    return datetime.fromtimestamp(ts).astimezone().isoformat(timespec='seconds')


def appendix_check(master, files):
    """True when every chamgap file still has the SHA-256 declared in the master document."""
    if master is None:
        return None
    text = Path(master).read_text(encoding='utf-8')
    declared = dict(re.findall(r'CHAMGAP_FILE_BEGIN path=(\S+) sha256=([0-9a-f]{64})', text))
    return all(declared.get(p) == h for p, h in files.items() if p.startswith('chamgap/'))


def truth_unknown(detail):
    rows = [r for r in detail if r['expected'] == 'UNKNOWN']
    n = len(rows)
    count = lambda k: sum(bool(r[k]) for r in rows)
    out = {'n_truth_unknown': n, 'handed_to_human': count('human'),
           'auto_answered': count('auto_wrong'), 'timeout': count('timeout')}
    for k, name in (('handed_to_human', 'handoff_rate'), ('auto_answered', 'auto_answer_rate'),
                    ('timeout', 'timeout_rate')):
        out[name] = out[k] / n if n else None
    return out


def policy_entry(eval_path):
    payload = json.loads(Path(eval_path).read_text(encoding='utf-8'))
    summary = payload['summary']
    if set(summary) != SUMMARY_KEYS:
        raise ValueError(f'{eval_path}: summary fields differ from chamgap.evaluate output')
    if not np.isclose(sum(summary[k] for k in ('auto_correct', 'auto_wrong', 'human', 'timeout')), 1):
        raise ValueError(f'{eval_path}: outcome rates do not sum to 1')
    entry = {k: payload[k] for k in ('domain', 'policy', 'episodes', 'seed_start', 'summary')}
    entry['truth_unknown'] = truth_unknown(payload['episodes_detail'])
    entry['source'] = {'file': Path(eval_path).as_posix(), 'sha256': file_sha256(eval_path),
                       'evaluated_at': iso(Path(eval_path).stat().st_mtime)}
    entry['model'], entry['connectome'] = None, None
    kind = 'fixed'
    if payload['policy'] != 'fixed':
        ckpt = Path(payload['policy'])
        meta = json.loads(ckpt.with_suffix('.json').read_text(encoding='utf-8'))
        if file_sha256(ckpt) != meta['checkpoint_sha256']:
            raise ValueError(f'{ckpt} changed since training; retrain/re-evaluate before export')
        if ckpt.stat().st_mtime > Path(eval_path).stat().st_mtime:
            raise ValueError(f'{eval_path} is older than {ckpt}; re-evaluate before export')
        kind = meta['kind']
        entry['model'] = {'kind': kind, 'trainable_parameters': meta['trainable_parameters'],
                          'epochs': meta['epochs'], 'seed': meta['seed'],
                          'checkpoint_sha256': meta['checkpoint_sha256'],
                          'dataset_sha256': meta['dataset_sha256']}
        entry['connectome'] = (meta['projection'] or {}).get('source_kind')
    label = LABELS.get(kind, payload['policy'])
    if kind == 'fly':
        label += ' (난수 연결)' if entry['connectome'] == lc.CONNECTOME else f" ({entry['connectome']})"
    entry['label'] = label
    return entry


def export_results(out, master, generated_at, curve_provenance):
    policies = [policy_entry(p) for p in EVALS]
    domains = sorted({p['domain'] for p in policies})
    files = current_hashes(sorted(Path('chamgap').glob('*.py')) + [THIS])
    doc = {'schema': 'chamgap.results/v1', 'generated_at': generated_at,
           'domain': domains[0] if len(domains) == 1 else domains,
           'code_sha256': tree_sha256(files), 'code_hash_method': HASH_METHOD,
           'code_files': files, 'code_matches_appendix_v4_1': appendix_check(master, files),
           'policies': policies,
           # Written together with learning_curve.csv so its code_sha256 can be recomputed.
           'learning_curve_csv': curve_provenance}
    (out / 'results.json').write_text(json.dumps(doc, ensure_ascii=False, indent=2) + '\n',
                                      encoding='utf-8')
    return doc


def export_learning_curve(results, out, master, generated_at):
    manifest = json.loads((results / 'learning_curve_manifest.json').read_text(encoding='utf-8'))
    recorded = manifest.get('code_sha256')
    if not recorded:
        raise ValueError('manifest has no code_sha256; re-run experiments.learning_curve')
    changed = [p for p, h in recorded.items() if not Path(p).exists() or file_sha256(p) != h]
    if changed:
        raise ValueError(f'code changed since the learning-curve run: {changed}; re-run it first')
    ep, runs = lc.load(results)
    scopes = {**lc.SCOPES, 'truth_unknown_only': lambda e: e[e['expected'] == 'UNKNOWN']}
    curve = lc.aggregate(ep, runs, scopes)
    files = {**recorded, **current_hashes([ANALYZER, THIS])}
    curve['run_started_at'] = iso(manifest['created_unix'])
    curve['generated_at'] = generated_at
    curve['code_sha256'] = tree_sha256(files)
    curve.to_csv(out / 'learning_curve.csv', index=False, lineterminator='\n')
    provenance = {'domain': lc.DOMAIN, 'connectome': lc.CONNECTOME, 'generated_at': generated_at,
                  'run_started_at': curve['run_started_at'].iloc[0],
                  'code_sha256': tree_sha256(files), 'code_files': files,
                  'code_matches_appendix_v4_1': appendix_check(master, files),
                  'test_seeds': manifest['design']['test_seeds']}
    return curve, provenance


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--results', default='results', type=Path)
    p.add_argument('--master', type=Path, help='master .md; verifies chamgap/ against its hashes')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().astimezone().isoformat(timespec='seconds')
    curve, prov = export_learning_curve(a.results, a.out, a.master, generated_at)
    doc = export_results(a.out, a.master, generated_at, prov)
    print(f"results.json: {len(doc['policies'])} policies, domain={doc['domain']}, "
          f"code_sha256={doc['code_sha256'][:12]}, matches appendix={doc['code_matches_appendix_v4_1']}")
    print(f"learning_curve.csv: {len(curve)} rows, scopes={sorted(curve['scope'].unique())}, "
          f"domain={sorted(curve['domain'].unique())}, code_sha256={prov['code_sha256'][:12]} "
          f"({len(prov['code_files'])} files), matches appendix={prov['code_matches_appendix_v4_1']}")
    print(f'generated_at={generated_at} -> {a.out}')


if __name__ == '__main__':
    main()
