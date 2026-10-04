"""Build the real PN->KC mask (connectome_v1) from neuPrint hemibrain:v1.2.1 by a fixed rule.

Rule, fixed before any model was trained on this matrix:
  PN   neurons whose type contains 'PN' and that have outputs in CA(R), excluding
       types starting with 'M_' (multiglomerular) and glomerulus labels (type before the
       first '_') starting with 'VP' (thermo/hygrosensory, incl. mixed labels like VP1d+VP4).
       Sister PNs are pooled into one column per glomerulus.
  KC   olfactory KC types: KCg-m, KCab-m, KCab-c, KCab-s, KCa'b'-ap1, KCa'b'-ap2, KCa'b'-m.
       KCab-p and KCg-d (visual input) are not candidates.
  edge (KC, glomerulus) = 1 when the summed synapse count inside CA(R) from that
       glomerulus' PNs is >= 3. Rows/columns without any edge are removed; their IDs are kept.

The selection is rule-based and still needs the team's review (design doc 12.4), so the
mask is saved with source_kind 'neuprint-hemibrain-rule-selected-pn-kc', NOT the appendix's
'neuprint-reviewed-pn-kc' label that --require-real accepts.

The token is read from NEUPRINT_TOKEN only; it is never printed or written to disk.
Run from chamgap_v4/:  python -m experiments.build_real_connectome
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from chamgap.connectome import load_mask, save_mask, sha256

DATASET = 'hemibrain:v1.2.1'
SERVER = 'neuprint.janelia.org'
ROI = 'CA(R)'
KC_TYPES = ('KCg-m', 'KCab-m', 'KCab-c', 'KCab-s', "KCa'b'-ap1", "KCa'b'-ap2", "KCa'b'-m")
MIN_SYNAPSES = 3
SOURCE_KIND = 'neuprint-hemibrain-rule-selected-pn-kc'
CONNECTOME_LABEL = 'HEMIBRAIN_V1.2.1_CA_R_OLF51_RULE_SELECTED_PENDING_REVIEW'
LICENSE_URL = 'https://creativecommons.org/licenses/by/4.0/'
TERMS_SOURCE = 'https://www.janelia.org/project-team/flyem/hemibrain'
CITATION = 'Scheffer et al. 2020, eLife 9:e57443, https://elifesciences.org/articles/57443'
OUT_DIR = Path('data/connectome')
STEM = 'hemibrain_v121_caR_olf51'
RULE = ('PN: type contains PN, outputs in CA(R), type not M_*, glomerulus (type before first _) '
        'not VP*; pooled per glomerulus. KC: ' + ', '.join(KC_TYPES) + '. Edge: summed CA(R) '
        f'synapses per (KC, glomerulus) >= {MIN_SYNAPSES}. Zero-degree rows/columns removed.')
PN_COLUMNS = ['bodyId', 'type', 'instance', 'status', 'cropped', 'pre', 'post']


def glomerulus(cell_type: str) -> str:
    return cell_type.split('_')[0]


def select_pns(pn: pd.DataFrame):
    glom = pn['type'].map(glomerulus)
    drop = pn['type'].str.startswith('M_') | glom.str.startswith('VP')
    kept = pn[~drop].assign(glomerulus=glom[~drop])
    return kept, pn[drop]


def build_matrix(pairs: pd.DataFrame, pns: pd.DataFrame, kcs: pd.DataFrame):
    """Binary KC x glomerulus matrix (rows = KC bodyId ascending, columns = glomerulus name)."""
    m = pairs.merge(pns[['bodyId', 'glomerulus']], left_on='bodyId_pre', right_on='bodyId')
    g = m.groupby(['bodyId_post', 'glomerulus'], as_index=False)['weight'].sum()
    g = g[g['weight'] >= MIN_SYNAPSES]
    kc_order = sorted(kcs['bodyId'].tolist())
    glom_order = sorted(pns['glomerulus'].unique().tolist())
    a = (pd.crosstab(g['bodyId_post'], g['glomerulus'])
         .reindex(index=kc_order, columns=glom_order, fill_value=0).clip(upper=1))
    rows, cols = a.sum(1) > 0, a.sum(0) > 0
    return a.loc[rows, cols], a.index[~rows].tolist(), a.columns[~cols].tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', type=Path, default=OUT_DIR)
    a = p.parse_args()
    token = os.environ.get('NEUPRINT_TOKEN')
    if not token:
        raise SystemExit('set NEUPRINT_TOKEN in this shell (load chamgap_v4/.env first); never commit it')
    from neuprint import Client, NeuronCriteria as NC, fetch_adjacencies, fetch_neurons

    client = Client(SERVER, dataset=DATASET, token=token)
    retrieved = datetime.now(timezone.utc).isoformat(timespec='seconds')
    server_version = client.fetch_version()
    pn_all, _ = fetch_neurons(NC(type='.*PN.*', regex=True, outputRois=[ROI]), client=client)
    kcs, _ = fetch_neurons(NC(type=list(KC_TYPES)), client=client)
    if set(kcs['type']) != set(KC_TYPES):
        raise RuntimeError(f'unexpected KC types returned: {sorted(set(kcs["type"]))}')
    pns, excluded = select_pns(pn_all)
    _, conn = fetch_adjacencies(pns['bodyId'].tolist(), kcs['bodyId'].tolist(), rois=[ROI], client=client)
    conn = conn[conn['roi'] == ROI]
    pairs = (conn.groupby(['bodyId_pre', 'bodyId_post'], as_index=False)['weight'].sum()
             .sort_values(['bodyId_pre', 'bodyId_post']))
    matrix, removed_kc, removed_glom = build_matrix(pairs, pns, kcs)

    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    raw = {f'{STEM}.pn_selected.csv': pns[PN_COLUMNS + ['glomerulus']].sort_values('bodyId'),
           f'{STEM}.pn_excluded.csv': excluded[PN_COLUMNS].sort_values('bodyId'),
           f'{STEM}.kc_candidates.csv': kcs[PN_COLUMNS].sort_values('bodyId'),
           f'{STEM}.pairs_caR.csv': pairs}
    for name, frame in raw.items():
        frame.to_csv(out / name, index=False)

    deg = matrix.sum(1)
    builder = Path(__file__).resolve().relative_to(Path.cwd().resolve()).as_posix()
    details = {
        'connectome_label': CONNECTOME_LABEL, 'review_status': 'PENDING_TEAM_REVIEW',
        'dataset': DATASET, 'server': SERVER, 'server_version': server_version, 'roi': ROI,
        'retrieved_utc': retrieved, 'selection_rule': RULE, 'min_synapses': MIN_SYNAPSES,
        'license_url': LICENSE_URL, 'license_terms_source': TERMS_SOURCE, 'citation': CITATION,
        'weight_rule': 'binary then row normalization in model; not biological conductance',
        'row_kc_ids': [int(x) for x in matrix.index], 'column_glomeruli': list(matrix.columns),
        'pn_ids_by_glomerulus': {g: sorted(int(x) for x in pns.loc[pns['glomerulus'] == g, 'bodyId'])
                                 for g in matrix.columns},
        'removed_zero_input_kc': [int(x) for x in removed_kc],
        'removed_zero_output_glomeruli': removed_glom,
        'excluded_pn': [{'bodyId': int(r.bodyId), 'type': r.type} for r in excluded.itertuples()],
        'counts': {'pn_candidates': len(pn_all), 'pn_selected': len(pns), 'pn_excluded': len(excluded),
                   'glomeruli_selected': int(pns['glomerulus'].nunique()),
                   'kc_candidates': len(kcs), 'kc_types': kcs['type'].value_counts().to_dict(),
                   'pair_rows': len(pairs), 'pair_synapses': int(pairs['weight'].sum()),
                   'kc_in_degree_mean': round(float(deg.mean()), 4),
                   'kc_in_degree_p5_p95': [int(deg.quantile(.05)), int(deg.quantile(.95))],
                   'density': round(float(matrix.values.mean()), 4)},
        'raw_sha256': {name: sha256(out / name) for name in raw},
        'builder': builder, 'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    path = out / f'{STEM}.npy'
    save_mask(path, matrix.values.astype(np.float32), SOURCE_KIND, details)
    m, meta = load_mask(path)   # appendix loader: hash, finiteness, no zero-input rows
    print(json.dumps({'out': str(path), 'shape_kc_glomerulus': list(m.shape), 'edges': meta['edges'],
                      'server_version': server_version, **details['counts']}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
