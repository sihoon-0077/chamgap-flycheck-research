"""Run from project root: python -m tools.fetch_connectome ...
Requires neuprint-python and a PERSONAL token. No token is included.
Selection CSV columns: bodyId, role (PN/KC), review_note.
Anatomical selection must be reviewed; this script does not infer roles.
"""
from __future__ import annotations
import argparse
import inspect
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from chamgap.connectome import save_mask, sha256


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--selection',required=True)
    p.add_argument('--dataset',default='hemibrain:v1.2.1')
    p.add_argument('--server',default='https://neuprint.janelia.org')
    p.add_argument('--out',default='data/real_pn_kc.npy')
    p.add_argument('--license-url',required=True)
    p.add_argument('--reviewed-by',required=True)
    a=p.parse_args()
    token=os.environ.get('NEUPRINT_TOKEN')
    if not token:
        raise SystemExit('set NEUPRINT_TOKEN locally; never commit it')
    from neuprint import Client, fetch_adjacencies
    selected=pd.read_csv(a.selection,dtype={'bodyId':'int64','role':str})
    required={'bodyId','role','review_note'}
    if not required.issubset(selected.columns) or selected['review_note'].isna().any():
        raise SystemExit('selection needs bodyId,role,review_note without empty review notes')
    if not selected['role'].isin(['PN','KC']).all() or selected['bodyId'].duplicated().any():
        raise SystemExit('role must be PN/KC and bodyId unique')
    pn=sorted(selected.loc[selected.role=='PN','bodyId'].tolist())
    kc=sorted(selected.loc[selected.role=='KC','bodyId'].tolist())
    if not pn or not kc:
        raise SystemExit('both reviewed PN and KC lists required')
    if len(pn)*len(kc)>10_000_000:
        raise SystemExit('select a smaller subcircuit; this starter uses a dense fixed projection')
    client=Client(a.server,dataset=a.dataset,token=token)
    kwargs={'client':client,'include_nonprimary':False}
    if 'omit_rois' in inspect.signature(fetch_adjacencies).parameters:
        kwargs={'client':client,'omit_rois':True}
    neurons, connections=fetch_adjacencies(pn,kc,**kwargs)
    pairs=connections.groupby(['bodyId_pre','bodyId_post'],as_index=False)['weight'].sum()
    if pairs.empty:
        raise SystemExit('no edges found; inspect anatomical selection and dataset')
    matrix=(pairs.pivot(index='bodyId_post',columns='bodyId_pre',values='weight')
            .reindex(index=kc,columns=pn,fill_value=0).fillna(0))
    keep_rows=matrix.sum(1)>0; keep_cols=matrix.sum(0)>0
    removed_kc=matrix.index[~keep_rows].tolist(); removed_pn=matrix.columns[~keep_cols].tolist()
    matrix=matrix.loc[keep_rows,keep_cols]
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    pairs.to_csv(out.with_suffix('.pairs.csv'),index=False)
    neurons.to_csv(out.with_suffix('.neurons.csv'),index=False)
    save_mask(out,(matrix.values>0).astype(np.float32),'neuprint-reviewed-pn-kc',{
        'dataset':a.dataset,'server':a.server,'license_url':a.license_url,
        'reviewed_by':a.reviewed_by,'selection_sha256':sha256(a.selection),
        'retrieved_utc':datetime.now(timezone.utc).isoformat(),
        'pn_ids':matrix.columns.tolist(),'kc_ids':matrix.index.tolist(),
        'removed_zero_degree_pn':removed_pn,'removed_zero_degree_kc':removed_kc,
        'weight_rule':'binary then row normalization in model; not biological conductance',
        'original_pairs_sha256':sha256(out.with_suffix('.pairs.csv'))})
    print(json.dumps({'shape':list(matrix.shape),'out':str(out)},indent=2))

if __name__=='__main__': main()
