from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .connectome import load_mask, sha256
from .model import ValueModel
from .contracts import FEATURES


def train(data, out, kind='mlp', matrix=None, require_real=False, epochs=40, seed=7, device='cpu'):
    if epochs < 1:
        raise ValueError('epochs must be positive')
    torch.set_num_threads(1)
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: use --device cpu or install the appropriate PyTorch build')
    with np.load(data, allow_pickle=False) as d:
        x, y = d['x'].copy(), d['y'].copy()
        known, groups = d['target_mask'].copy(), d['group'].copy()
    unique = np.unique(groups); rng.shuffle(unique)
    if len(unique) < 8:
        raise ValueError('need >= 8 independent episode groups for smoke-test split')
    ntrain = max(1, int(len(unique)*.7)); nval = max(1, int(len(unique)*.15))
    train_groups = unique[:ntrain]; val_groups = unique[ntrain:ntrain+nval]
    test_groups = unique[ntrain+nval:]
    tr = np.isin(groups, train_groups); va = np.isin(groups, val_groups); te = np.isin(groups, test_groups)
    adjacency, provenance = (load_mask(matrix, require_real) if kind == 'fly' else (None, {}))
    model = ValueModel(kind, adjacency).to(device)
    model.mean.copy_(torch.tensor(x[tr].mean(0), device=device))
    model.scale.copy_(torch.tensor(np.maximum(x[tr].std(0), .05), device=device))
    tx = torch.tensor(x, device=device); ty = torch.tensor(y, device=device)
    tm = torch.tensor(known, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=.003, weight_decay=1e-4)
    indices = np.flatnonzero(tr)
    best, best_state = float('inf'), None
    history = []
    def loss_for(ids):
        pred = model(tx[ids])
        errors = torch.nn.functional.smooth_l1_loss(pred, ty[ids], reduction='none')
        return errors[tm[ids]].mean()
    for epoch in range(epochs):
        model.train(); rng.shuffle(indices)
        for start in range(0,len(indices),64):
            ids = indices[start:start+64]
            loss = loss_for(ids)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            opt.step()
        model.eval()
        with torch.inference_mode():
            val = float(loss_for(np.flatnonzero(va)))
        history.append({'epoch':epoch+1, 'validation_huber':val})
        if val < best:
            best = val; best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    with torch.inference_mode():
        test_loss = float(loss_for(np.flatnonzero(te)))
    p = Path(out); p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'kind':kind, 'state_dict':best_state, 'top_fraction':.05,
                'matrix':None if adjacency is None else torch.tensor(adjacency),
                'features':list(FEATURES)}, p)
    meta = {'domain':'SIM_ONLY_UNCALIBRATED', 'kind':kind, 'seed':seed, 'epochs':epochs,
            'dataset_sha256':sha256(data), 'checkpoint_sha256':sha256(p),
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'validation_huber':best, 'test_huber':test_loss,
            'train_groups':train_groups.tolist(), 'val_groups':val_groups.tolist(),
            'test_groups':test_groups.tolist(), 'projection':provenance, 'history':history}
    p.with_suffix('.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(json.dumps({k:meta[k] for k in ('kind','domain','trainable_parameters','validation_huber','test_huber')},indent=2))


def load_model(path):
    # Load only files created by this project, not untrusted uploaded checkpoints.
    ck = torch.load(path, map_location='cpu', weights_only=True)
    if ck['features'] != list(FEATURES):
        raise ValueError('feature schema mismatch')
    m = ValueModel(ck['kind'], ck['matrix'], ck['top_fraction'])
    m.load_state_dict(ck['state_dict']); m.eval()
    return m

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/sim_train.npz'); p.add_argument('--out',required=True)
    p.add_argument('--kind',choices=['mlp','fly'],default='mlp'); p.add_argument('--matrix')
    p.add_argument('--require-real',action='store_true')
    p.add_argument('--epochs',type=int,default=40); p.add_argument('--seed',type=int,default=7)
    p.add_argument('--device',default='cpu')
    a = p.parse_args(); train(a.data,a.out,a.kind,a.matrix,a.require_real,a.epochs,a.seed,a.device)
