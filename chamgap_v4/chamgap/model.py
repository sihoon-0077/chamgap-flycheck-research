from __future__ import annotations
import torch
from torch import nn
from .contracts import D

class ValueModel(nn.Module):
    def __init__(self, kind: str = 'mlp', adjacency=None, top_fraction: float = .05):
        super().__init__()
        if kind not in ('mlp', 'fly'):
            raise ValueError(kind)
        if not 0 < top_fraction <= 1:
            raise ValueError('top_fraction must be in (0,1]')
        self.kind = kind
        self.register_buffer('mean', torch.zeros(D))
        self.register_buffer('scale', torch.ones(D))
        if kind == 'fly':
            if adjacency is None:
                raise ValueError('fly requires a source-audited or explicitly demo matrix')
            a = torch.as_tensor(adjacency, dtype=torch.float32)
            if a.ndim != 2 or (a.sum(1) <= 0).any():
                raise ValueError('KC rows must have inputs')
            self.k = max(1, round(a.shape[0]*top_fraction))
            # Fixed topology + fixed normalized positive weights. Not synaptic physiology.
            self.register_buffer('adjacency', a/a.sum(1, keepdim=True))
            self.adapter = nn.Linear(D, a.shape[1])
            self.head = nn.Linear(a.shape[0], 4)
        else:
            self.net = nn.Sequential(nn.Linear(D, 64), nn.ReLU(), nn.Linear(64, 4))

    def forward(self, obs):
        x = ((obs-self.mean)/self.scale).clamp(-10., 10.)
        if self.kind == 'mlp':
            return self.net(x)
        pn = torch.sigmoid(self.adapter(x))
        kc = pn @ self.adjacency.T
        values, indices = torch.topk(kc, self.k, dim=1)
        sparse = torch.zeros_like(kc).scatter(1, indices, values)
        # Gradients flow through selected VALUES, not discrete rank selection.
        # Do not replace values by pure 0/1 and expect adapter gradients.
        return self.head(sparse)


def choose_action(model, obs, mask):
    with torch.inference_mode():
        device = next(model.parameters()).device
        x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        scores = model(x)[0]
        available = torch.as_tensor(mask, dtype=torch.bool, device=device)
        if not available.any() or not torch.isfinite(scores).all():
            return 3, [0., 0., 0., 0.]
        a = int(scores.masked_fill(~available, -torch.inf).argmax().item())
        return a, scores.cpu().tolist()
