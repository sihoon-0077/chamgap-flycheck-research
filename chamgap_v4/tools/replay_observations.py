"""Replay PRE-action observable feature rows only; cannot invent water outcomes.
JSONL row: {"observation": [...], "timestamp": "...", "episode_id": "..."}
"""
import argparse
import json
import numpy as np
from chamgap.contracts import action_mask,check_observation,Action
from chamgap.train import load_model
from chamgap.model import choose_action

p=argparse.ArgumentParser()
p.add_argument('--jsonl',required=True); p.add_argument('--checkpoint',required=True)
a=p.parse_args(); model=load_model(a.checkpoint)
with open(a.jsonl,encoding='utf-8') as f:
    for number,line in enumerate(f,1):
        if not line.strip(): continue
        row=json.loads(line)
        obs=check_observation(np.asarray(row['observation'],dtype=np.float32))
        action,scores=choose_action(model,obs,action_mask(obs))
        print(json.dumps({'line':number,'mode':'REPLAY_RECOMMENDATION_ONLY',
                          'episode_id':row.get('episode_id'),'action':Action(action).name,'scores':scores}))
