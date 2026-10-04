"""Optional local inference API. READ-ONLY / DRY RUN: no pump commands.
Run: CHAMGAP_MODEL=runs/fly.pt uvicorn backend.api:app --host 127.0.0.1
"""
import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chamgap.contracts import D, Action, action_mask, check_observation
from chamgap.model import choose_action
from chamgap.train import load_model
from chamgap.policies import fixed_policy

app=FastAPI(title='CHAMGAP local inference — no hardware control')
model_path=os.getenv('CHAMGAP_MODEL')
model=load_model(model_path) if model_path else None

class ObservationRequest(BaseModel):
    observation: list[float]
    mode: str='simulation'

@app.get('/health')
def health():
    return {'ok':True,'feature_count':D,'model':model_path or 'fixed', 'hardware_enabled':False}

@app.post('/policy/next-check')
def next_check(req: ObservationRequest):
    if req.mode not in ('simulation','replay','shadow'):
        raise HTTPException(400,'only simulation/replay/shadow modes in this starter')
    try:
        obs=check_observation(np.array(req.observation,dtype=np.float32))
    except ValueError as e:
        raise HTTPException(400,str(e))
    mask=action_mask(obs)
    action,scores=(choose_action(model,obs,mask) if model else (fixed_policy(obs),None))
    return {'action':Action(action).name,'scores':scores,
            'allowed_mask':mask.tolist(),'hardware_enabled':False}
