"""Fit independent weighing data, not sensor AI outputs.
CSV: pump_id,duration_s,delivered_g. Assumes water ~1 g/mL for initial bench work.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

p=argparse.ArgumentParser()
p.add_argument('--csv',required=True); p.add_argument('--out',default='configs/pump_calibration.json')
a=p.parse_args()
df=pd.read_csv(a.csv)
if not {'pump_id','duration_s','delivered_g'}.issubset(df.columns):
    raise SystemExit('missing required columns')
result={'status':'BENCH_FIT_REQUIRES_ACCEPTANCE_REVIEW','water_density_assumption_g_ml':1.0,'pumps':{}}
for pump,g in df.groupby('pump_id'):
    if len(g)<6 or g.duration_s.nunique()<3 or not np.isfinite(g[['duration_s','delivered_g']]).all().all():
        raise SystemExit(f'{pump}: need >=6 finite trials at >=3 durations')
    x=g.duration_s.to_numpy(float); y=g.delivered_g.to_numpy(float)
    slope,intercept=np.polyfit(x,y,1)
    if slope<=0:
        raise SystemExit(f'{pump}: nonpositive flow fit')
    pred=slope*x+intercept
    result['pumps'][str(pump)]={'ml_s':float(slope),'offset_ml':float(intercept),
      'rmse_ml':float(np.sqrt(np.mean((pred-y)**2))),
      'duration_range_s':[float(x.min()),float(x.max())], 'trials':len(g)}
Path(a.out).parent.mkdir(parents=True,exist_ok=True)
Path(a.out).write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
