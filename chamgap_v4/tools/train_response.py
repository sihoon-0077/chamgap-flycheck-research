"""Fit NORMAL response delta; not a fault classifier and not field-validated.
CSV columns are checked below. Sensor/period holdout remains an evaluation task.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error
import joblib

FEATURES=['soil_before','delivered_estimate_ml','elapsed_s','temperature_c','humidity_pct']
p=argparse.ArgumentParser()
p.add_argument('--csv',required=True); p.add_argument('--out',default='runs/response.joblib')
a=p.parse_args(); df=pd.read_csv(a.csv)
required=set(FEATURES+['soil_after','session_id','verified_normal'])
if not required.issubset(df.columns): raise SystemExit(f'columns required: {sorted(required)}')
df=df[df.verified_normal==1].copy()
if df.session_id.nunique()<8: raise SystemExit('need >=8 normal sessions; this is only an initial code threshold')
x=df[FEATURES].to_numpy(float); y=(df.soil_after-df.soil_before).to_numpy(float)
if not np.isfinite(x).all() or not np.isfinite(y).all(): raise SystemExit('nonfinite training data')
tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=71).split(x,y,df.session_id))
model=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=15,random_state=71)
model.fit(x[tr],y[tr]); pred=model.predict(x[te])
path=Path(a.out); path.parent.mkdir(parents=True,exist_ok=True)
joblib.dump({'model':model,'features':FEATURES,'target':'soil_after-soil_before'},path)
meta={'model':'HistGradientBoostingRegressor','status':'DEVELOPMENT_ONLY',
      'train_sessions':sorted(df.iloc[tr].session_id.unique().tolist()),
      'heldout_sessions':sorted(df.iloc[te].session_id.unique().tolist()),
      'heldout_mae_soil_index':float(mean_absolute_error(y[te],pred)),
      'notice':'Fit operating thresholds on SEPARATE validation sessions. Do not use this holdout as final field test.'}
path.with_suffix('.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
print(json.dumps(meta,indent=2))
