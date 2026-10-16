from __future__ import annotations
import numpy as np
from .a1_features import extract_window_features,apply_final_mask,FEATURE_NAMES
HORIZONS=(.05,.15,.30,.50,.75)
ACTIVE_HISTORY=np.array([i for i in range(79) if i not in (72,73)],dtype=int)

def history_raw_matrix(record,horizons=HORIZONS):
    x=np.vstack([extract_window_features(record,t_horizon=float(f)*record['t_max']) for f in horizons]).astype(np.float32)
    return apply_final_mask(x).astype(np.float32)
def history_active_matrix(raw): return np.asarray(raw,dtype=np.float32)[...,ACTIVE_HISTORY]
def ic_storage_vector(ic):
    keys=['m0','m1','m2','a_inner','e_inner','inc_inner','Omega_inner','omega_inner','M_inner','a_outer','e_outer','inc_outer','Omega_outer','omega_outer','M_outer']
    return np.array([ic[k] for k in keys]+[ic['MA01_crit'],ic['MA01_margin'],0.0,0.0],dtype=np.float32)
def ic_operational_vector(ic):
    return np.array([ic['m0'],ic['m1'],ic['m2'],ic['a_inner'],ic['e_inner'],ic['a_outer'],ic['e_outer'],ic['inc_outer'],ic['m2']/(ic['m0']+ic['m1']+1e-30),ic['a_outer']/(ic['a_inner']+1e-30),ic['a_outer']*(1-ic['e_outer'])/(ic['a_inner']*(1+ic['e_inner'])+1e-30),np.cos(ic['inc_outer']),np.sin(ic['inc_outer']),ic['MA01_crit'],ic['MA01_margin']],dtype=np.float32)
def combined_matrix(ic_operational,history_active): return np.hstack([np.repeat(ic_operational[None,:],len(history_active),axis=0),history_active]).astype(np.float32)
def schema_record(): return {"stored_ic_columns":19,"operational_ic_columns":15,"raw_history_columns":len(FEATURE_NAMES),"disabled_indices":[72,73],"active_history_columns":77,"combined_columns":92,"horizons":list(HORIZONS)}
