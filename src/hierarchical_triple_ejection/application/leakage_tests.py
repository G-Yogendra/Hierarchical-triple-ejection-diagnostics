from __future__ import annotations
import numpy as np
from .a1_features import extract_window_features,apply_final_mask

def poison_future(record,horizon):
    out={k:np.array(v,copy=True) if isinstance(v,np.ndarray) else v for k,v in record.items()};post=out['t']>horizon
    for k in ['H','Omega','E_out']:out[k][post]=1e9
    for k in ['E_pair','L_pair','d_pair']:out[k][post]=1e9
    return out

def append_future(record,n=40):
    rng=np.random.default_rng(20261003);out={k:np.array(v,copy=True) if isinstance(v,np.ndarray) else v for k,v in record.items()};dt=np.median(np.diff(out['t']));t=out['t'][-1]+dt*np.arange(1,n+1);out['t']=np.concatenate([out['t'],t])
    for k in ['H','Omega','E_out']:out[k]=np.concatenate([out[k],rng.normal(size=n)*1e6])
    for k in ['E_pair','L_pair','d_pair']:
        tail=rng.normal(size=(n,)+out[k].shape[1:])*1e6;out[k]=np.concatenate([out[k],tail],axis=0)
    return out

def future_invariance(record,horizon):
    base=apply_final_mask(extract_window_features(record,t_horizon=horizon));p=apply_final_mask(extract_window_features(poison_future(record,horizon),t_horizon=horizon));a=apply_final_mask(extract_window_features(append_future(record),t_horizon=horizon))
    return {"poison_equal":bool(np.array_equal(base,p)),"append_equal":bool(np.array_equal(base,a)),"disabled_zero":bool(np.all(base[[72,73]]==0)),"pass":bool(np.array_equal(base,p) and np.array_equal(base,a) and np.all(base[[72,73]]==0))}
