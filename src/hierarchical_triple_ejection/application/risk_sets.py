"""Target role arrays and horizon-specific complete-case risk sets."""
import numpy as np

def target_arrays(rows,feature_ids):
    by={int(r['system_id']):r for r in rows};ids=np.asarray(feature_ids,dtype=np.int64)
    status=np.array([by[int(i)]['status'] for i in ids]);role=np.array([by[int(i)]['split_role'] for i in ids])
    terminal=np.array([np.inf if by[int(i)]['status']=='stable' else float(by[int(i)].get('t_terminal_outer') if by[int(i)].get('t_terminal_outer') is not None else -np.inf) for i in ids])
    return ids,status,role,terminal

def binary_risk_set(status,role,terminal,horizon,split_role='test'):
    inplay=terminal>float(horizon);valid=(status=='stable')|(status=='ejected');idx=np.flatnonzero(inplay&valid&(role==split_role));y=(status[idx]=='ejected').astype(np.int8)
    return idx,y

def excluded_inplay(status,role,terminal,horizon,kind,split_role='test'):
    return np.flatnonzero((status==kind)&(role==split_role)&(terminal>float(horizon)))
