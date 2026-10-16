"""Seven predeclared failure/competing-outcome stress scenarios."""
import numpy as np
SCENARIOS=('primary','numerical_as_control_stress','numerical_as_event_stress','collision_as_composite_event','collision_as_control_stress','all_scoreable_exclusions_as_control_stress','all_scoreable_exclusions_as_event_stress')

def build_scenarios(primary_idx,primary_y,numerical_idx,collision_idx):
    p=np.asarray(primary_idx,int);y=np.asarray(primary_y,np.int8);n=np.asarray(numerical_idx,int);c=np.asarray(collision_idx,int)
    return {
      'primary':(p,y),
      'numerical_as_control_stress':(np.r_[p,n],np.r_[y,np.zeros(len(n),np.int8)]),
      'numerical_as_event_stress':(np.r_[p,n],np.r_[y,np.ones(len(n),np.int8)]),
      'collision_as_composite_event':(np.r_[p,c],np.r_[y,np.ones(len(c),np.int8)]),
      'collision_as_control_stress':(np.r_[p,c],np.r_[y,np.zeros(len(c),np.int8)]),
      'all_scoreable_exclusions_as_control_stress':(np.r_[p,n,c],np.r_[y,np.zeros(len(n)+len(c),np.int8)]),
      'all_scoreable_exclusions_as_event_stress':(np.r_[p,n,c],np.r_[y,np.ones(len(n)+len(c),np.int8)]),
    }

def envelope(rows,keys=('roc_auc','pr_auc')):
    return {f'{k}_{side}':fn(float(r[k]) for r in rows) for k in keys for side,fn in [('min',min),('max',max)]}
