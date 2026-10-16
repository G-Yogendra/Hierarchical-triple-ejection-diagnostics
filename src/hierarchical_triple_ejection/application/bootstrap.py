"""Deterministic stratified and paired target-test bootstrap utilities."""
import numpy as np

def stratified_indices(y,n_boot,seed):
    y=np.asarray(y,dtype=np.int8);pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
    if not len(pos) or not len(neg):raise ValueError('class-degenerate bootstrap')
    rng=np.random.default_rng(seed)
    for _ in range(int(n_boot)):
        yield np.r_[rng.choice(pos,len(pos),True),rng.choice(neg,len(neg),True)]

def paired_differences(y,score_a,score_b,metric,n_boot,seed):
    y=np.asarray(y);a=np.asarray(score_a);b=np.asarray(score_b);out=[]
    for idx in stratified_indices(y,n_boot,seed):out.append(float(metric(y[idx],a[idx])-metric(y[idx],b[idx])))
    return np.asarray(out,dtype=float)
