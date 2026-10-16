#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,pickle,platform,sys,warnings
from pathlib import Path
import numpy as np

def score_hash(x):return hashlib.sha256(np.asarray(x,dtype=np.float64).tobytes()).hexdigest()
def vyn_inputs(ic):
 m0,m1,m2=ic[:,0],ic[:,1],ic[:,2]
 qin=np.minimum(m0,m1)/np.maximum(m0,m1);qout=m2/(m0+m1);alpha=ic[:,3]/ic[:,9]
 ein=ic[:,4];eout=ic[:,10];imut=ic[:,11]
 return np.c_[qin,qout,alpha,ein,eout,imut/np.pi]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--transfer-root',required=True);ap.add_argument('--target',action='append',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
 import sklearn,scipy
 if sklearn.__version__!='1.2.2':raise RuntimeError(f'legacy sklearn mismatch {sklearn.__version__}')
 root=Path(a.transfer_root)
 with warnings.catch_warnings():
  warnings.simplefilter('error')
  with open(root/'baseline_models/mlp_model_trip_v1.2.2.pkl','rb') as f:v22=pickle.load(f)
  with open(root/'baseline_models/mlp_model_trip_ghost_v1.2.2.pkl','rb') as f:v23=pickle.load(f)
 arrays={};manifest={'schema':'legacy-baseline-predictions-1.0','environment':{'python':sys.version.split()[0],'platform':platform.platform(),'numpy':np.__version__,'scipy':scipy.__version__,'scikit_learn':sklearn.__version__},'targets':{}}
 for item in a.target:
  name,path=item.split('=',1)
  with np.load(Path(path)/'features.npz',allow_pickle=False) as z:ids=z['system_id'].astype(np.int64);X=vyn_inputs(z['ic_raw'])
  p22=np.asarray(v22.predict_proba(X)[:,1],dtype=np.float64);p23=np.asarray(v23.predict_proba(X)[:,1],dtype=np.float64)
  arrays[name+'__ids']=ids;arrays[name+'__V22_MLP']=p22;arrays[name+'__V23_MLP']=p23
  manifest['targets'][name]={'n':len(ids),'id_hash':score_hash(ids),'V22_MLP_score_hash':score_hash(p22),'V23_MLP_score_hash':score_hash(p23)}
 out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,**arrays)
 mp=out.with_suffix('.manifest.json');mp.write_text(json.dumps(manifest,indent=2)+'\n');print(out,flush=True)
if __name__=='__main__':main()
