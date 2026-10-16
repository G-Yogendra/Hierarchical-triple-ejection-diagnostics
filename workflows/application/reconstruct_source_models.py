#!/usr/bin/env python3
"""Build explicitly named A1 deterministic source-model reconstructions.

These are new refits under the frozen protocol, not recovered original bytes.
The script fails closed unless the frozen source array, release manifest, and
pinned reconstruction environment match their declared identities.
"""
from __future__ import annotations
import argparse,hashlib,json,platform,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

EXPECTED_SOURCE_SHA="187835e0b45964f8ef0c295952eb49569ce5a5f9c9ec779e490d35f7aa352687"
EXPECTED_SOURCE_BYTES=1817502996
EXPECTED_RELEASE_MANIFEST_SHA="c57545553af5805b7f7714e44c35d61ee8f0a7df05e7945c8ce10d4f5f0a9fd2"
REQUIRED_VERSIONS={"python":"3.12","numpy":"2.5.3","scikit_learn":"1.9.1","xgboost":"3.4.1"}
SEED=26001;FIT_CAP=200000;FARS=(.001,.01,.05);FRACTION_INDEX=(0,2,4,6,8);ACTIVE=np.array([i for i in range(79) if i not in (72,73)])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def idhash(x):return hashlib.sha256(np.asarray(x,dtype=np.int64).tobytes()).hexdigest()
def canonical_hash(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def fail(msg):raise SystemExit("SOURCE_ARRAY_NOT_MOUNTED_OR_HASH_MISMATCH: "+msg)
def model(seed,n_jobs):return XGBClassifier(n_estimators=300,max_depth=3,learning_rate=.05,subsample=.8,colsample_bytree=.8,reg_lambda=2,objective='binary:logistic',eval_metric='logloss',random_state=seed,n_jobs=n_jobs,tree_method='hist')
def ic_tri(ic):return np.column_stack([ic[:,0],ic[:,1],ic[:,2],ic[:,3],ic[:,4],ic[:,9],ic[:,10],ic[:,11],ic[:,12]/(ic[:,0]+ic[:,1]+1e-30),ic[:,9]/(ic[:,3]+1e-30),ic[:,9]*(1-ic[:,10])/(ic[:,3]*(1+ic[:,4])+1e-30),np.cos(ic[:,11]),np.sin(ic[:,11]),ic[:,15],ic[:,16]]).astype(np.float32)
def split(y):
 valid=np.where(y>=0)[0];tr,hold=train_test_split(valid,test_size=.4,random_state=SEED,stratify=y[valid]);ca,te=train_test_split(hold,test_size=.5,random_state=SEED+1,stratify=y[hold]);return tr,ca,te
def threshold_payload(y,s):
 neg=s[y==0];return {"rule":"higher empirical quantile of source calibration negatives; alert score >= threshold","rows":[{"target_far":a,"threshold":float(np.quantile(neg,1-a,method='higher'))} for a in FARS]}
def environment():
 import sklearn,xgboost
 return {"python":sys.version.split()[0],"platform":platform.platform(),"numpy":np.__version__,"scikit_learn":sklearn.__version__,"xgboost":xgboost.__version__}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--merged-triples',required=True);ap.add_argument('--source-manifest',required=True);ap.add_argument('--environment-lock',required=True);ap.add_argument('--code-archive-sha256',required=True);ap.add_argument('--out',required=True);ap.add_argument('--xgb-threads',type=int,default=4);a=ap.parse_args();src=Path(a.merged_triples);rel=Path(a.source_manifest);lock=Path(a.environment_lock);out=Path(a.out)
 if not src.exists():fail(f"missing {src}")
 if src.stat().st_size!=EXPECTED_SOURCE_BYTES or sha(src)!=EXPECTED_SOURCE_SHA:fail(f"source bytes/hash mismatch: {src.stat().st_size} {sha(src)}")
 if not rel.exists() or sha(rel)!=EXPECTED_RELEASE_MANIFEST_SHA:fail("release manifest mismatch")
 if not lock.exists():fail("environment lock missing")
 env=environment()
 if not env['python'].startswith(REQUIRED_VERSIONS['python']+'.') or any(env[k]!=v for k,v in REQUIRED_VERSIONS.items() if k!='python'):fail(f"environment mismatch {env} != {REQUIRED_VERSIONS}")
 out.mkdir(parents=True,exist_ok=True)
 with np.load(src,allow_pickle=False) as z:
  required={'ids','statuses','t_event','t_max','ic','feats'}
  if not required<=set(z.files):fail(f"array keys missing: {sorted(required-set(z.files))}")
  d={k:z[k] for k in required};d['fractions']=z['fractions'] if 'fractions' in z.files else np.array([.05,.1,.15,.2,.3,.4,.5,.6,.75,.9,1],np.float32)
 if d['ids'].shape!=(500000,) or d['ic'].shape[0]!=500000 or d['feats'].shape!=(500000,11,79):fail(f"unexpected shapes ids={d['ids'].shape} ic={d['ic'].shape} feats={d['feats'].shape}")
 y=np.full(len(d['statuses']),-1,np.int8);y[d['statuses']=='stable']=0;y[d['statuses']=='ejected']=1;tr0,ca0,te0=split(y)
 if set(tr0)&set(ca0) or set(tr0)&set(te0) or set(ca0)&set(te0):fail("split overlap")
 event=np.where(np.isnan(d['t_event']),np.inf,d['t_event']);ic=ic_tri(d['ic']);mask_payload={"operational_ic":"15-column A1 ic_tri v1","active_history_indices":ACTIVE.tolist(),"disabled_indices":[72,73]};mask_hash=canonical_hash(mask_payload);preprocess_hash=canonical_hash({"mask":mask_payload,"nan_policy":"frozen arrays; no target scaling","combined_order":"IC then active history"});script_hash=sha(Path(__file__))
 manifest={"schema":"a1-deterministic-source-model-reconstruction-1.0","reconstruction_id":"reconstructed_source_v1","claim":"deterministic reconstruction from frozen arrays; not original ephemeral model bytes","source_array_path":str(src),"source_array_sha256":EXPECTED_SOURCE_SHA,"source_array_bytes":EXPECTED_SOURCE_BYTES,"source_manifest_sha256":EXPECTED_RELEASE_MANIFEST_SHA,"split_seed":SEED,"split_id_hashes":{"train":idhash(d['ids'][tr0]),"calibration":idhash(d['ids'][ca0]),"test":idhash(d['ids'][te0])},"active_feature_mask_hash":mask_hash,"preprocessing_hash":preprocess_hash,"environment":env,"environment_lock_path":str(lock),"environment_lock_hash":sha(lock),"script_path":str(Path(__file__)),"script_sha256":script_hash,"script_commit_or_archive_hash":a.code_archive_sha256,"target_files_read":[],"execution":{"xgb_threads":a.xgb_threads},"models":{}}
 for fi in FRACTION_INDEX:
  f=float(d['fractions'][fi]);h=f*d['t_max'];inplay=np.where((y>=0)&(event>h))[0];tr=tr0[event[tr0]>h[tr0]];ca=ca0[event[ca0]>h[ca0]];te=te0[event[te0]>h[te0]];cap_seed=30000+fi
  if len(tr)>FIT_CAP:tr=np.random.default_rng(cap_seed).choice(tr,FIT_CAP,False)
  hist=d['feats'][:,fi][:,ACTIVE].astype(np.float32);designs=[('IC_only',ic),('history',hist),('combined',np.hstack([ic,hist]).astype(np.float32))];block={"horizon_index":fi,"horizon_fraction":f,"in_play_id_hash":idhash(d['ids'][inplay]),"train_id_hash":idhash(d['ids'][tr]),"calibration_id_hash":idhash(d['ids'][ca]),"test_id_hash":idhash(d['ids'][te]),"cap_seed":cap_seed,"cap_size":len(tr),"arms":{}}
  for arm,(name,X) in enumerate(designs):
   rs=31000+10*fi+arm;m=model(rs,a.xgb_threads);m.fit(X[tr],y[tr]);stem=f'reconstructed_source_v1_{name}_j{fi}_f{f:.2f}';mp=out/f'{stem}.json';m.save_model(mp);sc=m.predict_proba(X[ca])[:,1];st=m.predict_proba(X[te])[:,1];tp=out/f'{stem}_thresholds.json';tp.write_text(json.dumps(threshold_payload(y[ca],sc),indent=2)+'\n');block['arms'][name]={"arm_index":arm,"random_state":rs,"model_file":mp.name,"model_file_sha256":sha(mp),"threshold_file":tp.name,"threshold_file_sha256":sha(tp),"threshold_rule":"higher empirical quantile; calibration negatives only","target_far":list(FARS),"source_test_auc":float(roc_auc_score(y[te],st)),"source_test_score_hash":hashlib.sha256(np.asarray(st,dtype=np.float64).tobytes()).hexdigest()}
  manifest['models'][str(f)]=block
 mp=out/'SOURCE_MODEL_MANIFEST.json';mp.write_text(json.dumps(manifest,indent=2)+'\n');(out/'SOURCE_MODEL_MANIFEST.sha256').write_text(f'{sha(mp)}  SOURCE_MODEL_MANIFEST.json\n');print(mp)
if __name__=='__main__':main()
