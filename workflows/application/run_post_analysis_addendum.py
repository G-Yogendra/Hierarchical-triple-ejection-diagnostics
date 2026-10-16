#!/usr/bin/env python3
"""Bounded post-analysis addendum. No N-body simulation and no model fitting."""
from __future__ import annotations
import argparse,csv,glob,hashlib,json,pickle,subprocess,sys,warnings
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from xgboost import XGBClassifier

FRACS=np.array([.05,.15,.30,.50,.75],float);FARS=(.001,.01,.05);BOOT=1000
ARMS=('IC_only','history','combined');BASES=('MA01_margin','V22_formula','V22_MLP','V23_MLP')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def score_hash(x):return hashlib.sha256(np.asarray(x,dtype=np.float64).tobytes()).hexdigest()
def loadj(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def find_transfer_root():
 hits=[]
 for x in glob.glob('/kaggle/input/**/results/ALL_TARGET_RESULTS.json',recursive=True):
  p=Path(x);r=p.parent.parent
  if (r/'RECONSTRUCTION_FREEZE.json').exists() and (r/'source_models_run1/SOURCE_MODEL_MANIFEST.json').exists():hits.append(r)
 if not hits:raise SystemExit('TRANSFER_RESULTS_DATASET_NOT_MOUNTED')
 # Multiple byte-identical mounted copies are harmless; differing results fail.
 groups={}
 for r in hits:groups.setdefault(sha(r/'results/ALL_TARGET_RESULTS.json'),[]).append(r)
 if len(groups)!=1:raise SystemExit('AMBIGUOUS_TRANSFER_RESULTS_DIFFERENT_CONTENT')
 return min(hits,key=lambda p:(len(p.parts),len(str(p)),str(p)))
def file_identity(d):
 return {n:{'bytes':(d/n).stat().st_size,'sha256':sha(d/n)} for n in ('features.npz','systems.jsonl','population_config.json')}
def find_targets(root):
 expected=loadj(root/'TARGET_DISCOVERY.json')['populations'];allowed=set(expected);cand={}
 for x in glob.glob('/kaggle/input/**/features.npz',recursive=True):
  d=Path(x).parent;cfg=d/'population_config.json';rows=d/'systems.jsonl'
  if not cfg.exists() or not rows.exists():continue
  name=loadj(cfg).get('name')
  if name in allowed and file_identity(d)==expected[name]['content_identity']:cand.setdefault(name,[]).append(d.resolve())
 missing=allowed-set(cand)
 if missing:raise SystemExit('MISSING_HASH_MATCHED_TARGETS: '+repr(sorted(missing)))
 return {k:min(set(v),key=lambda p:(len(p.parts),len(str(p)),str(p))) for k,v in cand.items()}
def recon_key(recon,f,tol=1e-6):
 m=[k for k in recon['models'] if abs(float(k)-float(f))<=tol]
 if len(m)!=1:raise RuntimeError(f'reconstruction fraction resolution failed {f} {m}')
 return m[0]
def load_xgb(p):
 m=XGBClassifier(n_jobs=4);m.load_model(p);m.set_params(n_jobs=4);return m
def metrics(y,s):
 y=np.asarray(y,int);s=np.asarray(s,float)
 return {'n':int(len(y)),'events':int(y.sum()),'controls':int((y==0).sum()),'prevalence':float(y.mean()),'roc_auc':float(roc_auc_score(y,s)),'pr_auc':float(average_precision_score(y,s))}
def op(y,s,t):
 y=np.asarray(y,int);s=np.asarray(s,float);p=s>=t;n=y==0;e=y==1
 return {'threshold':float(t),'achieved_far':float(p[n].mean()) if n.any() else None,'coverage':float(p[e].mean()) if e.any() else None,'precision':float(y[p].mean()) if p.any() else None,'alerts':int(p.sum()),'event_alerts':int((p&e).sum()),'control_alerts':int((p&n).sum())}
def stratified_draws(y,nboot,seed):
 y=np.asarray(y,int);pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0);rng=np.random.default_rng(seed)
 if not len(pos) or not len(neg):raise RuntimeError('class-degenerate bootstrap')
 for _ in range(nboot):yield np.r_[rng.choice(pos,len(pos),True),rng.choice(neg,len(neg),True)]
def interval(y,s,seed):
 point=metrics(y,s);roc=[];pr=[]
 for ii in stratified_draws(y,BOOT,seed):roc.append(roc_auc_score(y[ii],s[ii]));pr.append(average_precision_score(y[ii],s[ii]))
 return {**point,'roc_lo':float(np.percentile(roc,2.5)),'roc_hi':float(np.percentile(roc,97.5)),'pr_lo':float(np.percentile(pr,2.5)),'pr_hi':float(np.percentile(pr,97.5)),'n_boot':BOOT,'bootstrap':'stratified target test system IDs'}
def paired(y,a,b,seed):
 dr=[];dp=[]
 for ii in stratified_draws(y,BOOT,seed):dr.append(roc_auc_score(y[ii],a[ii])-roc_auc_score(y[ii],b[ii]));dp.append(average_precision_score(y[ii],a[ii])-average_precision_score(y[ii],b[ii]))
 return {'roc_point':float(roc_auc_score(y,a)-roc_auc_score(y,b)),'roc_lo':float(np.percentile(dr,2.5)),'roc_hi':float(np.percentile(dr,97.5)),'pr_point':float(average_precision_score(y,a)-average_precision_score(y,b)),'pr_lo':float(np.percentile(dp,2.5)),'pr_hi':float(np.percentile(dp,97.5)),'n_boot':BOOT,'unit':'paired stratified target test system IDs'}
def vyn_scores(ic,v22,v23):
 m0,m1,m2=ic[:,0],ic[:,1],ic[:,2];qin=np.minimum(m0,m1)/np.maximum(m0,m1);qout=m2/(m0+m1);alpha=ic[:,3]/ic[:,9];ein=ic[:,4];eout=ic[:,10];imut=ic[:,11]
 lk=1-(5/3)*np.cos(imut)**2;et=np.where(lk>=0,np.maximum(ein,.5*lk),ein);fac=.125*(1-.2*et+eout)*(np.cos(imut)-1)+1;Y=(1-eout)/((1+et)*alpha);Yc=2.4*(1+qout)**.4*(1+et)**(-.4)*(1-eout)**(-.2)*fac;X=np.c_[qin,qout,alpha,ein,eout,imut/np.pi]
 with warnings.catch_warnings():warnings.simplefilter('ignore');p22=v22.predict_proba(X)[:,1];p23=v23.predict_proba(X)[:,1]
 return {'MA01_margin':-ic[:,16],'V22_formula':np.log(np.maximum(Yc,1e-30)/np.maximum(Y,1e-30)),'V22_MLP':np.asarray(p22,float),'V23_MLP':np.asarray(p23,float)}
def write_csv(path,rows):
 if not rows:return
 keys=[]
 for r in rows:
  for k in r:
   if k not in keys:keys.append(k)
 with open(path,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--legacy-python',required=True);ap.add_argument('--legacy-script',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 import sklearn,xgboost
 if np.__version__!='2.5.3' or sklearn.__version__!='1.9.1' or xgboost.__version__!='3.4.1':raise RuntimeError(f'main environment mismatch numpy={np.__version__} sklearn={sklearn.__version__} xgb={xgboost.__version__}')
 root=find_transfer_root();targets=find_targets(root);published=loadj(root/'results/ALL_TARGET_RESULTS.json');recon=loadj(root/'source_models_run1/SOURCE_MODEL_MANIFEST.json')
 # Recheck frozen reconstruction identity before scoring.
 freeze=loadj(root/'RECONSTRUCTION_FREEZE.json')
 if freeze['status']!='FROZEN_MATCH' or freeze['pairs_checked']!=15:raise RuntimeError('reconstruction freeze invalid')
 for pair in freeze['pairs']:
  if sha(root/'source_models_run1'/recon['models'][pair['fraction']]['arms'][pair['arm']]['model_file'])!=pair['model_sha256']:raise RuntimeError('source model hash drift')
 # Reference baseline predictions under the pickle-producing sklearn version.
 legacy_npz=out/'legacy_baseline_scores.npz';cmd=[a.legacy_python,a.legacy_script,'--transfer-root',str(root),'--out',str(legacy_npz)]
 for name,path in targets.items():cmd += ['--target',f'{name}={path}']
 subprocess.run(cmd,check=True);legacy=np.load(legacy_npz)
 with warnings.catch_warnings(record=True) as caught:
  warnings.simplefilter('always')
  with open(root/'baseline_models/mlp_model_trip_v1.2.2.pkl','rb') as f:v22=pickle.load(f)
  with open(root/'baseline_models/mlp_model_trip_ghost_v1.2.2.pkl','rb') as f:v23=pickle.load(f)
 compatibility={'schema':'baseline-version-compatibility-1.0','reference_environment':loadj(legacy_npz.with_suffix('.manifest.json'))['environment'],'current_environment':{'python':sys.version.split()[0],'numpy':np.__version__,'scikit_learn':sklearn.__version__,'xgboost':xgboost.__version__},'pickle_hashes':{'V22_MLP':sha(root/'baseline_models/mlp_model_trip_v1.2.2.pkl'),'V23_MLP':sha(root/'baseline_models/mlp_model_trip_ghost_v1.2.2.pkl')},'load_warnings':[str(x.message) for x in caught],'targets':{}}
 abs_rows=[];pair_rows=[];sens_rows=[];envelopes=[];repro={'schema':'published-point-reproduction-1.0','tolerance':1e-12,'rows':[],'max_abs_difference':0.0};protocol={'schema':'bounded-post-analysis-addendum-1.0','bootstraps':BOOT,'bootstrap_unit':'stratified untouched target test IDs','no_simulation':True,'no_model_fit':True,'primary_score':'zero-shot reconstructed-source combined arm','sensitivity_scenarios':['primary','numerical_as_control_stress','numerical_as_event_stress','collision_as_composite_event','collision_as_control_stress','all_scoreable_exclusions_as_control_stress','all_scoreable_exclusions_as_event_stress'],'worker_errors':'counted with prevalence extremes only; no scores/features available'}
 for pidx,(name,path) in enumerate(sorted(targets.items())):
  rows=[json.loads(x) for x in (path/'systems.jsonl').read_text().splitlines()];by={int(r['system_id']):r for r in rows}
  z=np.load(path/'features.npz');ids=z['system_id'].astype(np.int64);pos={int(v):i for i,v in enumerate(ids)};roles=np.array([by[int(i)]['split_role'] for i in ids]);statuses=np.array([by[int(i)]['status'] for i in ids]);terminal=np.array([np.inf if by[int(i)]['status']=='stable' else float(by[int(i)].get('t_terminal_outer') if by[int(i)].get('t_terminal_outer') is not None else -np.inf) for i in ids])
  current_base=vyn_scores(z['ic_raw'],v22,v23);lid=legacy[name+'__ids'].astype(np.int64)
  if not np.array_equal(ids,lid):raise RuntimeError(f'legacy ID mismatch {name}')
  compatibility['targets'][name]={}
  selected_base={'MA01_margin':current_base['MA01_margin'],'V22_formula':current_base['V22_formula']}
  for bn in ('V22_MLP','V23_MLP'):
   cur=np.asarray(current_base[bn],float);old=np.asarray(legacy[name+'__'+bn],float);exact=bool(np.array_equal(cur,old));mx=float(np.max(np.abs(cur-old)));mean=float(np.mean(np.abs(cur-old)));order=bool(np.array_equal(np.argsort(cur,kind='stable'),np.argsort(old,kind='stable')));status='compatible_exact' if exact else ('compatible_numerical' if mx<=1e-12 and order else 'not_established')
   compatibility['targets'][name][bn]={'n':len(cur),'current_score_hash':score_hash(cur),'reference_score_hash':score_hash(old),'exact_array_equal':exact,'max_abs_difference':mx,'mean_abs_difference':mean,'stable_rank_order_equal':order,'status':status};selected_base[bn]=old
  for j,f in enumerate(FRACS):
   h=300*f;valid=((statuses=='stable')|(statuses=='ejected'))&(terminal>h)&(roles=='test');te=np.flatnonzero(valid);y=np.array([int(statuses[i]=='ejected') for i in te],int)
   designs={'IC_only':z['ic'],'history':z['history'][:,j,:],'combined':z['combined'][:,j,:]};score_sets={'zero_shot':{},'target_refit':{},'static_baseline':{}}
   rk=recon_key(recon,f)
   for ai,arm in enumerate(ARMS):
    smeta=recon['models'][rk]['arms'][arm];sm=load_xgb(root/'source_models_run1'/smeta['model_file']);score_sets['zero_shot'][arm]=sm.predict_proba(designs[arm])[:,1]
    rm=load_xgb(root/'results'/name/f'target_refit_{arm}_f{f:.2f}.json');score_sets['target_refit'][arm]=rm.predict_proba(designs[arm])[:,1]
   score_sets['static_baseline']=selected_base
   # Reproduce every published point using current-version scores for historical MLP rows.
   pub=published[name]['horizons'][j]
   for mode,pkey in [('zero_shot','zero_shot'),('target_refit','target_refit')]:
    for arm in ARMS:
     got=metrics(y,score_sets[mode][arm][te]);want=pub[pkey][arm]['metrics']
     d=max(abs(got[k]-want[k]) for k in ('roc_auc','pr_auc','prevalence'));repro['max_abs_difference']=max(repro['max_abs_difference'],d);repro['rows'].append({'population':name,'f':float(f),'mode':mode,'arm':arm,'max_abs_difference':d})
     if d>repro['tolerance'] or got['n']!=want['n'] or got['events']!=want['events']:raise RuntimeError(f'published point mismatch {name} {f} {mode} {arm} {d}')
   for bn in BASES:
    check_score=current_base[bn] if bn in ('V22_MLP','V23_MLP') else selected_base[bn];got=metrics(y,check_score[te]);want=pub['static_baselines'][bn]['metrics'];d=max(abs(got[k]-want[k]) for k in ('roc_auc','pr_auc','prevalence'));repro['max_abs_difference']=max(repro['max_abs_difference'],d);repro['rows'].append({'population':name,'f':float(f),'mode':'static_baseline_published_environment','arm':bn,'max_abs_difference':d})
    if d>repro['tolerance']:raise RuntimeError(f'published baseline mismatch {name} {f} {bn} {d}')
   # Absolute intervals for all published model rows; legacy scores are authoritative for pickle baselines.
   seed0=100000+pidx*1000+j*50
   for mi,(mode,arms) in enumerate([('zero_shot',ARMS),('target_refit',ARMS),('static_baseline',BASES)]):
    for ai,arm in enumerate(arms):
     q=interval(y,np.asarray(score_sets[mode][arm])[te],seed0+mi*10+ai);abs_rows.append({'population':name,'f':float(f),'horizon_Tout':float(h),'mode':mode,'arm':arm,'score_environment':'sklearn_1.2.2_reference' if mode=='static_baseline' and arm in ('V22_MLP','V23_MLP') else 'authorized_main',**q})
   # Paired A1 zero-shot combined minus static baseline intervals.
   aa=np.asarray(score_sets['zero_shot']['combined'])[te]
   for bi,bn in enumerate(BASES):
    q=paired(y,aa,np.asarray(selected_base[bn])[te],seed0+30+bi);pair_rows.append({'population':name,'f':float(f),'horizon_Tout':float(h),'comparison':f'A1_zero_shot_combined_minus_{bn}','baseline_compatibility':compatibility['targets'][name].get(bn,{}).get('status','formula_or_margin_not_pickle'),**q})
   # Failure/competing-risk scenarios for primary zero-shot combined.
   allscore=np.asarray(score_sets['zero_shot']['combined']);num=np.flatnonzero((statuses=='numerical_error')&(terminal>h)&(roles=='test'));col=np.flatnonzero((statuses=='collision')&(terminal>h)&(roles=='test'));worker=sum(r['status']=='worker_error' and r['split_role']=='test' for r in rows);srcmeta=recon['models'][rk]['arms']['combined'];thr=next(r['threshold'] for r in loadj(root/'source_models_run1'/srcmeta['threshold_file'])['rows'] if abs(r['target_far']-.01)<1e-15)
   scenarios=[('primary',te,y),('numerical_as_control_stress',np.r_[te,num],np.r_[y,np.zeros(len(num),int)]),('numerical_as_event_stress',np.r_[te,num],np.r_[y,np.ones(len(num),int)]),('collision_as_composite_event',np.r_[te,col],np.r_[y,np.ones(len(col),int)]),('collision_as_control_stress',np.r_[te,col],np.r_[y,np.zeros(len(col),int)]),('all_scoreable_exclusions_as_control_stress',np.r_[te,num,col],np.r_[y,np.zeros(len(num)+len(col),int)]),('all_scoreable_exclusions_as_event_stress',np.r_[te,num,col],np.r_[y,np.ones(len(num)+len(col),int)])]
   vals=[]
   for scen,ix,yy in scenarios:
    mm=metrics(yy,allscore[ix]);oo=op(yy,allscore[ix],thr);row={'population':name,'f':float(f),'horizon_Tout':float(h),'scenario':scen,'numerical_added':int(len(num)) if 'numerical' in scen or 'all_scoreable' in scen else 0,'collisions_added':int(len(col)) if 'collision' in scen or 'all_scoreable' in scen else 0,'worker_error_test_unknown':worker,**mm,'source_1pct_threshold':thr,'achieved_far':oo['achieved_far'],'coverage':oo['coverage'],'precision':oo['precision']};sens_rows.append(row);vals.append(row)
   denom=len(te)+worker;envelopes.append({'population':name,'f':float(f),'horizon_Tout':float(h),'scoreable_scenario_roc_min':min(v['roc_auc'] for v in vals),'scoreable_scenario_roc_max':max(v['roc_auc'] for v in vals),'scoreable_scenario_pr_min':min(v['pr_auc'] for v in vals),'scoreable_scenario_pr_max':max(v['pr_auc'] for v in vals),'primary_prevalence':float(y.mean()),'worker_unknown_prevalence_lower':float(y.sum()/denom) if denom else None,'worker_unknown_prevalence_upper':float((y.sum()+worker)/denom) if denom else None,'worker_error_test_unknown':worker})
 dump(out/'BASELINE_VERSION_COMPATIBILITY.json',compatibility);dump(out/'POINT_REPRODUCTION.json',repro);dump(out/'ADDENDUM_PROTOCOL.json',protocol);write_csv(out/'ABSOLUTE_METRIC_INTERVALS.csv',abs_rows);write_csv(out/'PAIRED_BASELINE_INTERVALS.csv',pair_rows);write_csv(out/'FAILURE_COMPETING_SENSITIVITY.csv',sens_rows);write_csv(out/'SENSITIVITY_ENVELOPES.csv',envelopes)
 # Do not retain bulky score arrays in the final package; hashes/manifests suffice.
 legacy.close();legacy_npz.unlink();legacy_npz.with_suffix('.manifest.json').rename(out/'LEGACY_BASELINE_ENVIRONMENT.json')
 manifest={'schema':'post-analysis-addendum-output-1.0','transfer_root':str(root),'transfer_results_sha256':sha(root/'results/ALL_TARGET_RESULTS.json'),'reconstruction_freeze_sha256':sha(root/'RECONSTRUCTION_FREEZE.json'),'target_identities':{k:file_identity(v) for k,v in targets.items()},'files':{}}
 for p in sorted(out.iterdir()):
  if p.is_file() and p.name!='ADDENDUM_MANIFEST.json':manifest['files'][p.name]={'bytes':p.stat().st_size,'sha256':sha(p)}
 dump(out/'ADDENDUM_MANIFEST.json',manifest);print(out,flush=True)
if __name__=='__main__':main()
