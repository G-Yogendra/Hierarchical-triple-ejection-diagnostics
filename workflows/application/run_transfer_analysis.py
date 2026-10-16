#!/usr/bin/env python3
"""Score reconstructed A1 source models on all application targets.

Requires two independently reconstructed source-model directories with matching
model/threshold hashes. Produces zero-shot source-threshold, target-calibrated,
and target-refit rows without changing features or hyperparameters.
"""
from __future__ import annotations
import argparse,hashlib,json,pickle,warnings
from concurrent.futures import ThreadPoolExecutor,as_completed
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from xgboost import XGBClassifier

FRACS=np.array([.05,.15,.30,.50,.75]);FARS=(.001,.01,.05);FI=(0,2,4,6,8);BOOT=500
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def loadj(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def model(seed,n_jobs):return XGBClassifier(n_estimators=300,max_depth=3,learning_rate=.05,subsample=.8,colsample_bytree=.8,reg_lambda=2,objective='binary:logistic',eval_metric='logloss',random_state=seed,n_jobs=n_jobs,tree_method='hist')
def cuts(y,s):
 neg=s[y==0];return {str(a):float(np.quantile(neg,1-a,method='higher')) for a in FARS}
def metrics(y,s):return {"n":len(y),"events":int(y.sum()),"controls":int((y==0).sum()),"prevalence":float(y.mean()),"roc_auc":float(roc_auc_score(y,s)),"pr_auc":float(average_precision_score(y,s))}
def operation(y,s,t):
 p=s>=t;n=y==0;e=y==1;return {"threshold":float(t),"achieved_far":float(p[n].mean()),"coverage":float(p[e].mean()),"precision":float(y[p].mean()) if p.any() else None,"alerts":int(p.sum()),"event_alerts":int((p&e).sum()),"control_alerts":int((p&n).sum())}
def paired(y,a,b,seed,nboot=BOOT):
 point=float(roc_auc_score(y,a)-roc_auc_score(y,b));rng=np.random.default_rng(seed);d=[]
 for _ in range(nboot):
  ii=rng.integers(0,len(y),len(y))
  if len(np.unique(y[ii]))<2:continue
  d.append(roc_auc_score(y[ii],a[ii])-roc_auc_score(y[ii],b[ii]))
 return {"point":point,"lo":float(np.percentile(d,2.5)),"hi":float(np.percentile(d,97.5)),"n_boot":len(d),"unit":"paired target test system IDs"}
def load_xgb(path,n_jobs):
 m=XGBClassifier(n_jobs=n_jobs);m.load_model(path);m.set_params(n_jobs=n_jobs);return m
def reconstruction_fraction_key(recon,f,tol=1e-6):
 # Source fractions were stored as float32 (for example
 # 0.05000000074505806), while target landmarks use the declared decimal
 # values (0.05). Resolve by a tight numeric identity tolerance, never by
 # horizon order or nearest-neighbour fallback.
 matches=[k for k in recon['models'] if abs(float(k)-float(f))<=tol]
 if len(matches)!=1:raise RuntimeError(f'source reconstruction fraction resolution failed: target={f!r}, matches={matches}, available={sorted(recon["models"],key=float)}')
 return matches[0]
def compare_recon(a,b):
 A=loadj(Path(a)/'SOURCE_MODEL_MANIFEST.json');B=loadj(Path(b)/'SOURCE_MODEL_MANIFEST.json')
 for f,x in A['models'].items():
  for arm,v in x['arms'].items():
   w=B['models'][f]['arms'][arm]
   if v['model_file_sha256']!=w['model_file_sha256'] or v['threshold_file_sha256']!=w['threshold_file_sha256']:raise RuntimeError(f'reconstruction mismatch {f} {arm}')
 return A
def source_split(y):
 from sklearn.model_selection import train_test_split
 valid=np.where(y>=0)[0];tr,hold=train_test_split(valid,test_size=.4,random_state=26001,stratify=y[valid]);ca,te=train_test_split(hold,test_size=.5,random_state=26002,stratify=y[hold]);return tr,ca,te
def vyn_scores(ic,v22,v23):
 m0,m1,m2=ic[:,0],ic[:,1],ic[:,2];qin=np.minimum(m0,m1)/np.maximum(m0,m1);qout=m2/(m0+m1);alpha=ic[:,3]/ic[:,9];ein=ic[:,4];eout=ic[:,10];imut=ic[:,11];lk=1-(5/3)*np.cos(imut)**2;et=np.where(lk>=0,np.maximum(ein,.5*lk),ein);fac=.125*(1-.2*et+eout)*(np.cos(imut)-1)+1;Y=(1-eout)/((1+et)*alpha);Yc=2.4*(1+qout)**.4*(1+et)**(-.4)*(1-eout)**(-.2)*fac
 with warnings.catch_warnings():
  warnings.simplefilter('ignore');p22=v22.predict_proba(np.c_[qin,qout,alpha,ein,eout,imut/np.pi])[:,1];p23=v23.predict_proba(np.c_[qin,qout,alpha,ein,eout,imut/np.pi])[:,1]
 return {"MA01_margin":-ic[:,16],"V22_formula":np.log(np.maximum(Yc,1e-30)/np.maximum(Y,1e-30)),"V22_MLP":p22,"V23_MLP":p23}
def source_baseline_cuts(source,v22,v23):
 with np.load(source,allow_pickle=False) as z:d={k:z[k] for k in ['statuses','t_event','t_max','ic']}
 y=np.full(len(d['statuses']),-1,int);y[d['statuses']=='stable']=0;y[d['statuses']=='ejected']=1;_,ca,_=source_split(y);event=np.where(np.isnan(d['t_event']),np.inf,d['t_event']);scores=vyn_scores(d['ic'],v22,v23);out={}
 for j,f in enumerate(FRACS):
  ci=ca[event[ca]>f*d['t_max'][ca]];out[str(float(f))]={k:cuts(y[ci],s[ci]) for k,s in scores.items()}
 return out
def verify_target(run):
 run=Path(run);fm=loadj(run/'FILE_MANIFEST.json') if (run/'FILE_MANIFEST.json').exists() else None
 if fm:
  for n,m in fm['files'].items():
   p=run/n
   if not p.exists() or p.stat().st_size!=m['bytes'] or sha(p)!=m['sha256']:raise RuntimeError(f'target file mismatch {p}')
 rows=[json.loads(x) for x in (run/'systems.jsonl').read_text().splitlines()];z=np.load(run/'features.npz');ids=z['system_id'].astype(int)
 if len(rows)!=len({r['system_id'] for r in rows}) or set(ids)!={r['system_id'] for r in rows if r['status']!='worker_error'}:raise RuntimeError('target ID mismatch')
 pos={int(v):i for i,v in enumerate(ids)};return rows,z,pos,fm
def warning(rows,test_ids,score_matrix,thresholds):
 by={r['system_id']:r for r in rows};events=[i for i in test_ids if by[i]['status']=='ejected' and by[i]['t_terminal_outer']>15];controls=[i for i in test_ids if by[i]['status']=='stable'];warned=[];false=0;leads=[]
 for sid in events:
  for j,f in enumerate(FRACS):
   s=score_matrix.get((sid,j))
   if s is not None and s>=thresholds[j]:warned.append(sid);leads.append(by[sid]['t_terminal_outer']-300*f);break
 for sid in controls:
  if any(score_matrix.get((sid,j),-np.inf)>=thresholds[j] for j in range(5)):false+=1
 return {"eligible_events":len(events),"warned_events":len(warned),"coverage":len(warned)/len(events) if events else None,"controls":len(controls),"control_alerts":false,"cumulative_control_alert_fraction":false/len(controls) if controls else None,"median_lead_Tout":float(np.median(leads)) if leads else None,"p25_lead_Tout":float(np.percentile(leads,25)) if leads else None,"p75_lead_Tout":float(np.percentile(leads,75)) if leads else None}
def fixed_growth(rows,test_ids,score_matrix,seed):
 by={r['system_id']:r for r in rows};ids=[i for i in test_ids if by[i]['status'] in ('stable','ejected') and (by[i]['status']=='stable' or by[i]['t_terminal_outer']>225)];y=np.array([int(by[i]['status']=='ejected') for i in ids]);point=[]
 for j in range(5):
  point.append(float(roc_auc_score(y,[score_matrix['combined'][(i,j)] for i in ids])-roc_auc_score(y,[score_matrix['IC_only'][(i,j)] for i in ids])))
 rng=np.random.default_rng(seed);slopes=[]
 for _ in range(500):
  ii=rng.integers(0,len(ids),len(ids));yy=y[ii]
  if len(np.unique(yy))<2:continue
  g=[]
  for j in range(5):
   co=np.array([score_matrix['combined'][(i,j)] for i in ids])[ii];ic=np.array([score_matrix['IC_only'][(i,j)] for i in ids])[ii];g.append(roc_auc_score(yy,co)-roc_auc_score(yy,ic))
  slopes.append(float(np.polyfit(FRACS,g,1)[0]))
 return {"n":len(ids),"events":int(y.sum()),"point_gains":point,"slope":float(np.polyfit(FRACS,point,1)[0]),"slope_lo":float(np.percentile(slopes,2.5)),"slope_hi":float(np.percentile(slopes,97.5)),"n_boot":len(slopes)}
def analyze_target(name,run,recon,recon_dir,source_cuts,v22,v23,out,xgb_threads):
 rows,z,pos,fm=verify_target(run);by={r['system_id']:r for r in rows};valid=[r for r in rows if r['status'] in ('stable','ejected') and r['system_id'] in pos];all_ids=np.array([r['system_id'] for r in valid]);yall=np.array([int(r['status']=='ejected') for r in valid]);role=np.array([r['split_role'] for r in valid]);terminal=np.array([np.inf if r['status']=='stable' else r['t_terminal_outer'] for r in valid]);ii=np.array([pos[int(x)] for x in all_ids]);base=vyn_scores(z['ic_raw'][ii],v22,v23);target_out=Path(out)/name;target_out.mkdir(parents=True,exist_ok=True);result={"population":name,"claim":"engineering transfer analysis","input_hashes":{"systems.jsonl":sha(Path(run)/'systems.jsonl'),"features.npz":sha(Path(run)/'features.npz')},"status_counts":dict(Counter(r['status'] for r in rows)),"horizons":[]};score_matrices={mode:{arm:{} for arm in ['IC_only','history','combined']} for mode in ['zero_shot','target_refit']};thresholds_by_mode={mode:{arm:[] for arm in ['IC_only','history','combined']} for mode in ['source','target_calibrated','target_refit']}
 for j,f in enumerate(FRACS):
  ip=terminal>300*f;tr=np.where(ip&(role=='train'))[0];ca=np.where(ip&(role=='calibration'))[0];te=np.where(ip&(role=='test'))[0];designs={'IC_only':z['ic'][ii],'history':z['history'][ii,j,:],'combined':z['combined'][ii,j,:]};source_fraction_key=reconstruction_fraction_key(recon,f);block={"f":float(f),"source_reconstruction_fraction_key":source_fraction_key,"n_train":len(tr),"n_calibration":len(ca),"n_test":len(te),"events_test":int(yall[te].sum()),"zero_shot":{},"target_refit":{},"static_baselines":{}}
  for arm,X in designs.items():
   srcmeta=recon['models'][source_fraction_key]['arms'][arm];sm=load_xgb(Path(recon_dir)/srcmeta['model_file'],xgb_threads);pc=sm.predict_proba(X[ca])[:,1];pt=sm.predict_proba(X[te])[:,1];src_thr=loadj(Path(recon_dir)/srcmeta['threshold_file']);srcmap={str(r['target_far']):r['threshold'] for r in src_thr['rows']};tgtmap=cuts(yall[ca],pc);zero={"metrics":metrics(yall[te],pt),"source_threshold_operation":{a:operation(yall[te],pt,srcmap[str(a)]) for a in FARS},"target_calibrated_operation":{a:operation(yall[te],pt,tgtmap[str(a)]) for a in FARS}};block['zero_shot'][arm]=zero
   for sid,sc in zip(all_ids[te],pt):score_matrices['zero_shot'][arm][(int(sid),j)]=float(sc)
   thresholds_by_mode['source'][arm].append(srcmap['0.01']);thresholds_by_mode['target_calibrated'][arm].append(tgtmap['0.01'])
   rs=31000+FI[j]*10+['IC_only','history','combined'].index(arm);m=model(rs,xgb_threads);m.fit(X[tr],yall[tr]);prc=m.predict_proba(X[ca])[:,1];prt=m.predict_proba(X[te])[:,1];rp=target_out/f'target_refit_{arm}_f{f:.2f}.json';m.save_model(rp);rmap=cuts(yall[ca],prc);block['target_refit'][arm]={"metrics":metrics(yall[te],prt),"model_sha256":sha(rp),"target_calibrated_operation":{a:operation(yall[te],prt,rmap[str(a)]) for a in FARS}}
   for sid,sc in zip(all_ids[te],prt):score_matrices['target_refit'][arm][(int(sid),j)]=float(sc)
   thresholds_by_mode['target_refit'][arm].append(rmap['0.01'])
  block['zero_shot']['combined_minus_IC']=paired(yall[te],np.array([score_matrices['zero_shot']['combined'][(int(s),j)] for s in all_ids[te]]),np.array([score_matrices['zero_shot']['IC_only'][(int(s),j)] for s in all_ids[te]]),91000+j)
  block['target_refit']['combined_minus_IC']=paired(yall[te],np.array([score_matrices['target_refit']['combined'][(int(s),j)] for s in all_ids[te]]),np.array([score_matrices['target_refit']['IC_only'][(int(s),j)] for s in all_ids[te]]),92000+j)
  for bn,bs in base.items():
   c=cuts(yall[ca],bs[ca]);block['static_baselines'][bn]={"metrics":metrics(yall[te],bs[te]),"source_threshold_operation":{a:operation(yall[te],bs[te],source_cuts[str(float(f))][bn][str(a)]) for a in FARS},"target_calibrated_operation":{a:operation(yall[te],bs[te],c[str(a)]) for a in FARS}}
  result['horizons'].append(block)
 test_ids=[r['system_id'] for r in valid if r['split_role']=='test'];result['warning_operation']={"zero_shot_source_threshold_combined_1pct":warning(rows,test_ids,score_matrices['zero_shot']['combined'],thresholds_by_mode['source']['combined']),"zero_shot_target_calibrated_combined_1pct":warning(rows,test_ids,score_matrices['zero_shot']['combined'],thresholds_by_mode['target_calibrated']['combined']),"target_refit_combined_1pct":warning(rows,test_ids,score_matrices['target_refit']['combined'],thresholds_by_mode['target_refit']['combined'])}
 result['fixed_landmark']={"zero_shot":fixed_growth(rows,test_ids,score_matrices['zero_shot'],93001),"target_refit":fixed_growth(rows,test_ids,score_matrices['target_refit'],93002)}
 late=result['horizons'][-1]['events_test'];result['late_support_decision']={"events_test":late,"rule":">=50 retain; 30-49 broad/consider expansion; <30 expansion required for headline","decision":"retain" if late>=50 else ("broad_expand_recommended" if late>=30 else "underpowered_expand_required")}
 dump(target_out/'TRANSFER_RESULTS.json',result);return result
def main(args):
 recon=compare_recon(args.recon1,args.recon2)
 with open(args.v22,'rb') as f:v22=pickle.load(f)
 with open(args.v23,'rb') as f:v23=pickle.load(f)
 source_cuts=source_baseline_cuts(args.source,v22,v23);out=Path(args.out);out.mkdir(parents=True,exist_ok=True);dump(out/'SOURCE_BASELINE_THRESHOLDS.json',source_cuts)
 items=[item.split('=',1) for item in args.target];completed={}
 if args.target_workers>1:
  with ThreadPoolExecutor(max_workers=args.target_workers) as ex:
   futures={ex.submit(analyze_target,name,Path(path),recon,args.recon1,source_cuts,v22,v23,out,args.xgb_threads):name for name,path in items}
   for fut in as_completed(futures):completed[futures[fut]]=fut.result()
 else:
  for name,path in items:completed[name]=analyze_target(name,Path(path),recon,args.recon1,source_cuts,v22,v23,out,args.xgb_threads)
 results={name:completed[name] for name,_ in items}
 manifest={"schema":"application-transfer-results-0.1","reconstruction_manifest_sha256":sha(Path(args.recon1)/'SOURCE_MODEL_MANIFEST.json'),"execution":{"target_workers":args.target_workers,"xgb_threads_per_target":args.xgb_threads},"files":{}}
 dump(out/'ALL_TARGET_RESULTS.json',results)
 for p in out.rglob('*'):
  if p.is_file():manifest['files'][str(p.relative_to(out))]={"bytes":p.stat().st_size,"sha256":sha(p)}
 dump(out/'OUTPUT_MANIFEST.json',manifest)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--recon1',required=True);ap.add_argument('--recon2',required=True);ap.add_argument('--v22',required=True);ap.add_argument('--v23',required=True);ap.add_argument('--target',action='append',required=True);ap.add_argument('--out',required=True);ap.add_argument('--xgb-threads',type=int,default=4);ap.add_argument('--target-workers',type=int,default=1);args=ap.parse_args();main(args)
