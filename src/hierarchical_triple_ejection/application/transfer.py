from __future__ import annotations
import json,pickle
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score

def model(seed=0):
    try:from xgboost import XGBClassifier
    except ImportError as e:raise RuntimeError('xgboost required for transfer fits') from e
    return XGBClassifier(n_estimators=300,max_depth=3,learning_rate=.05,subsample=.8,colsample_bytree=.8,reg_lambda=2,objective='binary:logistic',eval_metric='logloss',random_state=seed,n_jobs=4,tree_method='hist')
def target_far_threshold(y,score,alpha): return float(np.quantile(score[np.asarray(y)==0],1-alpha,method='higher'))
def metrics(y,score): return {"n":len(y),"prevalence":float(np.mean(y)),"roc_auc":float(roc_auc_score(y,score)),"pr_auc":float(average_precision_score(y,score))}
def operation(y,score,threshold):
    y=np.asarray(y);p=np.asarray(score)>=threshold;neg=y==0;pos=y==1
    return {"threshold":float(threshold),"achieved_far":float(p[neg].mean()),"coverage":float(p[pos].mean()),"precision":float(y[p].mean()) if p.any() else None,"n_alerts":int(p.sum()),"n_events":int(pos.sum()),"n_controls":int(neg.sum())}
def save_model(obj,path,meta):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('wb') as f:pickle.dump(obj,f)
    path.with_suffix(path.suffix+'.json').write_text(json.dumps(meta,indent=2)+'\n')
def load_model(path):
    with open(path,'rb') as f:return pickle.load(f)
