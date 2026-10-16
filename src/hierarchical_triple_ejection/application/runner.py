from __future__ import annotations
import json,time,os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from .config import load_config
from .integrator import simulate_one
from .provenance import environment
from .splits import split_role

def _job(args):return simulate_one(*args)
def _atomic_text(path,text):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(text);os.replace(tmp,path)
def _run_block(cfg,ch,ids,workers,endpoint_override,n_per_override):
 args=[(cfg,i,endpoint_override,n_per_override) for i in ids]
 if workers==1:res=[_job(x) for x in args]
 else:
  with ProcessPoolExecutor(max_workers=workers) as ex:res=list(ex.map(_job,args,chunksize=1))
 rows=[];arrays=[]
 for row,arr in res:
  row['split_role']=split_role(cfg['name'],row['system_id']);row['config_hash']=ch;rows.append(row)
  if arr is not None:arrays.append((row['system_id'],arr))
 return rows,arrays

def run(config_path,out_dir,n,workers=1,start_id=0,endpoint_override=None,n_per_override=None,block_size=500):
 cfg,ch=load_config(config_path);out=Path(out_dir);out.mkdir(parents=True,exist_ok=True);blocks=out/'blocks';blocks.mkdir(exist_ok=True);t0=time.perf_counter();block_records=[];new_blocks=0;reused_blocks=0
 for lo in range(start_id,start_id+n,block_size):
  hi=min(start_id+n,lo+block_size);stem=f'block_{lo:08d}_{hi:08d}';jp=blocks/f'{stem}.jsonl';npz=blocks/f'{stem}.npz';done=blocks/f'{stem}.done.json'
  if done.exists() and jp.exists():
   meta=json.loads(done.read_text())
   if meta.get('config_hash')!=ch or meta.get('ids')!=[lo,hi] or meta.get('endpoint_override')!=endpoint_override or meta.get('n_per_override')!=n_per_override:raise RuntimeError(f'checkpoint identity mismatch {stem}')
   reused_blocks+=1;block_records.append((lo,hi,jp,npz if npz.exists() else None));continue
  ids=list(range(lo,hi));rows,arrays=_run_block(cfg,ch,ids,workers,endpoint_override,n_per_override)
  _atomic_text(jp,''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
  if arrays:
   tmp=blocks/f'{stem}.tmp.npz';np.savez_compressed(tmp,system_id=np.array([x[0] for x in arrays]),ic=np.stack([x[1]['ic'] for x in arrays]),ic_raw=np.stack([x[1]['ic_raw'] for x in arrays]),history=np.stack([x[1]['history'] for x in arrays]),history_raw=np.stack([x[1]['history_raw'] for x in arrays]),combined=np.stack([x[1]['combined'] for x in arrays]));os.replace(tmp,npz)
  _atomic_text(done,json.dumps({"ids":[lo,hi],"config_hash":ch,"endpoint_override":endpoint_override,"n_per_override":n_per_override,"n_rows":len(rows),"n_feature_rows":len(arrays)},indent=2)+'\n')
  new_blocks+=1;block_records.append((lo,hi,jp,npz if npz.exists() else None));print(f'checkpoint {cfg["name"]} {lo}:{hi}',flush=True)
 rows=[]
 with (out/'systems.jsonl').open('w') as f:
  for lo,hi,jp,npz in sorted(block_records):
   text=jp.read_text();f.write(text);rows.extend(json.loads(x) for x in text.splitlines())
 arrays=[]
 for lo,hi,jp,npz in sorted(block_records):
  if npz and npz.exists():
   with np.load(npz) as z:
    for j,sid in enumerate(z['system_id']):arrays.append((int(sid),{k:z[k][j] for k in ['ic','ic_raw','history','history_raw','combined']}))
 arrays.sort(key=lambda x:x[0])
 if arrays:np.savez_compressed(out/'features.npz',system_id=np.array([x[0] for x in arrays]),ic=np.stack([x[1]['ic'] for x in arrays]),ic_raw=np.stack([x[1]['ic_raw'] for x in arrays]),history=np.stack([x[1]['history'] for x in arrays]),history_raw=np.stack([x[1]['history_raw'] for x in arrays]),combined=np.stack([x[1]['combined'] for x in arrays]))
 walls=np.array([r['wall_seconds'] for r in rows]);status={k:sum(r['status']==k for r in rows) for k in sorted(set(r['status'] for r in rows))}
 summary={"population":cfg['name'],"config_path":str(config_path),"config_hash":ch,"id_range":[start_id,start_id+n],"n":n,"workers":workers,"block_size":block_size,"new_blocks":new_blocks,"reused_blocks":reused_blocks,"endpoint_override":endpoint_override,"n_per_override":n_per_override,"status_counts":status,"wall_this_invocation_seconds":time.perf_counter()-t0,"per_system_seconds":{"p50":float(np.percentile(walls,50)),"p95":float(np.percentile(walls,95)),"p99":float(np.percentile(walls,99)),"max":float(walls.max())},"bytes_per_system":(out/'systems.jsonl').stat().st_size/n,"environment":environment(),"production_allowed":cfg.get('production_allowed',False),"calibration_status":cfg.get('calibration_status')}
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');return summary
