from __future__ import annotations
import argparse,json
from pathlib import Path
from .runner import run

def main():
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest='cmd',required=True)
    for cmd in ['smoke','benchmark','pilot','production']:
        p=sp.add_parser(cmd);p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--n',type=int,default=20 if cmd=='smoke' else (1000 if cmd=='benchmark' else 5000));p.add_argument('--workers',type=int,default=1 if cmd=='smoke' else 96);p.add_argument('--start-id',type=int,default=0);p.add_argument('--endpoint-override',type=float,default=2.0 if cmd=='smoke' else None);p.add_argument('--n-per-override',type=int,default=8 if cmd=='smoke' else None);p.add_argument('--block-size',type=int,default=500)
    a=ap.parse_args();
    if a.cmd=='production':
        cfg=json.loads(Path(a.config).read_text())
        if not cfg.get('production_allowed',False):raise SystemExit('STOP: population is not literature/source locked; production_allowed=false')
    print(json.dumps(run(a.config,a.out,a.n,a.workers,a.start_id,a.endpoint_override,a.n_per_override,a.block_size),indent=2))
if __name__=='__main__':main()
