from __future__ import annotations
import hashlib,json,platform,subprocess,sys
from pathlib import Path

def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def git_commit(root):
    try:return subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    except Exception:return 'UNAVAILABLE'
def environment():
    import numpy
    try:import rebound;rv=rebound.__version__
    except Exception:rv='UNAVAILABLE'
    return {"python":sys.version.split()[0],"platform":platform.platform(),"numpy":numpy.__version__,"rebound":rv}
def write_manifest(root,files,out,extra=None):
    root=Path(root);d={"files":{},"environment":environment(),"extra":extra or {}}
    for p in sorted(Path(x) for x in files):
        q=p if p.is_absolute() else root/p;d['files'][str(q.relative_to(root))]={"bytes":q.stat().st_size,"sha256":sha256(q)}
    Path(out).write_text(json.dumps(d,indent=2)+'\n');return d
