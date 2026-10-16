#!/usr/bin/env python3
"""Fail closed if a v1.1.0 registered application artifact drifts."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[2]
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
reg=json.loads((ROOT/'application_data/artifact_registry.json').read_text())
errors=[]
for rel,m in reg['files'].items():
    p=ROOT/rel
    if not p.exists():errors.append(f'missing {rel}');continue
    if p.stat().st_size!=m['bytes']:errors.append(f'bytes {rel}')
    if sha(p)!=m['sha256']:errors.append(f'hash {rel}')
if errors:raise SystemExit('APPLICATION_RELEASE_VERIFICATION_FAILED: '+'; '.join(errors))
print(f"application release verification passed: {len(reg['files'])} files")
