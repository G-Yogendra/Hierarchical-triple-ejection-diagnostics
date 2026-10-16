"""Hash-verified loading of deterministic reconstructed source models."""
from pathlib import Path
import hashlib,json

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def load_manifest(recon_dir):
    p=Path(recon_dir)/'SOURCE_MODEL_MANIFEST.json';m=json.loads(p.read_text())
    if m.get('claim')!='deterministic reconstruction from frozen arrays; not original ephemeral model bytes':
        raise RuntimeError('unexpected reconstruction claim')
    return m

def fraction_key(manifest,f,tol=1e-6):
    matches=[k for k in manifest['models'] if abs(float(k)-float(f))<=tol]
    if len(matches)!=1:raise RuntimeError(f'fraction resolution failed {f}: {matches}')
    return matches[0]

def load_arm(recon_dir,manifest,f,arm,n_jobs=4):
    from xgboost import XGBClassifier
    key=fraction_key(manifest,f);meta=manifest['models'][key]['arms'][arm];p=Path(recon_dir)/meta['model_file']
    got=sha256(p)
    if got!=meta['model_file_sha256']:raise RuntimeError(f'model hash mismatch {p}: {got}')
    t=Path(recon_dir)/meta['threshold_file']
    if sha256(t)!=meta['threshold_file_sha256']:raise RuntimeError(f'threshold hash mismatch {t}')
    model=XGBClassifier(n_jobs=n_jobs);model.load_model(p);model.set_params(n_jobs=n_jobs)
    return model,json.loads(t.read_text()),meta
