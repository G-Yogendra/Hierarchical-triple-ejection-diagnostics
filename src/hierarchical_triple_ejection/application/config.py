from __future__ import annotations
import hashlib, json
from pathlib import Path


def load_config(path: str | Path) -> tuple[dict, str]:
    path=Path(path)
    raw=path.read_bytes()
    cfg=json.loads(raw)
    required={"name","population_code","campaign_seed","sampler","endpoint_outer_periods","n_per_orbit","horizons"}
    missing=required-set(cfg)
    if missing: raise ValueError(f"missing config keys: {sorted(missing)}")
    return cfg,hashlib.sha256(raw).hexdigest()


def canonical_json_sha256(obj) -> str:
    b=json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    return hashlib.sha256(b).hexdigest()
