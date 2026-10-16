#!/usr/bin/env python3
"""Download and hash-verify the two public triple-stability baseline pickles."""
from pathlib import Path
import argparse,hashlib,urllib.request
FILES={
 'mlp_model_trip_v1.2.2.pkl':('https://raw.githubusercontent.com/pavanvyn/triple-stability/main/mlp_model_trip_v1.2.2.pkl','f8bc2f192813145f42683669adaef6448a287523875895343453055b3acf1436'),
 'mlp_model_trip_ghost_v1.2.2.pkl':('https://raw.githubusercontent.com/pavanvyn/triple-stability/main/mlp_model_trip_ghost_v1.2.2.pkl','a260845fc9c5b80ede3ab0028ac582bffa8450dad0c23f3ce78d73987932b866'),
}
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',default='baseline_models');a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 for name,(url,want) in FILES.items():
  data=urllib.request.urlopen(url,timeout=120).read();got=sha(data)
  if got!=want:raise SystemExit(f'BASELINE_HASH_MISMATCH {name} {got} != {want}')
  (out/name).write_bytes(data);print(want,name)
if __name__=='__main__':main()
