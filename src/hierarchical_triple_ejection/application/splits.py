from __future__ import annotations
import hashlib

def split_role(population:str,system_id:int,seed:int=26001)->str:
    b=f'{seed}|{population}|{int(system_id)}'.encode();u=int.from_bytes(hashlib.sha256(b).digest()[:8],'big')/2**64
    return 'train' if u<.6 else ('calibration' if u<.8 else 'test')
