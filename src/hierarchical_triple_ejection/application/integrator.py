from __future__ import annotations
import os,time,traceback
import numpy as np
from .. import core as a1_core
TripleConfig=a1_core.TripleConfig
from .features import history_raw_matrix,history_active_matrix,ic_storage_vector,ic_operational_vector,combined_matrix,schema_record
from .populations import sample_population

def simulate_one(cfg:dict,system_id:int,endpoint_override=None,n_per_override=None):
    ic,flags=sample_population(cfg,system_id)
    endpoint=float(endpoint_override or cfg['endpoint_outer_periods']);nper=int(n_per_override or cfg['n_per_orbit'])
    point=cfg.get('collision_policy')=='point_mass_none'
    tc=TripleConfig(n_systems=1,seed=int(cfg['campaign_seed']),t_max_outer_periods=endpoint,n_per_orbit=nper,timeout_s=float(cfg.get('timeout_seconds',900)),max_relative_energy_error=1e-8,use_physical_radii=not point,collision_radius_factor=0.0 if point else .01)
    t0=time.perf_counter()
    try:
        # Reuse the canonical event loop while setting the production tolerance
        # explicitly on the Simulation object it constructs.
        original_make=a1_core.make_triple_sim
        def make_with_epsilon(initial_conditions):
            sim=original_make(initial_conditions)
            sim.integrator.epsilon=float(cfg['ias15_epsilon'])
            return sim
        a1_core.make_triple_sim=make_with_epsilon
        try:
            rec=a1_core.simulate_triple(system_id,tc,ic_override=ic)
        finally:
            a1_core.make_triple_sim=original_make
        wall=time.perf_counter()-t0
        hist_raw=history_raw_matrix(rec,cfg['horizons']);hist=history_active_matrix(hist_raw);ic_raw=ic_storage_vector(ic);icv=ic_operational_vector(ic);comb=combined_matrix(icv,hist)
        pout=a1_core.outer_period_years(ic);terminal=rec['t_event']
        row={"system_id":int(system_id),"population":cfg['name'],"seed_words":flags['seed_words'],"m0":ic['m0'],"m1":ic['m1'],"m2":ic['m2'],"a_in":ic['a_inner'],"a_out":ic['a_outer'],"e_in":ic['e_inner'],"e_out":ic['e_outer'],"mutual_inclination":ic['inc_outer'],"omega_inner":ic['omega_inner'],"M_inner":ic['M_inner'],"Omega_outer":ic['Omega_outer'],"omega_outer":ic['omega_outer'],"M_outer":ic['M_outer'],"status":rec['status'],"t_terminal":terminal,"t_terminal_outer":None if terminal is None else float(terminal/pout),"initial_outer_period_years":pout,"escaper_index":rec['escaper'],"endpoint_outer_periods":endpoint,"endpoint_300":int(rec['status']=='ejected') if endpoint==300 else None,"endpoint_3000_if_run":int(rec['status']=='ejected') if endpoint==3000 else None,"max_relative_energy_error":rec['max_rel_err'],"wall_seconds":wall,"integrator_settings":{"name":"IAS15","epsilon":cfg['ias15_epsilon'],"stored_checks_per_outer_period":nper,"timeout_seconds":tc.timeout_s,"event_checks":"stored states only","status_precedence":["numerical_error","timeout","collision","ejected"]},"feature_schema":schema_record(),"code_commit":os.environ.get('ASTRO_TRANSFER_COMMIT','UNCOMMITTED_WORKSPACE'),"vendor_core_sha256":"de07bbc2bbdb52f0aa4f84336d4e8f85f9f16885c43c10e8e46426fef7b6b342","vendor_features_sha256":"3d6f682d3380963cc67d268d72fb80d54df682453361b243b5d5f2bb4de7ad69","population_flags":{k:v for k,v in flags.items() if k!='seed_words'}}
        arrays={"ic":icv,"ic_raw":ic_raw,"history":hist,"history_raw":hist_raw,"combined":comb}
        return row,arrays
    except Exception as e:
        wall=time.perf_counter()-t0
        row={"system_id":int(system_id),"population":cfg['name'],"seed_words":flags['seed_words'],"status":"worker_error","t_terminal":None,"wall_seconds":wall,"error_type":type(e).__name__,"error":str(e),"traceback":traceback.format_exc(),"population_flags":{k:v for k,v in flags.items() if k!='seed_words'}}
        return row,None
