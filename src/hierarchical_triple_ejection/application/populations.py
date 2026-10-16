from __future__ import annotations
import math
import numpy as np
from ..core import TripleConfig, sample_triple_ic
TWOPI=2*math.pi
RSCH_AU_PER_MSUN=1.973269804e-8
C_AU_PER_YR=63241.0770843
G_AU3_MSUN_YR2=4*math.pi**2


def seed_words(cfg:dict,system_id:int)->list[int]:
    if cfg.get("seed_contract")=="a1_exact_entropy_words_[42,system_id]":
        return [42,int(system_id)]
    return [int(cfg["campaign_seed"]),int(cfg["population_code"]),int(system_id)]

def rng_for(cfg,system_id): return np.random.default_rng(np.random.SeedSequence(seed_words(cfg,system_id)))
def loguniform(rng,lo,hi): return float(10**rng.uniform(math.log10(lo),math.log10(hi)))
def powerlaw(rng,lo,hi,alpha):
    a=alpha+1;u=rng.random()
    return float((u*(hi**a-lo**a)+lo**a)**(1/a)) if abs(a)>1e-12 else loguniform(rng,lo,hi)
def truncnorm(rng,mean,sigma,lo,hi):
    for _ in range(10000):
        x=rng.normal(mean,sigma)
        if lo<=x<=hi:return float(x)
    raise RuntimeError("truncated normal rejection failed")
def beta_scaled(rng,a,b,maximum=1.0,minimum=0.0): return float(minimum+(maximum-minimum)*rng.beta(a,b))
def thermal(rng,maximum): return float(maximum*math.sqrt(rng.random()))
def isotropic_i(rng,max_deg=180):
    cmin=math.cos(math.radians(max_deg));return float(math.acos(rng.uniform(cmin,1.0)))
def phases(rng):
    return dict(inc_inner=0.0,Omega_inner=0.0,omega_inner=float(rng.uniform(0,TWOPI)),M_inner=float(rng.uniform(0,TWOPI)),Omega_outer=float(rng.uniform(0,TWOPI)),omega_outer=float(rng.uniform(0,TWOPI)),M_outer=float(rng.uniform(0,TWOPI)))
def ma_fields(ic):
    q=ic['m2']/(ic['m0']+ic['m1']);e=ic['e_outer'];inc=ic['inc_outer']
    crit=2.8*((1+q)*(1+e)/math.sqrt(1-e))**(2/5)*(1-.3*inc/math.pi)
    ic['MA01_crit']=float(crit);ic['MA01_margin']=float(ic['a_outer']/ic['a_inner']/crit);return ic

def hierarchy_ok(ain,aout,ein,eout): return aout*(1-eout)>=2.5*ain*(1+ein)
def draw_ratio_until(rng,ain,ein,eout,lo,hi):
    required=2.5*(1+ein)/max(1e-12,1-eout)
    lower=max(lo,required)
    if lower>hi: return None
    return ain*loguniform(rng,lower,hi)

def draw_architecture(rng,ain,e_sampler,ratio_lo,ratio_hi):
    for _ in range(10000):
        ein,eout=e_sampler();aout=draw_ratio_until(rng,ain,ein,eout,ratio_lo,ratio_hi)
        if aout is not None:return ein,eout,aout
    raise RuntimeError('hierarchy-conditioned architecture rejection failed')

def draw_mass(rng,spec):
    if spec['distribution']=='power_law':return powerlaw(rng,spec['min'],spec['max'],spec['alpha'])
    if spec['distribution']=='truncated_normal':return truncnorm(rng,spec['mean'],spec['sigma'],spec['min'],spec['max'])
    raise ValueError(spec['distribution'])
def choose(rng,weights):
    keys=list(weights);return str(rng.choice(keys,p=np.array([weights[k] for k in keys])/sum(weights.values())))

def source_reference(cfg,rng,system_id):
    # Use the canonical generator itself so draw order and entropy words match A1.
    tc=TripleConfig(seed=42,mass_range=tuple(cfg['mass_range_msun']),a_in_range=tuple(cfg['a_inner_range_au']),a_out_range=tuple(cfg['a_outer_range_au']),e_in_range=tuple(cfg['e_inner_range']),e_out_range=tuple(cfg['e_outer_range']),inc_range_deg=tuple(cfg['mutual_inclination_deg']))
    return sample_triple_ic(int(system_id),tc),{"population_class":"source_reference","source_regeneration":"canonical_a1_sample_triple_ic"}

def compact(cfg,rng):
    cls=choose(rng,cfg['inner_class_weights'])
    bh=lambda:draw_mass(rng,cfg['bh_mass_msun']);ns=lambda:draw_mass(rng,cfg['ns_mass_msun'])
    if cls=='BBH':m0,m1=bh(),bh()
    elif cls=='BHNS':m0,m1=bh(),ns()
    else:m0,m1=ns(),ns()
    m2=bh() if choose(rng,cfg['outer_class_weights'])=='BH' else ns()
    ain=loguniform(rng,cfg['a_inner_au']['min'],cfg['a_inner_au']['max'])
    es=lambda:(float(rng.uniform(cfg['e_inner']['min'],cfg['e_inner']['max'])),float(rng.uniform(cfg['e_outer']['min'],cfg['e_outer']['max'])))
    ein,eout,aout=draw_architecture(rng,ain,es,cfg['a_ratio']['min'],cfg['a_ratio']['max']);inc=isotropic_i(rng,cfg['mutual_inclination']['max_deg'])
    ic=ma_fields(dict(m0=m0,m1=m1,m2=m2,a_inner=ain,e_inner=ein,a_outer=aout,e_outer=eout,inc_outer=inc,**phases(rng)))
    rp=ain*(1-ein);rs=RSCH_AU_PER_MSUN*(m0+m1);v=math.sqrt(G_AU3_MSUN_YR2*(m0+m1)/rp)/C_AU_PER_YR
    flags={"population_class":cls,"inner_pericenter_over_combined_rsch":rp/rs,"characteristic_v_over_c":v,"newtonian_flag":bool(rp/rs>=cfg['validity_flags']['min_inner_pericenter_schwarzschild'] and v<=cfg['validity_flags']['max_v_over_c'])}
    return ic,flags

def stellar(cfg,rng,young=False):
    m0=draw_mass(rng,cfg['primary_mass_msun'])
    qspec=cfg['inner_mass_ratio'];q=beta_scaled(rng,qspec['a'],qspec['b'],1,qspec.get('min',0)) if qspec['distribution']=='beta' else float(rng.uniform(qspec['min'],qspec['max']))
    m1=m0*q;ospec=cfg['outer_mass_ratio'];qo=float(rng.uniform(ospec['min'],ospec['max']));m2=(m0+m1)*qo
    a=cfg['a_inner_au'];ain=loguniform(rng,a['min'],a['max']) if a['distribution']=='log_uniform' else float(np.clip(10**rng.normal(a['log10_mean'],a['log10_sigma']),a['min'],a['max']))
    def de(key):
        z=cfg[key]
        return thermal(rng,z['max']) if z['distribution']=='thermal_truncated' else beta_scaled(rng,z['a'],z['b'],z['max'])
    r=cfg['a_ratio'];ein,eout,aout=draw_architecture(rng,ain,lambda:(de('e_inner'),de('e_outer')),r['min'],r['max'])
    isp=cfg['mutual_inclination']
    if isp['distribution']=='isotropic':inc=isotropic_i(rng,isp['max_deg'])
    else:
        inc=abs(float(rng.normal(0,math.radians(isp['coplanar_sigma_deg'])))) if rng.random()<isp['coplanar_weight'] else isotropic_i(rng,180)
        inc=min(inc,math.pi)
    return ma_fields(dict(m0=m0,m1=m1,m2=m2,a_inner=ain,e_inner=ein,a_outer=aout,e_outer=eout,inc_outer=inc,**phases(rng))),{"population_class":"young_cluster_like" if young else "field_like"}

def sample_population(cfg:dict,system_id:int):
    rng=rng_for(cfg,system_id);kind=cfg['sampler']
    if kind=='source_reference':ic,flags=source_reference(cfg,rng,system_id)
    elif kind=='compact_object':ic,flags=compact(cfg,rng)
    elif kind=='field_like':ic,flags=stellar(cfg,rng,False)
    elif kind=='young_cluster_like':ic,flags=stellar(cfg,rng,True)
    else:raise ValueError(kind)
    return ic,{**flags,"seed_words":seed_words(cfg,system_id)}
