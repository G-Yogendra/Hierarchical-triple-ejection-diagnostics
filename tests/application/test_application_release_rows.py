from pathlib import Path
import csv,json
import numpy as np
from hierarchical_triple_ejection.application.bootstrap import stratified_indices
from hierarchical_triple_ejection.application.sensitivity import SCENARIOS,build_scenarios
ROOT=Path(__file__).parents[2]
def rows(name):return list(csv.DictReader((ROOT/'application_data'/name).open()))
def one(rs,**kw):return next(r for r in rs if all(r[k]==str(v) for k,v in kw.items()))

def test_official_row_count_invariants():
    assert len(rows('ABSOLUTE_METRIC_INTERVALS.csv'))==150
    assert len(rows('PAIRED_BASELINE_INTERVALS.csv'))==60==3*5*4
    assert len(rows('FAILURE_COMPETING_SENSITIVITY.csv'))==105==3*5*7
    assert len(rows('SENSITIVITY_ENVELOPES.csv'))==15==3*5

def test_headline_zero_shot_cells():
    r=rows('ABSOLUTE_METRIC_INTERVALS.csv')
    expected={
      ('field_like_stellar','0.05'):(.9936982836495032,.8682239294645977),
      ('compact_object_newtonian','0.05'):(.9700094508033184,.6742598883120775),
      ('young_cluster_like_stellar','0.05'):(.9682025484397025,.7110184927913125),
      ('field_like_stellar','0.5'):(.9911865205608578,.5624854040439006),
      ('compact_object_newtonian','0.5'):(.969084099421178,.37949797808395264),
      ('young_cluster_like_stellar','0.5'):(.9556146333005837,.29564265139166473),
    }
    for (pop,f),(roc,pr) in expected.items():
      q=one(r,population=pop,f=f,mode='zero_shot',arm='combined');assert float(q['roc_auc'])==roc and float(q['pr_auc'])==pr

def test_refit_spot_cells():
    r=rows('ABSOLUTE_METRIC_INTERVALS.csv')
    compact=one(r,population='compact_object_newtonian',f='0.05',mode='target_refit',arm='combined')
    young=one(r,population='young_cluster_like_stellar',f='0.5',mode='target_refit',arm='combined')
    assert round(float(compact['roc_auc']),3)==.988 and round(float(compact['pr_auc']),3)==.842
    assert round(float(young['roc_auc']),3)==.971 and round(float(young['pr_auc']),3)==.354

def test_stratified_bootstrap_is_seeded_and_class_preserving():
    y=np.array([0]*8+[1]*3);a=list(stratified_indices(y,5,123));b=list(stratified_indices(y,5,123))
    assert all(np.array_equal(x,z) for x,z in zip(a,b))
    assert all((y[x]==0).sum()==8 and (y[x]==1).sum()==3 for x in a)

def test_seven_sensitivity_scenarios():
    s=build_scenarios([0,1],[0,1],[2,3],[4]);assert tuple(s)==SCENARIOS
    assert len(s)==7 and s['all_scoreable_exclusions_as_event_stress'][1].sum()==4

def test_application_registry_hashes_are_well_formed():
    reg=json.loads((ROOT/'application_data/artifact_registry.json').read_text());assert reg['release']=='v1.1.0'
    for m in reg['files'].values():assert m['bytes']>0 and len(m['sha256'])==64

def test_v110_manifest_is_additive_to_v103():
    m=json.loads((ROOT/'results/final_manifest_v1.1.0.json').read_text())
    assert m['manifest_version']=='1.1.0' and m['base_release']['tag']=='v1.0.3'
    assert m['base_release']['immutable'] and not m['scientific_change_to_v1_0_3']
    assert m['row_counts']=={'ABSOLUTE_METRIC_INTERVALS.csv':150,'PAIRED_BASELINE_INTERVALS.csv':60,'FAILURE_COMPETING_SENSITIVITY.csv':105,'SENSITIVITY_ENVELOPES.csv':15}
