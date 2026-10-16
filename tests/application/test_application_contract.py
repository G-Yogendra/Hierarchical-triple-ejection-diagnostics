from pathlib import Path
import json
import numpy as np
from hierarchical_triple_ejection.application.populations import sample_population,hierarchy_ok,seed_words
from hierarchical_triple_ejection.application.features import schema_record,ic_operational_vector,ic_storage_vector,combined_matrix
from hierarchical_triple_ejection.application.splits import split_role

ROOT=Path(__file__).parents[2]
CONFIGS=['source_reference.json','field_calibrated.json','compact_object_newtonian.json','young_cluster_calibrated.json']

def load(name):return json.loads((ROOT/'configs'/name).read_text())

def test_configs_and_seed_contracts():
    for name in CONFIGS:
        cfg=load(name);assert cfg['endpoint_outer_periods']==300 and cfg['n_per_orbit']==20
        a=sample_population(cfg,17);b=sample_population(cfg,17);assert a==b
        assert a[1]['seed_words']==seed_words(cfg,17)

def test_target_hierarchy_and_dimensions():
    for name in CONFIGS:
        cfg=load(name);ic,_=sample_population(cfg,123)
        assert hierarchy_ok(ic['a_inner'],ic['a_outer'],ic['e_inner'],ic['e_outer'])
        assert ic_storage_vector(ic).shape==(19,)
        op=ic_operational_vector(ic);assert op.shape==(15,)
        assert combined_matrix(op,np.zeros((5,77),np.float32)).shape==(5,92)

def test_feature_contract():
    assert schema_record()=={'stored_ic_columns':19,'operational_ic_columns':15,'raw_history_columns':79,'disabled_indices':[72,73],'active_history_columns':77,'combined_columns':92,'horizons':[.05,.15,.30,.50,.75]}

def test_stable_hash_split_is_deterministic_and_disjoint():
    for pop in ['compact_object_newtonian','field_like_stellar','young_cluster_like_stellar']:
        roles=[split_role(pop,i) for i in range(10000)]
        assert set(roles)=={'train','calibration','test'}
        assert roles==[split_role(pop,i) for i in range(10000)]
        assert 5800<roles.count('train')<6200 and 1800<roles.count('test')<2200
