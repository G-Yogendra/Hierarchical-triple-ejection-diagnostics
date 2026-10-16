from pathlib import Path
ROOT=Path(__file__).parents[2]

def test_transfer_script_contains_required_rows_and_gates():
    s=(ROOT/'workflows/application/run_transfer_analysis.py').read_text()
    for token in ['compare_recon','source_threshold_operation','target_calibrated_operation','target_refit','combined_minus_IC','static_baselines','late_support_decision','paired target test system IDs']:
        assert token in s

def test_addendum_is_analysis_only_and_has_requested_outputs():
    s=(ROOT/'workflows/application/run_post_analysis_addendum.py').read_text()
    assert '.fit(' not in s and 'import rebound' not in s
    for token in ['ABSOLUTE_METRIC_INTERVALS.csv','PAIRED_BASELINE_INTERVALS.csv','FAILURE_COMPETING_SENSITIVITY.csv','SENSITIVITY_ENVELOPES.csv','BASELINE_VERSION_COMPATIBILITY.json','POINT_REPRODUCTION.json']:
        assert token in s

def test_reconstruction_is_fail_closed_and_non_original():
    s=(ROOT/'workflows/application/reconstruct_source_models.py').read_text()
    for token in ['SOURCE_ARRAY_NOT_MOUNTED_OR_HASH_MISMATCH','EXPECTED_SOURCE_SHA','FIT_CAP=200000','deterministic reconstruction from frozen arrays; not original ephemeral model bytes']:
        assert token in s

def test_support_locks_and_public_baseline_hashes_are_present():
    assert (ROOT/'application_support/reconstruction-environment-lock.txt').exists()
    assert (ROOT/'application_support/vendor/a1_v1.0.3/final_manifest.json').exists()
    s=(ROOT/'workflows/application/fetch_public_baselines.py').read_text()
    for token in ['f8bc2f192813145f42683669adaef6448a287523875895343453055b3acf1436','a260845fc9c5b80ede3ab0028ac582bffa8450dad0c23f3ce78d73987932b866','BASELINE_HASH_MISMATCH']:
        assert token in s

# --- R1 exact archival launchers -------------------------------------------

R1_LAUNCHERS = {
    'workflows/kaggle/KAGGLE_COMPACT_SELFCONTAINED_10000.py':
        (1105706, '41009d2f2664748ecc43cea7fbda5bfca0d9e6f70d80ddae206e430ce80247d3'),
    'workflows/kaggle/KAGGLE_STELLAR_SELFCONTAINED_10000.py':
        (1105829, '3d86f4fb43e868b154a19475039bbc40260ef8f49847651450e9b6763b101cae'),
}
R1_EMBEDDED_PACKAGE=(816912,'6885ee927831e2491717319d5da39597a725d225932a234a291ce32cb3c19261')

def test_r1_archival_launchers_preserve_exact_declared_bytes():
    import hashlib
    for rel,(nbytes,digest) in R1_LAUNCHERS.items():
        p=ROOT/rel
        assert p.is_file(),rel
        data=p.read_bytes()
        assert len(data)==nbytes,(rel,len(data))
        assert hashlib.sha256(data).hexdigest()==digest,rel

def test_r1_embedded_package_identity_and_cross_script_equality():
    import base64,hashlib,re
    nbytes,digest=R1_EMBEDDED_PACKAGE
    decoded={}
    for rel in R1_LAUNCHERS:
        text=(ROOT/rel).read_text(errors='replace')
        assert 'EXPECTED_ARCHIVE' in text and digest in text, rel
        blob=re.search(r'ARCHIVE_B64\s*=\s*r?"""\n(.*?)\n"""',text,re.S).group(1)
        raw=base64.b64decode(''.join(blob.split()))
        assert len(raw)==nbytes,rel
        assert hashlib.sha256(raw).hexdigest()==digest,rel
        decoded[rel]=raw
    a,b=decoded.values()
    assert a==b,'embedded package must be byte-identical from both scripts'

def test_r1_archival_launchers_contain_no_heartbeat_code():
    import re
    for rel in R1_LAUNCHERS:
        text=(ROOT/rel).read_text(errors='replace')
        assert 'heartbeat' not in text.lower(),rel
        code=re.sub(r'ARCHIVE_B64\s*=\s*r?"""\n.*?\n"""','',text,flags=re.S)
        assert 'heartbeat' not in code.lower(),rel
        assert 'threading' not in code,rel
        assert 'import jax' not in code,rel
