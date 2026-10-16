# Application code index

| Paper/result function | Canonical source |
|---|---|
| Compact/field/young deterministic generators | `src/hierarchical_triple_ejection/application/populations.py` |
| Campaign seed words and hierarchy conditioning | `populations.py: seed_words, draw_architecture, compact, stellar` |
| REBOUND/IAS15 target integration and status precedence | `application/integrator.py` |
| 19/15 IC, 79/77 history, 92 combined schema | `application/features.py`, `application/a1_features.py` |
| Stable deterministic 60/20/20 roles | `application/splits.py` |
| Checkpointed multiprocessing target runs | `application/runner.py` |
| CLI for smoke/pilot/production | `application/cli.py`, `workflows/application/run_target_simulation.py` |
| Double source reconstruction and thresholds | `workflows/application/reconstruct_source_models.py` |
| Hash-verified reconstructed-model loading | `application/model_io.py: load_manifest, load_arm` |
| Horizon-specific complete-case risk sets | `application/risk_sets.py` |
| Deterministic stratified/paired resampling | `application/bootstrap.py` |
| Seven sensitivity scenario construction | `application/sensitivity.py` |
| Zero-shot and target-calibrated scoring | `workflows/application/run_transfer_analysis.py: analyze_target` |
| Target-refit controls | `run_transfer_analysis.py: analyze_target` |
| Paired combined-minus-IC intervals | `run_transfer_analysis.py: paired` |
| Five-landmark warning coverage/control alerts/lead | `run_transfer_analysis.py: warning` |
| Static MA01/V22/V23 rows | `run_transfer_analysis.py: vyn_scores, source_baseline_cuts` |
| 225-event support rule | `run_transfer_analysis.py: late_support_decision` |
| Legacy sklearn 1.2.2 predictions | `workflows/application/legacy_baseline_predict.py` |
| Baseline pickle acquisition and hashes | `workflows/application/fetch_public_baselines.py` |
| 150 absolute bootstrap rows | `run_post_analysis_addendum.py: interval` |
| 60 paired baseline rows | `run_post_analysis_addendum.py: paired` |
| 105 failure/competing scenarios | `run_post_analysis_addendum.py: scenarios` |
| 15 sensitivity envelopes | `run_post_analysis_addendum.py: envelopes` |
| Point-reproduction gate | `run_post_analysis_addendum.py: POINT_REPRODUCTION` |
| Application figure regeneration | `workflows/application/make_application_figures.py` |
| Archival launcher: compact application, 10,000 systems | `workflows/kaggle/KAGGLE_COMPACT_SELFCONTAINED_10000.py` |
| Archival launcher: stellar application, 10,000 systems | `workflows/kaggle/KAGGLE_STELLAR_SELFCONTAINED_10000.py` |
| Config contracts | `configs/*.json` |
| Official machine-readable rows | `application_data/*.csv` |
| External arrays/models identities | `application_data/PROVENANCE.md`, `EXTERNAL_ARTIFACTS.json` |
| Release verification | `workflows/application/verify_release.py` |
