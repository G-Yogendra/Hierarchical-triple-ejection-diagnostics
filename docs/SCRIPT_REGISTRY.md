# Script Registry — v1.1.0 application layer

A paper-result-to-function map is provided in `docs/APPLICATION_CODE_INDEX.md`.

| Script | Role | Scientific write access |
|---|---|---|
| `workflows/application/run_target_simulation.py` | Deterministic target simulation entry point | Writes target rows/features only |
| `workflows/application/reconstruct_source_models.py` | Double deterministic source reconstruction | Writes reconstructed models/thresholds |
| `workflows/application/run_transfer_analysis.py` | Zero-shot, target-calibrated, target-refit, static-comparator analysis | Writes transfer results |
| `workflows/application/fetch_public_baselines.py` | Download/hash-verify public V22/V23 pickle objects | Writes verified baseline model files only |
| `workflows/application/legacy_baseline_predict.py` | scikit-learn 1.2.2 baseline compatibility scores | Writes compatibility scores |
| `workflows/application/run_post_analysis_addendum.py` | Absolute/paired intervals and failure sensitivity | Analysis only; contains no `.fit()` or REBOUND call |

Package modules under `src/hierarchical_triple_ejection/application/` implement configs, target generators, deterministic splits, integration, feature schema, block runner, leakage checks, transfer utilities, and provenance.

## Archival launchers (`workflows/kaggle/`)

| Script | Role | Scientific write access |
|---|---|---|
| `KAGGLE_COMPACT_SELFCONTAINED_10000.py` | Exact executed compact-application launcher (bytes unmodified) | None — archival record; embeds the frozen simulation package and runs it on Kaggle |
| `KAGGLE_STELLAR_SELFCONTAINED_10000.py` | Exact executed stellar-application launcher (bytes unmodified) | None — archival record; same embedded package |

Both launchers are hash-pinned in `application_data/artifact_registry.json` and were verified byte-exact against the identities declared for the application campaign. They are archives of executed code, not supported repository entry points; see `docs/APPLICATION_TRANSFER.md`.

## Frozen contracts

- endpoint: 300 initial outer periods;
- horizons: 0.05, 0.15, 0.30, 0.50, 0.75;
- campaign seed: 20261003 for application targets;
- feature dimensions: 15 IC, 77 active history, 92 combined;
- disabled raw history slots: 72 and 73;
- target test IDs never enter fitting or threshold selection.
