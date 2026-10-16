# Hierarchical Triple Ejection Diagnostics

Causal, finite-time forecasting of ejection in hierarchical triple systems and of outcomes in binary–single encounters — with calibrated warning lead times and censoring-aware validation.

> **Release status:** `v1.0.3` is the immutable frozen A1 source-layer release.
> `v1.1.0` is the integrated application-transfer release (additive; the `v1.0.3` source layer remains frozen).
> It adds deterministic engineering-proxy generators, transfer/reconstruction workflows,
> official machine-readable interval/sensitivity tables, provenance registries, tests,
> and CI. It does not change any v1.0.3 scientific value, display, or tag.

## Scientific scope

REBOUND/IAS15 integrations plus physical-time observation windows are used to study:

- ejection by a finite integration cutoff in hierarchical triples;
- inner-member versus outer-member ejection;
- flyby, exchange, and ionization in binary–single encounters;
- warning lead time at calibrated false-alarm rates;
- time remaining to ejection;
- right-censoring and delayed ejection in a matched deep-tail sample.

The term **stable** means *not ejected by the stated integration cutoff*. It does not mean permanently stable. High IID discrimination is not exact prediction of chaotic trajectories; the boundary suite, blocked splits, PR curves, and fixed-FAR tests are the stringent checks.

## Production samples (final)

| Suite | Size | Integration / evaluation |
|---|---:|---|
| Main triples | 500,000 | to 300 outer periods, seed 42 |
| Near-MA01 boundary triples | 100,000 | causal boundary-stratum evaluation |
| Binary–single encounters | 1,000,000 | flyby / exchange / ionization |
| Matched deep tail | 100,000 (2 × 50,000) | to 3000 outer periods, seed 42 |

Deep-tail ledger (verified): chunk 0 ejected 20,064 / chunk 1 ejected 20,015; merged ejected 40,079, stable 59,853, numerical-error 67, collision 1.

## Leakage policy (FINAL)

The historical 79-column stored schema contained two future-length-dependent columns: index 73 (`n_frac`) and index 72 (`first_breach` under full-record normalization). In every FINAL model input, **both are identically zero at training and serving time** (see `results/final_analysis_4core/analysis_audit.json`, the `apply_final_mask` helper, and `docs/LEAKAGE_CORRECTION.md`). New extraction keeps a zero placeholder at 73 and normalizes 72 by the causal window. Invariance tests require post-horizon modifications to leave feature vectors bit-identical.

## Explicitly superseded (do not use for paper claims)

- `results/corrected_analysis_4core/` — index 73 only; superseded by `results/final_analysis_4core/`.
- `workflows/kaggle/CORRECTED_*` and earlier `FINAL_*` drafts — provenance only; paper outputs come from `workflows/final/` (frozen nine, receipt tracked in `SCRIPT_REGISTRY.json`).
- `attribution-growth` v1 (platform-staged) — superseded by `attribution-growth-v2`.
- `results/adversarial_audit_4core/` remains valid as the adversarial-controls record.

## Review / release policy

- During review the manuscript cites the frozen GitHub release only if compatible with ApJ dual-anonymous instructions; otherwise the commit/archive goes privately to editor/referees. The manuscript cites no external staging datasets.
- Large NPZ/PKL products stay staged (available to editor/referees on request) and are deposited on Zenodo upon acceptance, verified against [`results/artifact_registry.json`](results/artifact_registry.json). See [`docs/ARTIFACT_REGISTRY.md`](docs/ARTIFACT_REGISTRY.md).

## Repository layout

```text
src/hierarchical_triple_ejection/  simulation and causal-feature package
workflows/final/                   frozen FINAL run scripts + SHA-256 registry
workflows/kaggle/                  archived earlier direct-execution scripts (provenance)
results/final_analysis_4core/      FINAL analysis JSONs (hash-verified on arrival)
results/final_audit_expanded/      expanded audit JSONs + checksums (NPZs excluded)
results/provenance/                chunk metas, deep-tail manifest + checksums
results/tolerance/                 tolerance report, ID list, delayed-join summary, disposition
results/final_manifest.json        frozen hashes, ledger, unresolved fields
results/artifact_registry.json     95-file artifact map + Zenodo plan
scripts/make_release_tables.py     machine-checked LaTeX tables from results
scripts/make_release_figures.py    machine-checked figures from results
scripts/production/                display generators for the A1 main figures/tables
tests/                             invariance, mask, schema, compilation, release tests
docs/                              methods, manifests, registries, verification log
paper/                             anonymous AASTeX manuscript (structure; prose by manuscript agent)
paper/display_sources/             display source manifest (release paths + SHA-256)
paper/drafts/                      author draft (not the submission manuscript; see its README)
```

## v1.1.0 application-transfer layer

The additive application layer evaluates a deterministic reconstruction of the frozen source diagnostic on field-like, compact-object-Newtonian, and young-cluster-like engineering proxies. Exact contracts are in `configs/`; source code is in `src/hierarchical_triple_ejection/application/`; runnable workflows are in `workflows/application/`; and official 150/60/105/15-row tables are in `application_data/`.

The release keeps zero-shot source-threshold, target-calibrated, and target-refit rows separate. Target test IDs never enter fitting or threshold selection. Collisions and failures never become primary stable controls. See [`docs/APPLICATION_TRANSFER.md`](docs/APPLICATION_TRANSFER.md), [`docs/APPLICATION_CODE_INDEX.md`](docs/APPLICATION_CODE_INDEX.md), and [`application_data/PROVENANCE.md`](application_data/PROVENANCE.md).

The targets remain engineering proxies; no realistic-population, PN/GW, merger-rate, individual-system, or full-cluster claim is authorized. Large arrays/models remain external and hash-pinned.

## Installation

Pinned release environment:

```bash
python -m pip install -r requirements-release.txt
python -m pip install -e . --no-deps
pytest
```

Development installation:

```bash
python -m pip install -e '.[dev]'
pytest
```

## Reproduce tables and figures

```bash
python scripts/make_release_tables.py
python scripts/make_release_figures.py
```

Outputs land in `paper/tables/` and `paper/figures/`. Inputs that have not arrived yet print `SKIP (input pending)` lines; see `results/*/PENDING.json` and [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

The main-text display bundle is regenerated by `scripts/production/make_a1_main_results_figures.py` and `scripts/production/make_a1_main_table_data.py`, which load the pinned tolerance summary rather than hard-coding delayed-tolerance counts. Their inputs are recorded per panel and table in [`paper/display_sources/MAIN_DISPLAY_SOURCE_MANIFEST.json`](paper/display_sources/MAIN_DISPLAY_SOURCE_MANIFEST.json); the author-staging path aliases are listed under `display_provenance` in `results/final_manifest.json`.

## License and citation

The software is released under the [MIT License](LICENSE). Citation metadata are provided in [`CITATION.cff`](CITATION.cff); the release DOI is minted on acceptance.
