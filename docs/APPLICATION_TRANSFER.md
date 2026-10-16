# Application-transfer protocol (v1.1.0)

## Scope

The application layer applies a deterministic reconstruction of the frozen v1.0.3 source diagnostic to three declared engineering proxies. It does not modify the v1.0.3 source release.

## Target contracts

- campaign seed: `20261003`;
- population codes: compact `202`, field `303`, young `404`;
- endpoint: 300 initial outer periods;
- horizons: 15, 45, 90, 150, 225 outer periods;
- REBOUND/IAS15, epsilon `1e-9`, 20 stored checks per outer period;
- event status precedence: numerical error, timeout, collision, ejection;
- collision/failure rows never become primary stable controls;
- operational features: 15 IC, 77 history, 92 combined; raw slots 72/73 disabled.

Exact distributions are in `configs/`.

## Transfer rows

Every target/horizon preserves separate rows for:

1. frozen reconstructed source model + source calibration threshold;
2. same source model + target-calibration-negative threshold;
3. target-refit control with fixed model family/hyperparameters.

Target test IDs never enter model fitting or threshold selection.

## Addendum outputs

- 150 absolute interval rows;
- 60 paired A1-minus-comparator rows;
- 105 failure/competing-risk scenarios;
- 15 sensitivity envelopes;
- scikit-learn 1.2.2/1.9.1 baseline compatibility record;
- exact published-point reproduction gate.

## Archival launchers (exact bytes)

The two self-contained Kaggle launchers that ran the application campaign are
archived under `workflows/kaggle/` **without any byte modification**:

| Script | Bytes | SHA-256 |
|---|---:|---|
| `workflows/kaggle/KAGGLE_COMPACT_SELFCONTAINED_10000.py` | 1,105,706 | `41009d2f2664748ecc43cea7fbda5bfca0d9e6f70d80ddae206e430ce80247d3` |
| `workflows/kaggle/KAGGLE_STELLAR_SELFCONTAINED_10000.py` | 1,105,829 | `3d86f4fb43e868b154a19475039bbc40260ef8f49847651450e9b6763b101cae` |

These are the exact identities that were previously declared but unmatched; they are
now matched byte-for-byte, and no substitute or renamed file was used. The embedded
simulation package carried by both scripts decodes to **816,912 bytes**,
SHA-256 `6885ee927831e2491717319d5da39597a725d225932a234a291ce32cb3c19261`, and the
decoded bytes are identical from both scripts. Each launcher checks that hash at
runtime (`EXPECTED_ARCHIVE`) before unpacking.

The decoded package is **not** committed separately — repository policy excludes
binary archives, and the exact scripts already carry it. Neither launcher contains
heartbeat code (zero `heartbeat` tokens; no `threading` import). The embedded package
does include a historical `scripts/KAGGLE_HEARTBEAT_SETUP.py` member inside its
base64 payload, which is recorded here for completeness; no standalone heartbeat file
is added to the tree.

## Reproducibility boundary

The v1.1.0 repository contains code, configs, small official tables, manifests, and the two exact archival launchers. Large source/target arrays, serialized models, and the decoded binary package remain external and are pinned in `application_data/PROVENANCE.md` and `application_data/artifact_registry.json`. The source reconstruction is a declared deterministic refit, not recovered original model bytes.
