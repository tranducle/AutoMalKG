# AutoMalKG S2 public hosted audit

This anonymous package is a public-safe audit projection of the frozen hosted
run.  It includes the exact system prompts and user-prompt construction
template, executed runner snapshot, sanitized run configuration and manifests,
aggregate metrics, scientific gates, source/output hashes, and a pseudonymized
case-level stage ledger.

## Contents

| Path | Contents |
| --- | --- |
| `prompts.json` | System prompt and user-prompt construction template. |
| `hosted_run/` | The reported run `hosted_round3_cc0_20260810T004042Z`: executed runner snapshot, sanitized run configuration, pre-call manifest, aggregate metrics, paired-displacement analysis, pricing snapshot, scientific gates, and pre/post credential scans. |
| `offline_replay/` | Frozen-response end-to-end replay: metrics, integrity gates, manifest, and the projected per-call stage ledger. |
| `cluster_analysis/` | **Superseded, retained as a record.** Offline clustered reanalysis of the earlier 126-call hosted stress run. No result reported in the manuscript comes from it; see `cluster_analysis/README.md`. |
| `MANIFEST.sha256`, `verify_archive.py` | SHA-256 manifest of this package and its verifier. |
| `PACKAGE_METADATA.json` | Per-member packaged and source digests. |

## What is withheld, and what is not

Withheld from this package: raw model responses, free-text predictions,
`cases.jsonl`, validation diagnostics that reproduce model-generated entity
text, injection-pair entity sets, request identifiers, populated ontologies,
and case-level logs and artifacts.  The stage ledger's SHA-256 digests
establish response identity without republishing the text.

**The evaluation corpus is not withheld.**  The reported hosted run uses 16
author-authored contexts dedicated under CC0-1.0, and they are published in
full -- with gold annotations, manifest, design ledger and dedication -- at
`benchmark/round3_cc0/` in the artifact repository and at
`5_Data_and_Experiments/acceptance_loop/round3_cc0_benchmark_v2/` in supplement
S1.  Nothing here is held back for want of a licence to the evaluation
contexts.

The one third-party corpus this package does not carry is CTI-HAL, which the
manuscript uses for related work and which supplied the contexts of the
superseded stress run projected in `cluster_analysis/`.  Third-party licensing
for hosted processing of CTI-HAL was not established, so CTI-HAL text is not
redistributed here; hashes and source coordinates remain the audit bridge to an
independently obtained pinned checkout.  This bears on the superseded
reanalysis only, not on the reported round-3 run.

The projected ledger preserves condition, repeat, source/response hashes,
pipeline stage statuses, scalar entity/relation counts, runtime, and no-hosted-
call evidence.  `prompts.json` uses `{{CTI_CONTEXT}}` as the context redaction
and `{{INJECTION_NOTE}}` as the injection-note redaction;
`executed_run_hosted_round3_cc0.py` is the exact historical prompt constructor.
`paired_displacement.json` reports the paired clean-vs-injected displacement
analysis (status, canonical graph, gold-score) for the 48 matched pairs.

Verify an extracted package with:

```bash
python3 verify_archive.py .
```

These data measure response-contract behavior and offline pipeline attrition,
not factual extraction accuracy.
