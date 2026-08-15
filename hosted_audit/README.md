# AutoMalKG S2 public hosted audit

This anonymous package is a public-safe audit projection of the frozen hosted
run.  It includes the exact system prompts and user-prompt construction
template, executed runner snapshot, sanitized run configuration and manifests,
aggregate metrics, scientific gates, source/output hashes, and a pseudonymized
case-level stage ledger.

Excluded by design: CTI-HAL text, source excerpts, cases.jsonl, raw responses,
predictions, validation diagnostics that reproduce model-generated entity text,
injection-pair entity sets, request identifiers, populated ontologies, and
case-level logs/artifacts.  CTI-HAL licensing/redistribution permission was not
established by this build, so hashes and source coordinates remain the audit
bridge to an independently obtained pinned dataset checkout.

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
