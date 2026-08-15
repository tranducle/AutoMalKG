# Hosted-response end-to-end replay

## Scope

This run replays all 126 frozen response texts from
`5_Data_and_Experiments/acceptance_loop/runs/hosted_stress_20260809T211342Z` through the production extractor,
canonicalizer, ontology loader, real Pellet reasoner, inference/proof ledger,
and fail-closed serializer. It makes zero hosted calls and does not repair any
response. CTI-HAL is not MALOnt factual ground truth, so these results measure
pipeline integration attrition, not extraction accuracy.

## Attrition

| Stage | Cases passing/completing |
|---|---:|
| Source/raw hash verified | 126 |
| JSON parse | 126 |
| Schema validation | 80 |
| Canonicalization completed (including empty) | 80 |
| Non-empty canonical graph | 30 |
| Ontology load validated | 30 |
| Pellet consistent | 30 |
| Inference ledger emitted | 30 |
| Verified artifact serialized | 30 |

Extraction status distribution: `{"empty": 50, "schema_invalid": 46, "success": 30}`.
Inferred relations: 0; with proof ledgers:
0.

## Reproduction

Run from the paper root:

```bash
python src/run_hosted_end_to_end_replay.py \
  --source-run 5_Data_and_Experiments/acceptance_loop/runs/hosted_stress_20260809T211342Z \
  --output-dir 5_Data_and_Experiments/acceptance_loop/runs/hosted_end_to_end_replay_20260809
```

The runner refuses to overwrite an existing output directory. Hashes for the
source ledgers, runner, production modules, logs, and all serialized artifacts
are recorded in `manifest.json`. Both the replay-specific logical G6 gate and
the repository-standard numeric G6 gate are included in the package.
