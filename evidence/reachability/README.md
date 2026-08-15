# Offline stage ablation and OWL reachability

This is a **fixture-only offline capability ablation**. It reads no hosted cases or raw responses and makes zero hosted calls.

## Exact replay

```bash
python3 src/run_stage_reachability_ablation.py --fixtures-only --output-dir reproduced_runs/reachability
```

The runner refuses to overwrite an existing directory. `manifest.json` hashes the source ledgers, observations, gates, logs, environment freeze, and executed source snapshot.

## Fixture comparison

| Method | Accepted | Rejected | Silent known violations | Provenance completeness | Proof capability |
|---|---:|---:|---:|---:|---:|
| json_parse_only | 26 | 3 | 23 | 0.0 | 0.0 |
| strict_json_schema | 16 | 13 | 13 | 0.0 | 0.0 |
| production_canonicalization | 3 | 26 | 0 | 1.0 | 0.0 |
| full_automalkg | 3 | 26 | 0 | 1.0 | 1.0 |

## Hosted-input scope

No hosted inputs were read or evaluated in this execution scope.

## Reachability result

The current OWL vocabulary has 17 tested inconsistency classes: 0 are reachable from schema-valid/type-aware production input, while 17 are reachable under explicit reasoner-layer injection. The broad `Targets.domain = Entity` axiom is non-discriminating and remains consistent for a ThreatActor subject; the production extraction signature is stricter.

## Claim boundary

The evidence supports failure-localization and mechanism-coverage statements only. It does not establish provider-native constrained-output performance, factual CTI accuracy, statistical superiority, or production latency.

Scientific gates: G1=PASS, G2=PASS, G4=PASS, G6=PASS. Baseline-fairness verdict: PASS_WITH_WARNINGS (capability claims only).
