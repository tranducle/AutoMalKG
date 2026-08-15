# Round-3 CC0 controlled benchmark design ledger

## Frozen design

- Benchmark: `automalkg-round3-cc0-controlled-v2`
- Contexts: 16 original author-written synthetic texts
  (2 of them empty-gold negative controls)
- Gold: 38 entities and 24 relations, frozen before calls
- Coverage: all five supported entity types and all four directed relation signatures
  (coverage is required only on the 14 non-negative-control contexts)
- Families: `{"campaign": 1, "campaign_indicator": 1, "indicator": 2, "multi_signature": 2, "negative_control_empty": 2, "targets": 1, "targets_indicator": 1, "uses": 2, "uses_campaign": 2, "uses_indicator": 1, "uses_targets": 1}`
- Conditions: minimal JSON; typed evidence; typed evidence plus an original simulated untrusted note
- Repeats: 3 per context/condition (144 hosted calls total)
- Order: deterministic Latin rotation; each condition occupies each within-context ordinal once across repeats
- Temperature: 0.0
- Gold policy: the injected note never changes the clean context or its gold graph
- Protocol version: `confirmatory-v1.1`
- Pilot-driven pre-confirmatory amendment: Two diagnostic pilots were stopped early. The 1,536-token pilot exposed empty-content responses under an insufficient cap. The 4,096-token pilot exposed ambiguity between a relation signature and the predicate token, plus a historical-global-canary evaluator defect. Before confirmatory calls, the prompt was amended with an exact formatting example and an explicit predicate-token rule; attack evaluation was changed to the case-local canary. Context prose and all gold annotations remained unchanged.

## Analysis unit

The 16 context clusters are the inferential/descriptive unit of analysis. The
three prompt conditions and three repeats are nested within each cluster, so
the 144 requests are NOT 144 independent samples. All gold-score and paired
displacement reporting must use per-cluster nested summaries, not treat the
144 requests as independent observations.

## Empty-gold negative controls

The two empty-gold negative controls (`r3c15`, `r3c16`) carry gold
`{"entities": [], "relations": []}` and are marked `negative_control: true`
and `gold_empty: true`. They test empty-gold false-positive behavior: a model
that emits non-empty entities/relations when the gold is empty registers as
false positives. They are excluded from the entity-type / relation-signature
coverage requirement.

## Construct boundary

Exact entity/relation set metrics measure performance only on this deliberately
authored controlled design set. They are not estimates of performance on real
CTI, do not define a population sampling frame, and do not license claims of
generalization, deployment readiness, or operational robustness.

## Governance rubric

1. No context was copied or paraphrased from third-party CTI.
2. Names are fictitious; IPv4/domain strings use documentation reservations;
   CVE-like identifiers use the fictitious year 2099.
3. No personal, sensitive, credential, customer, or live-incident data occur.
4. Every gold evidence string is a non-empty exact substring of its clean context
   (the empty-gold negative controls have no gold evidence).
5. Gold passes the production extraction schema and covers all supported
   types/signatures (the 14 non-negative-control contexts).
6. Source, context, and gold hashes are frozen before hosted calls.
7. CC0 applies to the authored dataset; hosted model outputs remain separate audit data.

## Reproduction

```bash
python src/generate_round3_cc0_benchmark.py \
  --source 5_Data_and_Experiments/acceptance_loop/round3_cc0_benchmark_v2/author_source.json \
  --output-dir 5_Data_and_Experiments/acceptance_loop/round3_cc0_benchmark_v2
```
