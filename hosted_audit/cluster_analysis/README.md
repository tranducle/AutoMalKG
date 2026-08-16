> **Superseded -- retained as a record.**
>
> This reanalysis belongs to the earlier hosted stress run
> `hosted_stress_20260809T211342Z`: 126 calls over 14 sampled CTI-HAL contexts
> in 7 groups.  It is superseded by `hosted_round3_cc0_20260810T004042Z`, the
> run projected in `../hosted_run/`: 144 calls over 16 author-authored CC0-1.0
> contexts.  Round 3 replaced the CTI-HAL sample with a corpus whose licensing
> permits publication, and it is the run every hosted result in the manuscript
> reports.  No number below is reported in the manuscript.  The reanalysis is
> retained so that the earlier analysis, and the change of corpus, stay
> auditable.

# Hosted Clustered Reanalysis

This is an offline reanalysis of the frozen 126-call hosted run. It issued **no
hosted/model/network calls** and did not repair or replace any stored response.

## Inputs

- Hosted source run: `5_Data_and_Experiments/acceptance_loop/runs/hosted_stress_20260809T211342Z`
- Semantic fixture run: `5_Data_and_Experiments/acceptance_loop/runs/semantic_benchmark_seed20260809_comprehensive`
- Required hosted input hashes were checked against the frozen source manifest.

## Analysis contract

- Unit of clustering: 14 sampled CTI contexts (7 groups, 2 contexts/group).
- Nested observations: 3 repeats/context/condition.
- Valid fraction: calls with stored status exactly `valid`, divided by 3 in each context.
- Uncertainty: deterministic paired percentile bootstrap resampling whole contexts,
  10,000 replicates. The interval is a sensitivity summary over these observed
  contexts, not a population confidence interval.
- Semantic comparison: ID-insensitive normalized entity `(type, name)` sets and
  relation `(resolved subject, predicate, resolved object)` sets. Empty/empty
  Jaccard is defined as 1. Evidence text is represented by SHA-256-bearing owner
  records in the pair ledger.
- Schema expressibility is not scored: no response-independent validated mapping
  from these CTI-HAL contexts to full MALOnt gold exists.

## Exact descriptive results

- Valid calls: minimal JSON 0/42; typed evidence
  40/42; typed evidence + injection 40/42.
- Clean typed vs injected pairs: 42 aligned,
  27 exact-prediction equal,
  15 with any exact-prediction
  displacement, and 13 with changed canonical
  semantic sets (including 10
  clean-nonempty to injected-empty transitions).
- Mean combined-set Jaccard: 0.714286.
- Canary success/presence: 0 frozen-definition
  successes; 0 raw responses and
  0 parsed predictions contained the canary.

## Design limitations

The minimal-to-typed contrast bundles ontology type restrictions, relation
signatures, evidence requirements, empty-output guidance, and untrusted-data
instructions. Conditions also ran in a fixed, non-counterbalanced order. The
paired differences therefore cannot isolate prompt components or temporal
effects. Output displacement under injection does not establish factual
corruption or unsupported facts because task-matched factual gold is absent.

## Reproduce

From the paper root:

```bash
python src/reanalyze_hosted_clusters.py \
  --source-run 5_Data_and_Experiments/acceptance_loop/runs/hosted_stress_20260809T211342Z \
  --semantic-run 5_Data_and_Experiments/acceptance_loop/runs/semantic_benchmark_seed20260809_comprehensive \
  --output 5_Data_and_Experiments/acceptance_loop/runs/hosted_clustered_reanalysis_20260809
```

The command refuses to overwrite a non-empty output directory unless
`--overwrite` is given.
