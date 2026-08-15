# AutoMalKG

**Auditable failure contracts for LLM-to-OWL malware knowledge graph pipelines.**

> **Anonymous artifact for double-blind review.** This repository is the review artifact for a
> manuscript under submission. It contains no author names, credentials, raw model responses,
> manuscript sources, or figures. Author and citation metadata will be added upon acceptance.

## What AutoMalKG is

Hosted large language models (LLMs) are attractive for populating cyber threat intelligence (CTI)
knowledge graphs, but fluent output does not guarantee a valid schema, stable entity identity, or
factual correctness. AutoMalKG treats the LLM as an **untrusted parser** and places a fail-closed,
typed structural-validation contract between probabilistic extraction and formal OWL reasoning:

- **Typed response contract.** Every hosted response must parse as JSON and pass a strict typed
  schema (five entity classes, four relation signatures, no unknown keys, no duplicate IDs, no
  dangling endpoints). Parse and provider failures are data-quality outcomes, never recoded as
  valid empty graphs.
- **Deterministic identity and provenance.** Unicode NFKC normalization, case folding, and
  SHA-256-prefixed type-aware identifiers merge repeated mentions; every accepted fact retains
  its source chunk, span, and hashes.
- **Fail-closed OWL consistency.** Canonical graphs load into an isolated OWL 2 world checked by
  the Pellet reasoner. Only a `CONSISTENT` outcome reaches verified serialization; `INCONSISTENT`
  and `UNVERIFIED` stop the pipeline with typed diagnostics.
- **Auditable rule inference.** One SWRL rule (actor--Uses-->malware--Targets-->vulnerability)
  materializes an actor-to-vulnerability edge only when both asserted premises are present, and
  every conclusion carries a two-premise proof ledger traceable to source chunks.

A central design boundary: **structural consistency is not factual truth.** The contract localizes
where and how extraction fails; it does not certify content. Well-typed disinformation passes
every structural check.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/` | The complete Python implementation: contract validation, canonical merge, Pellet integration, benchmark/runners for every experiment, and the supplement builder. |
| `tests/` | The 107-test regression suite (typed failures, reasoner states, identity collisions, provenance union, frozen-response replay guards, staged ablation, injection isolation). |
| `ontology/` | The code-exported OWL/SWRL schema (`automalkg_schema.owl`). |
| `benchmark/round3_cc0/` | The 16 CC0-1.0 evaluation contexts with gold annotations, manifest, design ledger, and provenance statement. All names are fictitious; IPs use IANA documentation ranges. |
| `evidence/semantic/` | Seeded OWL-consistency and SWRL-completion controls (25+25 cases each) with observations, metrics, and raw Pellet logs. |
| `evidence/reachability/` | The 29-fixture offline stage ablation (26 violations + 3 controls), the 17-class OWL contradiction reachability matrix, and per-fixture Pellet logs. |
| `hosted_audit/` | The public-safe audit projection of the frozen hosted run: exact prompts, executed runner snapshot, pre/post credential scans, pricing snapshot, per-call stage ledger (pseudonymized), paired-displacement analysis, and offline replay metrics. Self-verifying with its own manifest. |
| `requirements.txt` | Pinned direct dependencies for the artifact. |
| `MANIFEST.sha256`, `verify_archive.py` | SHA-256 manifest of the tree and its verifier. |

## Requirements

- Python 3.12+ (validated on CPython 3.12 and 3.14)
- A Java runtime on `PATH` (required by Owlready2/Pellet for reasoning; the suite exercises the
  typed `reasoner_unavailable` path when Java is absent)

```bash
python3 -m pip install -r requirements.txt
```

## Quick start

From the repository root:

```bash
# 1. Verify the tree against its manifest
python3 verify_archive.py .

# 2. Run the full regression suite (107 tests)
python3 -m pytest -q

# 3. Reproduce a minimal semantic benchmark (deterministic, seeded)
python3 src/run_semantic_benchmark.py \
    --output-dir reproduced_runs/semantic-minimal \
    --seed 20260809 \
    --consistency-cases-per-class 1 --swrl-controls-per-class 1 \
    --scale-sizes 10 --scale-repeats 1

# 4. Reproduce the fixture-only stage ablation and reachability matrix
python3 src/run_stage_reachability_ablation.py \
    --fixtures-only --output-dir reproduced_runs/reachability
```

Runners refuse to overwrite an existing output directory. Runtime values are environment
sensitive; contract and status outcomes are the reproducibility target.

## What the artifact reproduces

| Manuscript result | Where it comes from |
| --- | --- |
| 107/107 regression tests pass | `python3 -m pytest -q` |
| Parse-only / strict JSON Schema / production acceptance of the 26 encoded violations (23 / 13 / 0) | `run_stage_reachability_ablation.py --fixtures-only`; recorded evidence in `evidence/reachability/` |
| 0/17 OWL contradiction classes reachable from production-valid input; 17/17 by direct reasoner injection | reachability matrix in `evidence/reachability/` |
| Pellet classifies 25/25 inconsistent and 25/25 consistent seeded controls; SWRL P/R/F1 = 1.00 with complete two-premise ledgers | `run_semantic_benchmark.py`; recorded evidence in `evidence/semantic/` |
| Hosted validity 0/48, 47/48, 48/48; canary 0/48; paired displacement analysis | recorded in `hosted_audit/` (not re-executable here; see below) |
| Frozen-response end-to-end attrition 95 schema-valid, 5 empty, 90 serialized, 12 Rule 1 inferences | recorded in `hosted_audit/offline_replay/` stage ledger |

## Hosted experiment: what is and is not included

The hosted study issued 144 hash-locked calls (16 CC0 contexts x 3 conditions x 3 repeats,
temperature 0, single pinned route) to one dated model checkpoint through a paid API account.
This repository includes everything needed to audit and re-issue the run except the raw responses
themselves:

**Included** (`hosted_audit/`): the exact system and user prompt construction (`prompts.json`,
`executed_run_hosted_round3_cc0.py`), sanitized run configuration and pre-call manifest, per-call
stage ledger with condition/repeat/status/hashes/counts, aggregate and paired-displacement
metrics, pricing snapshot, scientific gate reports, and pre/post credential scans (both PASS with
zero findings).

**Excluded by design:** raw model responses and free-text predictions (bulk redistribution of a
provider's model outputs is not clearly permitted by the terms under which they were obtained,
and the SHA-256 digests in the stage ledger already establish response identity without
republishing text); API credentials (never present); CTI-HAL source text (third-party licensing
for hosted processing was not established; the manuscript's related work uses it for context
only). A holder of the raw responses can verify each one against the ledger digests; the authors
retain the complete raw responses and will supply them to editors and reviewers on request.

To re-issue the hosted run with your own credentials and account:

```bash
export OPENROUTER_API_KEY=...   # never committed; see .gitignore
python3 src/run_hosted_round3_cc0.py --help
```

The runner is read-only against `benchmark/round3_cc0` and writes a fresh hash-locked run
directory under `reproduced_runs/`.

## Benchmark provenance

The 16 evaluation contexts, all gold annotations, fictitious entity names, and the injected-note
text were newly authored for this study on 2026-08-09 and are not copied or paraphrased from any
third-party CTI source (statement and CC0-1.0 dedication in `benchmark/round3_cc0/`). Names are
fictitious; domains and IPv4 addresses use IANA documentation ranges; CVE-like identifiers use
the fictitious year 2099. No personal, sensitive, customer, credential, or operational incident
data is present.

Third-party resources referenced by the implementation but not redistributed here:

- **MALOnt** (malware ontology vocabulary alignment) - https://github.com/aiforsec/MALOnt
- **CTI-HAL** (human-annotated CTI dataset; not used for hosted processing here) -
  paper record at https://doi.org/10.1109/EuroSPW67616.2025.00014
- **Pellet** (delivered inside the OWL API distribution used by Owlready2) -
  https://github.com/stardog-union/pellet

## Responsible use

The failure taxonomy in `evidence/reachability/fixtures.jsonl` is a defensive instrument: it lets
a team measure which malformed record shapes its own validator admits. The indirect-injection
note included in `benchmark/round3_cc0/author_source.json` is a single-turn instruction in a
dedicated untrusted channel, already documented in the prompt-injection literature and in
provider guidance; the specific wording confers no additional capability. Well-typed
disinformation passes every structural check in this pipeline, and the proof ledger traces a
fabricated record as faithfully as a correct one: a serialized, reasoner-consistent graph is not
vetted intelligence. Coordinated disclosure was judged inapplicable because the material
identifies no defect in any third-party product.

## Integrity and license

`MANIFEST.sha256` covers every file in this tree (`hosted_audit/` carries its own internal
manifest; verify it with `python3 verify_archive.py hosted_audit` from the repository root or
`python3 verify_archive.py .` inside it). Code and benchmark materials are released under
CC0-1.0 (see `LICENSE`), matching the license the benchmark contexts already carry.
