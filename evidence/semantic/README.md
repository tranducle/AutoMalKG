# AutoMalKG semantic benchmark

Scope: deterministic OWL consistency detection, SWRL positive/negative controls, and local reasoner scaling. This run does not estimate natural-language extraction accuracy or external validity.

Replay: `/opt/homebrew/opt/python@3.14/bin/python3.14 ./src/run_semantic_benchmark.py --output-dir ./5_Data_and_Experiments/acceptance_loop/runs/semantic_benchmark_seed20260809_comprehensive --seed 20260809 --consistency-cases-per-class 25 --swrl-controls-per-class 25 --scale-sizes 100,300,1000,3000 --scale-repeats 10 --java /opt/homebrew/Cellar/openjdk/26.0.1/libexec/openjdk.jdk/Contents/Home/bin/java`

Artifacts: `cases.jsonl`, `predictions.jsonl`, `metrics.json`, `failures.jsonl`, `G3_method_mechanism.json`, and raw `pellet_logs/*.log`.
UNVERIFIED outcomes remain explicit and are excluded from accuracy denominators.
