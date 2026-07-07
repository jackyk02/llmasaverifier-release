# Directory Structure

```text
.
├── run.py                       # registry-driven launcher
├── terminal_bench_progress.py   # re-score + plot the progress-tracking example
├── add_new_benchmark.md         # paste-into-Claude-Code recipe for new tasks
├── criteria/                    # verifier criteria + ground-truth notes
│   ├── TEMPLATE.md              #   copy this to write your own
│   ├── terminal_bench.md
│   ├── swe_bench.md
│   └── medagentbench.md
├── llm_verifier/                # the reusable framework (import llm_verifier)
│   ├── __init__.py              #   llm_verifier.select(...) / .compare(...)
│   ├── __main__.py              #   python -m llm_verifier <file.md>: preview criteria
│   ├── benchmarks.py            #   BENCHMARKS registry (one Benchmark / launch)
│   ├── fine_grained_reward.py   #   R(x,τ): Gemini logprob scoring + cache
│   ├── progress.py              #   llm_verifier.track(...): per-step progress curve
│   ├── pivot_tournament.py      #   PPT: O(Nk²) selection (Bradley-Terry)
│   ├── prompts.py               #   load criteria/*.md + normalize criteria args
│   └── loaders.py               #   per-benchmark trajectory loaders
├── data/                        # agent trajectories per benchmark
├── cache/                       # verifier score caches (written per run)
├── results/                     # result tables (written after each run)
└── docs/                        # this documentation site (Sphinx)
```

## Where each concept lives

| Concept | Documentation | Implementation |
|---|---|---|
| Fine-grained reward $R(x,\tau)$ | [Fine-Grained Reward Estimation](../advanced_features/fine_grained_reward.md) | `llm_verifier/fine_grained_reward.py` |
| Best-of-N selection | [Best-of-N Selection](../advanced_features/best_of_n_at_scale.md) | `llm_verifier/__init__.py` (`select`) |
| Probabilistic Pivot Tournament | [Probabilistic Pivot Tournament](../advanced_features/pivot_tournament.md) | `llm_verifier/pivot_tournament.py` |
| Progress tracking | [Progress Tracking](../basic_usage/progress_tracking.md) | `llm_verifier/progress.py` |
| Criteria files | [Writing Verifier Criteria](../basic_usage/criteria.md) | `llm_verifier/prompts.py`, `criteria/` |
| Benchmark registry | [Running the Bundled Benchmarks](../benchmarks/running_benchmarks.md) | `llm_verifier/benchmarks.py`, `run.py` |
