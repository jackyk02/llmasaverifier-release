# Running the Bundled Benchmarks

The repository ships three test-time scaling benchmarks with their agent trajectories included under `data/`, so you can reproduce the headline results end-to-end.

## Expected results

All benchmark results use **Gemini 2.5 Flash** (`gemini-2.5-flash`, the default verifier model) as the verifier.

| Benchmark | Base Model | Harness | Pass@1 | LLM-as-a-Verifier | Oracle |
|---|---|---|---|---|---|
| Terminal-Bench 2.0 | GPT-5.5 (×5) | Capy | 83.1% | **86.5%** | 92.1% |
| SWE-bench Verified | Opus 4.5 / Opus 4.6 / Gemini 3 Flash | mini-swe-agent | 76.1% | **78.2%** | 84.4% |
| MedAgentBench | Claude Opus 4.8 (×5) | AgentBench | 70.2% | **73.3%** | 75.0% |

## Setup

```bash
pip install google-genai tqdm
```

Create a `.env` file with your Vertex AI API key (required for logprob extraction):

```bash
echo "VERTEX_API_KEY=your_key_here" > .env
```

## Run a benchmark

Run a benchmark by name (`python run.py` with no argument lists them):

```bash
python run.py terminal_bench
python run.py swe_bench
python run.py medagentbench
```

The tournament defaults can be overridden on the command line:

```bash
python run.py swe_bench --pivots 2 --n-evaluations 8 --seed 0 --max-workers 50
```

## What the launcher does

`run.py` is a registry-driven launcher. For each benchmark it:

1. Loads the benchmark trajectories (`llm_verifier/loaders.py`).
2. Loads the verifier criteria and ground-truth note (`criteria/<benchmark>.md`).
3. Splits tasks into *all-pass* (every trial succeeds — no verification needed), *all-fail* (unwinnable), and *swing* tasks (the N trials disagree). For every swing task it runs a [Probabilistic Pivot Tournament](../advanced_features/pivot_tournament.md): a random ring pass, pivots = empirical leaders, then pivot rounds. Only the directed pairs the tournament needs are scored, with caching (`llm_verifier/fine_grained_reward.py`).
4. Reports **Pass@1 vs. LLM-as-a-Verifier vs. Oracle**.

Scoring is two-phase because the pivots depend on the ring-pass results: first all ring pairs are scored, then pivots are chosen, then the pivot rounds are scored.
Verifier scores are cached under `cache/` (re-runs are incremental) and result tables are written under `results/`.

## The benchmark registry

Benchmarks are defined in `llm_verifier/benchmarks.py` — add or tweak one there.
Each benchmark is a typed `Benchmark` dataclass rather than a YAML file: defaults live in one place, a bad field is a type/attribute error instead of a silent `None`, and paths/criteria are checked by your editor.

```python
@dataclass
class Benchmark:
    name: str                       # human-readable title shown in the report
    loader: str                     # key into llm_verifier.loaders.LOADERS
    prompts: str                    # criteria name (criteria/<name>.md) or a path
    data: dict                      # loader-specific data locations
    cache: str                      # path to the verifier-score cache (JSON)
    results: str                    # path to write the result table
    criteria: list                  # criterion ids, in order
    n_evaluations: int = 8        # repeated verifications K per criterion
    pivots: int = 2                 # number of pivots k in the tournament
    seed: int = 0                   # seed for the random ring pass
```

CLI flags (`--pivots`, `--n-evaluations`, `--seed`) override the registry values at launch time.

To stand up a verifier for a *new* task, see [Adding a New Benchmark](add_new_benchmark.md).
