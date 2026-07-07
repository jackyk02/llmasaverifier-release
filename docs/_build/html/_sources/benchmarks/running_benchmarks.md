# Running the Agentic Benchmarks

The repository ships three agentic benchmarks with their agent trajectories included under `data/`, so you can reproduce the results end-to-end.

## Expected results

All benchmark results use **Gemini 2.5 Flash** (`gemini-2.5-flash`, the default model) as the verifier.

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

For each benchmark, `run.py` loads the trajectories and criteria, splits tasks into *all-pass* / *all-fail* / *swing* (only swing tasks — where the N trials disagree — need verification), runs a [Probabilistic Pivot Tournament](../advanced_features/pivot_tournament.md) per swing task, and reports **Pass@1 vs. LLM-as-a-Verifier vs. Oracle**.
Verifier scores are cached under `cache/` (re-runs are incremental) and result tables are written under `results/`.

## The benchmark registry

Benchmarks are typed `Benchmark` dataclasses in `llm_verifier/benchmarks.py` — add or tweak one there:

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
