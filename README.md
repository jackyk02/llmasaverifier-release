<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <img alt="LLM-as-a-Verifier" src="figures/logo.png" width=78%>
  </picture>
</p>

<h3 align="center">
Any modality, Many Applications, One Unified Verification Framework
</h3>

<p align="center">
| <a href="https://llm-as-a-verifier.ai"><b>Website</b></a> | <a href="https://docs.llm-as-a-verifier.ai"><b>Documentation</b></a> | <a href="https://arxiv.org/"><b>Paper</b></a> | <a href="https://blog.llm-as-a-verifier.ai"><b>Blog</b></a> | <a href="https://x.com/"><b>Twitter/X</b></a> | <a href="https://slack.llm-as-a-verifier.ai"><b>Slack</b></a> |
</p>

🔥 LLM-as-a-Verifier achieves SOTA performance across agentic benchmarks, including Terminal-Bench V2, SWE-Bench Verified, MedAgentBench, RoboRewardBench and more. We invite the community to contribute more use cases!


---

## Installation

```bash
pip install llm-verifier
```

To install the latest from a clone:

```bash
pip install -e .
```

---

## About

LLM-as-a-Verifier is a general-purpose framework that provides **fine-grained
feedback** for any agent. The key idea is simple: 1) use fine-grained scoring
granularity, 2) take the expectation over the full logprob distribution of LLM
score tokens, and 3) scale repeated evaluation and criteria decomposition. The
resulting fine-grained feedback can be used for test-time scaling, progress
tracking, and reinforcement learning.

<p align="center">
  <img src="figures/llmoverview.png" alt="LLM-as-a-Verifier overview" width="100%">
</p>

---

## Quickstart

### Select the best of N agent trajectories

Given a task and a pool of agent trajectories, pick the best one in a few
lines of code.

```python
import llm_verifier

problem = "Fix the failing test in utils.py."
trajectories = [traj_1, traj_2, traj_3, traj_4, traj_5]  # N trajectories

result = llm_verifier.select(
    problem=problem,
    trajectories=trajectories,
    criteria={"Root cause": "Did the agent fix the real cause?",
              "Verification": "Did the agent confirm the fix?"},
    model="gemini-2.5-flash",          # verifier model (needs VERTEX_API_KEY for logprobs)
    n_verifications=4,                 # repeated evaluations per criterion
    pivots=2,                          # pivots < N; reduced verification cost
)

print("Best trajectory:", result.index)           # result.best is the trajectory itself
print("Ranking:", result.ranking)                 # all trajectories, best-first
```

Under the hood, `select` runs the
[Probabilistic Pivot Tournament](#probabilistic-pivot-tournament) to rank all
`N` trajectories using `O(Nk²)` pairwise verifications instead of a full
`O(N²)` round-robin. `pivots` trades cost for accuracy: more pivots = more
comparisons = higher accuracy.

### Score a pair of trajectories directly

`select` is built on a pairwise reward model. For the raw fine-grained rewards
of a single comparison, call `compare`:

```python
r_a, r_b = llm_verifier.compare(
    problem, trace_a, trace_b,
    criteria={"Overall": "Did the agent solve the task?"},
)
print(r_a, r_b)   # fine-grained rewards in [0, 1]
```
---

## Test-Time Scaling with LLM-as-a-Verifier

Each benchmark ships with its agent trajectories (`data/`). Expected results:

| Benchmark | Base Model | Harness | Pass@1 | LLM-as-a-Verifier | Oracle |
|---|---|---|---|---|---|
| Terminal-Bench 2.0 | GPT-5.5 (×5) | Capy | 83.1% | **86.5%** | 92.1% |
| SWE-bench Verified | Opus 4.5 / Opus 4.6 / Gemini 3 Flash | mini-swe-agent | 76.1% | **78.2%** | 84.4% |
| MedAgentBench | Claude Opus 4.8 (×5) | AgentBench | 70.2% | **73.3%** | 75.0% |

### Setup

```bash
pip install google-genai tqdm
```

Create a `.env` file with your Vertex AI API key (required for logprob
extraction):

```bash
echo "VERTEX_API_KEY=your_key_here" > .env
```

Run a benchmark by name (`python run.py` with no argument lists them):

```bash
python run.py terminal_bench
python run.py swe_bench
python run.py medagentbench
```

The tournament defaults can be overridden on the command line:

```bash
python run.py swe_bench --pivots 2 --n-verifications 8 --seed 0 --max-workers 50
```

Benchmarks are defined in `llm_verifier/benchmarks.py` — add or tweak one there.

---

## Adapt LLM-as-a-Verifier for your own use case

Use the verifier for your own task in three steps — Claude Code does the rest
(generates the criteria, writes a runner, and selects the best-of-N for you):

1. **Add your data.** Copy your agent trajectories into `data/task_name_trajs/`.
2. **Update naming.** Replace every `task_name` in
   [`add_new_benchmark.md`](add_new_benchmark.md) with the name of your task.
3. **Spin up Claude Code in this repo** (or Codex, or whatever you like — with
   permissions disabled) and paste the contents of `add_new_benchmark.md` to let
   it run.

---

## Directory Structure

```
.
├── run.py                       # registry-driven launcher
├── criteria/                    # verifier criteria + ground-truth notes
│   ├── TEMPLATE.md              #   copy this to write your own
│   ├── terminal_bench.md
│   ├── swe_bench.md
│   └── medagentbench.md
├── llm_verifier/                  # the reusable framework (import llm_verifier)
│   ├── __init__.py              #   llm_verifier.select(...) / .compare(...)
│   ├── __main__.py              #   python -m llm_verifier <file.md>: preview criteria
│   ├── benchmarks.py            #   BENCHMARKS registry (one Benchmark / launch)
│   ├── fine_grained_reward.py   #   R(t,τ): Gemini logprob scoring + cache
│   ├── pivot_tournament.py      #   PPT: O(Nk²) selection (Bradley-Terry)
│   ├── prompts.py               #   load criteria/*.md + normalize criteria args
│   └── loaders.py               #   per-benchmark trajectory loaders
├── data/                        # agent trajectories per benchmark
├── cache/                       # verifier score caches (written per run)
└── results/                     # result tables (written after each run)
```

---

## How it works

### Fine-grained Reward Estimation

Rather than reducing each distribution into a single discrete score (as in
LLM-as-a-Judge), LLM-as-a-Verifier approximates the reward of a trajectory
$\tau$ on task $x$ as:

$$
R(x, \tau)
= \frac{1}{CK} \sum_{c=1}^{C} \sum_{k=1}^{K}
\sum_{g=1}^{G} p_{\theta}(v_g \mid x, c, \tau)\,\phi(v_g)
$$

- $C$ = number of evaluation criteria
- $K$ = number of repeated verifications
- $G$ = number of score tokens (granularity level)
- $p_{\theta}(v_g \mid x, c, \tau)$ = probability assigned by model $\theta$ to score token $v_g$
- $\phi(v_g)$ = maps each scoring token to a scalar value
- $V_{\text{score}} = \{v_1, \ldots, v_G\}$ = ordered set of discrete score tokens

This lives in `llm_verifier/fine_grained_reward.py`.

### Probabilistic Pivot Tournament

<p align="center">
  <img src="figures/pivot_tournament.png" alt="Probabilistic Pivot Tournament" width="100%">
</p>

To pick the best of `N` candidate trajectories, a round-robin tournament scores
all $\binom{N}{2}$ pairs — `O(N²)`. Probabilistic Pivot Tournament (PPT) is a
cost efficient ranking algorithm in which every candidate is compared only
against a small set of pivots, reducing the budget from $\mathcal{O}(N^2)$ to
$\mathcal{O}(Nk^2)$.

1. **Candidates:** the pool $\{\tau_1,\dots,\tau_N\}$ to be ranked.
2. **Ring pass:** a random Hamiltonian cycle scores the $N$ adjacent pairs so
   every candidate appears once in the "A" slot and once in "B", canceling
   the model's positional bias.
3. **Pivot selection:** candidates are ranked by their ring-pass scores
   $w_{(i)}$, and the top-$k$ candidates form the pivot set $\mathcal{P}$.
4. **Pivot tournament:** every *non-pivot–vs–pivot* and *pivot–vs–pivot* pair
   is scored via the pairwise preference
   $p(a \succ b) = \sigma(R_a - R_b)$, concentrating the budget on uncertain
   top candidates and cutting cost from $\mathcal{O}(N^2)$ to
   $\mathcal{O}(Nk^2)$.
5. **Selection:** comparisons are aggregated into win mass $w_i$ and count
   $c_i$, and the candidate with the highest normalized $w_i/c_i$ is
   returned.

This lives in `llm_verifier/pivot_tournament.py`.
