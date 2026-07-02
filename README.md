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
| <a href="https://llm-as-a-verifier.com"><b> Website</b></a> | <a href="https://arxiv.org/"><b>Paper</b></a> | <a href="https://blog.llm-as-a-verifier.ai"><b>Blog Post</b></a> | <a href="https://x.com/"><b>Twitter/X</b></a> | <a href="https://slack.llm-as-a-verifier.ai"><b>Slack</b></a> |
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

### Simple Best-of-N Selection

Run a first end-to-end selection (requires
`VERTEX_API_KEY` in `.env`):

```python
import llm_verifier

problem = "Write a function that reverses a string."
candidates = [
    "def rev(s): return s[::-1]",
    "def rev(s): return s",
    "def rev(s): return ''.join(sorted(s))",
]

result = llm_verifier.select(
    problem=problem,
    candidates=candidates,
    criteria={"Correctness": "Does the code actually reverse the string?"},
)
print(result.index)   # index of the best candidate: 0
print(result.scores)  # candidate scores: [0.73104, 0.38446, 0.38449]
```

### Score a pair of trajectories directly

`select` is built on a pairwise reward model. For the raw fine-grained rewards
of a single comparison, call `compare`:

```python
reward_a, reward_b = llm_verifier.compare(
    problem, candidates[0], candidates[1],
    criteria={"Overall": "Does the code solve the problem?"},
)
print(reward_a, reward_b)   # fine-grained rewards in [0, 1]: 0.99994 4.68254e-05
```

### Fine-grained Progress Tracking

The same fine-grained reward can also score an agent's progress after each
step with `track`:

```python
steps = [
    'Read the problem statement',
    'Wrote def rev(s): return s ',
    'Tested: rev("abc") returned "abc"',
    'Changed to def rev(s): return s[::-1]',
    'Tested: rev("abc") returned "cba"',
]

result = llm_verifier.track(problem=problem, steps=steps,
                            checkpoint_steps=[1, 2, 3, 4, 5], n_evaluations=4)
print(result.scores)  # progress after each step: [0.00106, 0.02417, 0.03143, 0.62004, 0.99978]
```
---

## Test-Time Scaling for Agentic Benchmarks

Each benchmark ships with its agent trajectories (`data/`). Expected results:

| Benchmark | Base Model | Harness | Pass@1 | LLM-as-a-Verifier | Oracle |
|---|---|---|---|---|---|
| Terminal-Bench V2 | GPT-5.5 (Best-of-5) | Capy | 83.1% | **86.5%** | 92.1% |
| SWE-Bench Verified | Opus 4.5 / Opus 4.6 / Gemini 3 Flash (Best-of-3) | mini-swe-agent | 76.1% | **78.2%** | 84.4% |
| MedAgentBench | Claude Opus 4.8 (Best-of-5) | AgentBench | 70.2% | **73.3%** | 75.0% |

### Reproduce Results

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

Benchmarks are defined in `llm_verifier/benchmarks.py` — add or tweak one there.

### Select Best of N agent trajectories

Given a task and a pool of agent trajectories, pick the best one in a few
lines of code.

```python
import llm_verifier

problem = "Fix the failing test in utils.py."
candidates = [traj_1, traj_2, traj_3, traj_4, traj_5]

result = llm_verifier.select(
    problem=problem,
    candidates=candidates,
    criteria={"Root cause": "Did the agent fix the real cause?",
              "Verification": "Did the agent confirm the fix?"},
    model="gemini-2.5-flash",          # verifier model
    n_evaluations=4,                 # repeated evaluations per criterion
    pivots=2,                          # pivots < N; reduced verification cost
)

print("Best candidate:", result.index)            
print("Ranking:", result.ranking)                
```

Under the hood, `select` runs the
[Probabilistic Pivot Tournament](#probabilistic-pivot-tournament) to rank all
`N` trajectories using `O(Nk²)` pairwise verifications instead of a full
`O(N²)` round-robin. `pivots` trades cost for accuracy: more pivots = more
comparisons = higher accuracy.

### Adapt LLM-as-a-Verifier for your own use case

Use the verifier for your own task in three steps — Claude Code does the rest
(generates the criteria, writes a runner, and selects the best-of-N for you):

1. **Add your data.** Copy your agent trajectories into `data/task_name_trajs/`.
2. **Update naming.** Replace every `task_name` in
   [`add_new_benchmark.md`](add_new_benchmark.md) with the name of your task.
3. **Spin up Claude Code in this repo** (or Codex, or whatever you like — with
   permissions disabled) and paste the contents of `add_new_benchmark.md` to let
   it run.

---

## Progress Tracking

The same fine-grained reward can score a trajectory *at every step*, not just
at the end. `track` shows the verifier the task and the numbered agent steps,
and asks at each checkpoint whether the agent's current state would already
satisfy the task's hidden grader. One verifier call scores all checkpoints;
`n_evaluations` repeats are averaged into a progress curve in [0, 1]:

```python
result = llm_verifier.track(
    problem=problem,
    steps=agent_steps,       # one string per agent step (action + observed output)
    n_evaluations=16,      # repeats K; the curve is their mean
)

print(result.steps)          # checkpoint step numbers
print(result.scores)         # progress score after each step
print(result.final)          # score at the last checkpoint
```

The curve separates runs long before the grader does. Below, two **Terminus 2**
runs of the Terminal-Bench 2 task **`pytorch-model-cli`** (Gemini 2.5 Pro base
model, K=16): the successful run stays near 0 while it reads the code and
installs the toolchain, climbs as the right artifacts appear, and peaks once
its verification passes — the failed run of the same task hits a disk-space
wall, plateaus on a broken compilation, and never catches up. Bands are ±1 std
over the K repeats; steps and scores are normalized to [0, 1].

<p align="center">
  <img src="figures/progress_pytorch_model_cli.png" alt="Progress curves for two pytorch-model-cli runs" width="100%">
</p>

Reproduce this figure with:

```bash
python plot_progress.py cache/progress_pytorch-model-cli_k16.json
```

### Online progress tracking

`track` scores a finished trajectory (one verifier call per repeat, which
shows the verifier the whole trajectory). For an agent that is **still
running**, use `ProgressTracker`: each `update` scores only the steps taken
so far, so the verifier structurally cannot peek at the future — at the cost
of one scoring call per step per repeat.

```python
tracker = llm_verifier.ProgressTracker(problem, n_evaluations=4)

for step in agent_run():                 # as the agent executes
    score = tracker.update(step)         # progress in [0, 1] so far
result = tracker.result()                # same shape as track()'s
```

---

## Directory Structure

```
.
├── run.py                       # registry-driven launcher
├── plot_progress.py             # render progress-trace figures from track() curves
├── criteria/                    # verifier criteria + ground-truth notes
│   ├── TEMPLATE.md              #   copy this to write your own
│   ├── terminal_bench.md
│   ├── swe_bench.md
│   └── medagentbench.md
├── llm_verifier/                  # the reusable framework (import llm_verifier)
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
