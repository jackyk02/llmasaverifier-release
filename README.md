<p align="center">
  <strong>LLM-as-a-Verifier</strong>
</p>

<p align="center">
  <strong>Pick the best of N agent rollouts with fine-grained, logprob-based verification — and an O(N·k) tournament instead of O(N²).</strong>
</p>

<p align="center">
  <a href="#quick-start"><strong>Quick Start</strong></a> &ensp;|&ensp;
  <a href="#how-it-works"><strong>How it works</strong></a> &ensp;|&ensp;
  <a href="#reproducing-the-results"><strong>Reproduce</strong></a> &ensp;|&ensp;
  <a href="#adding-a-benchmark"><strong>Add a benchmark</strong></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9%2B-3776ab?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/verifier-Gemini%202.5%20Flash-4285f4?logo=googlegemini&logoColor=white" alt="Gemini">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

---

## What is LLM-as-a-Verifier?

**LLM-as-a-Verifier** is a general-purpose verification framework that gives
*fine-grained* feedback by scaling scoring granularity, repeated verification,
and criteria decomposition. Unlike LLM-as-a-Judge — which collapses a judgement
into a single discrete label — it reads the verifier's full probability
distribution over an ordered set of score tokens and takes its expectation,
turning every judgement into a continuous reward.

Used as a trajectory reward model for test-time scaling, it selects the best of
`N` agent rollouts per task with a **Pivot Preference Tournament (PPT)** that
costs `O(N·k)` verifier calls instead of the `O(N²)` of a full round-robin —
while matching round-robin accuracy.

**If you can describe what "good" looks like, you can verify it**: coding
agents, SWE tasks, medical agents, tool-use trajectories, and more.

### Key Results

| Benchmark | Scaffold · Model | Pass@1 | LLM-as-a-Verifier | Oracle |
|---|---|---|---|---|
| Terminal-Bench 2.0 | Capy · GPT-5.5 (×5) | 83.1% | **86.5%** | 92.1% |
| SWE-bench Verified | mini-swe-agent (×3) | 76.1% | **78.2%** | 84.4% |
| MedAgentBench | Claude-Opus-4.8 max-effort (×5) | 70.2% | **73.3%** | 75.0% |

---

## Installation

```bash
pip install llm-verifier
```

To install the latest from a clone:

```bash
pip install -e .
```

The verifier is Gemini 2.5 Flash and reads token-level **logprobs**, so a
Vertex AI key (`VERTEX_API_KEY`) is preferred; a `GEMINI_API_KEY` also works.
Put it in a `.env` file at the repo root (or your environment):

```bash
cp .env.example .env   # then add your Vertex AI / Gemini key
```

The cached scores for all three benchmarks are committed, so reproducing the
table above needs **no key at all** (see [Reproducing](#reproducing-the-results)).

---

## Quick Start

### Select the best of N rollouts

Given a task and a handful of agent trajectories, pick the best one in a few
lines of code:

```python
import llm_verifier

problem = "Fix the failing test in utils.py so the suite passes."
trajectories = [rollout_1, rollout_2, rollout_3, rollout_4, rollout_5]  # strings

result = llm_verifier.select(
    problem=problem,
    trajectories=trajectories,
    criteria="prompts/swe_bench.md",   # decomposed evaluation criteria
    n_verifications=8,                 # repeated verifications per criterion
    pivots=2,                          # O(N·k) tournament, k = pivots
)

print("Best rollout:", result.index)
print("Verifier calls:", result.n_comparisons)   # ~N·k, not N²
```

`llm_verifier.select` samples a random ring pass (so the verifier's slot bias
cancels), picks the empirical leaders as pivots, scores only the directed pairs
the tournament needs, and returns the winner. `criteria` is a path to a
`prompts/*.md` file — or a list of `{"id", "name", "description"}` dicts:

```python
result = llm_verifier.select(
    problem=problem,
    trajectories=trajectories,
    criteria=[
        {"id": "root_cause",   "name": "Root cause",   "description": "Did the agent fix the real cause?"},
        {"id": "verification", "name": "Verification", "description": "Did the agent confirm the fix?"},
    ],
    ground_truth_note="The hidden test checks utils.parse() on empty input.",
)
```

### Score one trajectory directly

For the raw fine-grained reward over a pairwise comparison, drop down to the
reward model:

```python
from llm_verifier.fine_grained_reward import create_gemini_client, score_pair_criterion

client = create_gemini_client()
criterion = {"id": "overall", "name": "Overall", "description": "Did the agent solve the task?"}

r_a, r_b = score_pair_criterion(client, problem, trace_a, trace_b, criterion, ground_truth_note="")
print(r_a, r_b)   # fine-grained rewards in [0, 1]
```

---

## Reproducing the results

Run a benchmark by name (`python run.py` with no argument lists them):

```bash
python run.py terminal_bench
python run.py swe_bench
python run.py medagentbench
```

For example, `python run.py terminal_bench` reproduces the first row of the
table:

```
Method                               Score     Rate
------------------------------------------------------------------------
Pass@1                         74.00/89    83.1%
LLM-as-a-Verifier                 77/89    86.5%
Oracle (Bo5)                      82/89    92.1%
```

The cached verifier scores from the live runs above are committed, so these
reproduce the numbers in seconds and require **no API key**. Delete a cache file
(or change `--seed`) to score from scratch against Gemini. Override the defaults
on the command line if you like:

```bash
python run.py swe_bench --pivots 2 --n-verifications 8 --seed 0
```

Benchmarks are defined in `llm_verifier/benchmarks.py` — add or tweak one there.

---

## Layout

```
.
├── run.py                       # registry-driven launcher
├── llm_verifier/                  # the reusable framework (import llm_verifier)
│   ├── __init__.py              #   llm_verifier.select(...): best-of-N in one call
│   ├── benchmarks.py            #   BENCHMARKS registry (one Benchmark / launch)
│   ├── fine_grained_reward.py   #   R(t,τ): Gemini logprob scoring + cache
│   ├── pivot_tournament.py      #   PPT: O(N·k) selection (Bradley-Terry)
│   ├── prompts.py               #   load criteria from prompts/*.md
│   └── loaders.py               #   per-benchmark trajectory loaders
├── prompts/                     # criteria + ground-truth note, human-editable
│   ├── terminal_bench.md
│   ├── swe_bench.md
│   └── medagentbench.md
├── data/                        # agent trajectories per benchmark
├── cache/                       # cached verifier scores (committed)
└── results/                     # result tables (written after each run)
```

---

## How it works

### 1. Fine-grained reward

Rather than collapsing each judgement into a single discrete label (as in
LLM-as-a-Judge), LLM-as-a-Verifier reads the verifier's probability
distribution over an ordered set of score tokens and takes its expectation. The
reward of trajectory `τ` on task `t` is

$$
R(t, \tau)
= \frac{1}{CK} \sum_{c=1}^{C} \sum_{k=1}^{K}
\sum_{g=1}^{G} p_{\theta}(v_g \mid t, c, \tau)\,\phi(v_g)
$$

- $C$ — number of evaluation criteria (decomposed in `prompts/*.md`)
- $K$ — number of repeated verifications
- $G$ — number of ordered score tokens (granularity; here `g=20`, letters A–T)
- $p_{\theta}(v_g \mid t, c, \tau)$ — probability the verifier assigns to token $v_g$
- $\phi(v_g)$ — the scalar value of score token $v_g$

This lives in `llm_verifier/fine_grained_reward.py`.

### 2. Pivot Preference Tournament

To pick the best of `N` candidate trajectories, a round-robin tournament scores
all $\binom{N}{2}$ pairs — `O(N²)`. **PPT** reaches the same selection in three
steps, scoring directed pairs (candidate `a` in slot A, `b` in slot B):

1. **Ring pass.** Sample a uniformly random Hamiltonian cycle $\gamma$ over the
   `N` candidates and score the `N` adjacent pairs
   $\{(\gamma_t, \gamma_{t+1 \bmod N})\}$. Around the cycle every candidate sits
   in slot A exactly once and slot B exactly once, so the verifier's
   **slot bias cancels** in expectation.
2. **Pivot selection.** Rank candidates by their ring-pass mean preference
   $w_i / c_i$ and take the top-`k` as the pivot set `P` — the *empirical
   leaders*, so the remaining budget distinguishes the strongest candidates
   rather than re-scoring weak anchors.
3. **Pivot rounds.** Score every non-pivot-vs-pivot pair and every
   pivot-vs-pivot pair, aggregating all comparisons into the same $w_i, c_i$.
   The winner is $\arg\max_i w_i / c_i$; normalizing by $c_i$ removes the bias
   that pivots take part in more comparisons.

Each comparison's two fine-grained rewards $(R_a, R_b)$ become a soft win via
the Bradley-Terry model, $p(a \text{ beats } b) = \sigma(R_a - R_b)$. Total
comparisons: $N + k(N-k) + \binom{k}{2}$ — linear in `N` for fixed `k`. This
lives in `llm_verifier/pivot_tournament.py`.

---

## Adding a benchmark

1. Drop a loader in `llm_verifier/loaders.py` returning `tasks` as
   `{task_id: [{trial_name, reward, problem, trace}, ...]}` and register it in
   `LOADERS`.
2. Write the criteria + ground-truth note in `prompts/<benchmark>.md`.
3. Add a `Benchmark(...)` entry to `BENCHMARKS` in `llm_verifier/benchmarks.py` pointing
   at the loader, prompts, data, and cache, then `python run.py <benchmark>`.
