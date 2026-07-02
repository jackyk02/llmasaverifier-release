# Progress Tracking

The same fine-grained reward that ranks trajectories can score a trajectory *at every step*, not just at the end.
The verifier is shown the task and the numbered agent steps, and asked at each checkpoint whether the agent's current state would already satisfy the task's hidden grader.
Each answer letter (A = 0% progress … T = 100%) is decoded as the expectation over the verifier's top-20 logprobs at the answer position, giving a continuous progress curve in [0, 1].
On successful trajectories the curve rises toward 1; on failing ones it plateaus or falls.

## Offline: `track`

`track` scores a finished trajectory.
One verifier call scores all checkpoints; `n_evaluations` repeats are averaged into the curve, so cost is `O(K)` calls regardless of trajectory length:

```python
import llm_verifier

result = llm_verifier.track(
    problem=problem,
    steps=agent_steps,       # one string per agent step (action + observed output)
    n_evaluations=16,      # repeats K; the curve is their mean
)

print(result.steps)          # checkpoint step numbers
print(result.scores)         # progress score after each step
print(result.final)          # score at the last checkpoint
```

By default the interior steps `2 .. T-1` are scored (the first and last step anchor the scale); pass `checkpoint_steps` (1-indexed) to choose your own checkpoints.

The curve separates runs long before the grader does.
Below, two **Terminus 2** runs of the Terminal-Bench 2 task **`pytorch-model-cli`** (Gemini 2.5 Pro base model, K=16): the successful run stays near 0 while it reads the code and installs the toolchain, climbs as the right artifacts appear, and peaks once its verification passes — the failed run of the same task hits a disk-space wall, plateaus on a broken compilation, and never catches up.
Bands are ±1 std over the K repeats; steps and scores are normalized to [0, 1].

<img src="../_static/image/progress_pytorch_model_cli.png" alt="Progress curves for two pytorch-model-cli runs" width="100%">

Reproduce this figure from the repository with (`--plot-only` renders the
committed curves without an API key; drop it to re-score from the raw
trajectories in `data/tacking_examples/`):

```bash
python terminal_bench_progress.py --plot-only
```

## Online: `ProgressTracker`

`track` shows the verifier the whole trajectory per call, so earlier checkpoints could in principle be influenced by the visible ending.
For an agent that is **still running**, use `ProgressTracker`: each `update` scores only the steps taken so far, so the verifier structurally cannot peek at the future — at the cost of one scoring call per step per repeat:

```python
tracker = llm_verifier.ProgressTracker(problem, n_evaluations=4)

for step in agent_run():                 # as the agent executes
    score = tracker.update(step)         # progress in [0, 1] so far
    if tracker.steps[-1] >= 8 and score < 0.05:
        break                            # abandon a hopeless rollout early

result = tracker.result()                # same shape as track()'s
```

Use it to stop hopeless rollouts early or to decide when to branch or resample.

### The online scoring prompt

Each `update` builds the neutral progress-scoring prompt below over the prefix accumulated so far, with a single checkpoint at the current step `k` (offline `track` uses the same template, with the full trajectory and one `<c1>…<cN>` line per checkpoint).
The template never reveals whether the trajectory eventually succeeded — successes and failures see the same prompt — and the answer letter is decoded as the expectation over the verifier's top-20 logprobs at the `<c1>` position:

```text
You are a strict, skeptical evaluator of agent task attempts. Agents routinely
declare victory while their environment still shows errors, edit the wrong
target, or never actually run the verification the task asks for. Trust
observed output — NOT the agent's narration.

**Task instruction:**
{problem}

**Agent trajectory ({k} agent steps; each step is one action by the agent,
with its observed output):**
{numbered steps 1 .. k — the prefix so far}

You will score the trajectory at 1 CHECKPOINTS. The score measures exactly
ONE thing:

    "Given everything the agent has done up to and including this step,
    would the agent's CURRENT state actually satisfy the task's hidden
    grader (i.e. produce the expected files / output / behavior the task
    requires)?"

Use the 20-letter A..T scale:
  A = certainly NO — nothing useful done yet, or the agent is going down a
      clearly wrong path.
  B-G = leans NO — partial work exists but key pieces are missing or broken.
  H-M = uncertain — a plausible solution is taking shape, but no convincing
      verification yet.
  N-S = leans YES — the right artifacts appear to be in place and partial
      verification has worked, with minor concerns.
  T = essentially certain YES — the agent has run the relevant verification
      and the observed output literally matches what the task calls for,
      with no outstanding errors.

CRITICAL CALIBRATION RULES:
  * Effort, exploration, step count, and confident-sounding narration are
    NOT progress. An agent that ran 20 commands and still has not produced
    the right output deserves a score near A.
  * Default to skepticism. The hidden grader is NOT visible to you. A result
    with no real verification step should not exceed ~K, and even a
    verified-looking one should rarely exceed ~R unless the verification
    clearly matches the task's stated success criterion.
  * Treat the agent's prose declarations ("done!", "all tests pass") as ZERO
    evidence. Ground your score in the actual actions and the actual output
    you can see.

EXPECTED PATTERNS — successive checkpoints do NOT have to rise:
  * On a trajectory that genuinely solves the task, scores typically rise
    from A toward T.
  * On a trajectory committed to a WRONG approach, scores should PLATEAU
    once the wrong artifact is in place.
  * If the agent regresses (breaks something that worked), scores should
    DECREASE.

The N checkpoints to score are:
  Checkpoint 1 = state right after Agent Step {k}

Score each checkpoint INDEPENDENTLY based on the agent's current best
attempt at that point in the trajectory. Output EXACTLY N lines and nothing
else, in the format:
<c1>LETTER</c1>

where each LETTER is a single letter from A to T.
```

The prompt builder lives in `llm_verifier/progress.py` (`build_progress_prompt`).

```{note}
For progress scoring the letter scale is inverted relative to the pairwise reward scale: A = 0% progress and T = 100% progress (in pairwise scoring, A is the best score).
```

## Choosing between them

| | `track` (offline) | `ProgressTracker` (online) |
|---|---|---|
| When | trajectory is finished | agent is still running |
| Verifier sees | full trajectory per call | only the prefix so far |
| Cost for a T-step run | `n_evaluations` calls | `T × n_evaluations` calls |
| Future leakage | possible in principle | structurally impossible |

Both return a [`ProgressResult`](../references/api.md#progressresult) with `.steps`, `.scores`, `.per_rep_scores`, and `.final`.

```{tip}
Steps are plain strings — concatenate the agent's action and its observed output. Truncate very long observations yourself if needed.
```
