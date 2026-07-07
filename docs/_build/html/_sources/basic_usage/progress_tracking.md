# Progress Tracking

The same fine-grained reward can score a trajectory *at every step*.

## Offline: `track`

`track` scores a finished trajectory.
One verifier call scores all checkpoints; `n_evaluations` repeats are averaged into the curve, so cost is `O(K)` calls regardless of trajectory length:

```python
import llm_verifier

problem = "Write a function that reverses a string."
steps = [
    'Read the problem statement',
    'Wrote def rev(s): return s ',
    'Tested: rev("abc") returned "abc"',
    'Changed to def rev(s): return s[::-1]',
    'Tested: rev("abc") returned "cba"',
]

result = llm_verifier.track(
    problem=problem,
    steps=steps,
    checkpoint_steps=[1, 2, 3, 4, 5],  # steps taken
    n_evaluations=4,                   # independent repeats K; the curve is their mean
    max_workers=8,                     # concurrency for the K repeats
    model="gemini-2.5-flash",          # verifier model
)
print(result.scores)  # progress after each step: [0.00106, 0.02417, 0.03143, 0.62004, 0.99978]
```

For a real-trajectory example on Terminal-Bench, see [Progress Tracking Case Study](../advanced_features/progress_case_study.md).

## Online: `ProgressTracker`

`track` shows the verifier the whole trajectory per call. For an agent that is **still running**, use `ProgressTracker`: each `update` scores only the steps taken so far:

```python
tracker = llm_verifier.ProgressTracker(
    problem,
    n_evaluations=4,                   # independent repeats K per update
    max_workers=8,                     # concurrency for the K repeats
    model="gemini-2.5-flash",          # verifier model
)

score = tracker.update('Read the problem statement')            # 0.00002
score = tracker.update('Wrote def rev(s): return s')            # 0.00013
score = tracker.update('Changed to def rev(s): return s[::-1]') # 0.73938
score = tracker.update('Tested: rev("abc") returned "cba"')     # 0.98604

result = tracker.result()                # same shape as track()'s
```

## Choosing between them

| | `track` (offline) | `ProgressTracker` (online) |
|---|---|---|
| When | trajectory is finished | agent is still running |
| Verifier sees | full trajectory per call | only the prefix so far |
| Cost for a T-step run | `n_evaluations` calls | `T × n_evaluations` calls |
| Future leakage | possible | structurally impossible |

Both return a [`ProgressResult`](../references/api.md#progressresult) with `.steps`, `.scores`, `.per_rep_scores`, and `.final`.
