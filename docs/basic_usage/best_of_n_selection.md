# Best-of-N Selection

<!-- Given a task and candidate agent trajectories, LLM-as-a-Verifier picks the best one in a few lines of code.

<img src="../_static/image/llmoverview.png" alt="LLM-as-a-Verifier overview" width="100%">

## Run a first end-to-end selection -->

`select` takes a task, candidate solutions, and criteria to judge them by, and returns the best candidate.
With a [verifier backend configured](../get_started/installation.md#set-up-a-verifier-backend), this makes real verifier calls:

```python
import llm_verifier

problem = "Write a function that reverses a string."
candidates = [
    "def rev(s): return s[::-1]",              # correct reversal
    "def rev(s): return s",                    # identity — returns the string unchanged
    "def rev(s): return ''.join(sorted(s))",   # sorts the characters instead
]

result = llm_verifier.select(
    problem=problem,
    candidates=candidates,
    criteria={"Correctness": "Does the code actually reverse the string?"},
    n_evaluations=8,                   # repeated evaluations per criterion
    pivots=2,                          # pivots k < N; reduced verification cost
    max_workers=50,                    # concurrency for the scoring calls
    model="gemini-2.5-flash",          # verifier model
)
print(result.index)   # index of the best candidate: 0
print(result.scores)  # candidate scores: [0.73104, 0.38446, 0.38449]
```

In the example above, the verifier picks the correct string reversal over two incorrect candidates.

Under the hood, `select` runs the [Probabilistic Pivot Tournament](../advanced_features/pivot_tournament.md) to rank all `N` trajectories efficiently — `O(Nk²)` pairwise verifications instead of a full `O(N²)` round-robin, where `k` is the user-set number of `pivots` (`k < N`).

## Key arguments

- `criteria` *(required)*: a benchmark name, a criteria file path, a `{name: description}` dict, or a list of strings — see [Writing Verifier Criteria](criteria.md).
- `n_evaluations=8`: repeats `K` per criterion; averaging reduces per-call noise.
- `pivots=2`: pivots `k` in the tournament; cost grows as `O(Nk²)`.
- `model="gemini-2.5-flash"`: the verifier model.

See [`VerifierResult`](../references/api.md#verifierresult) for all fields (`.best`, `.ranking`, …) and arguments.

