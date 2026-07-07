# Best-of-N Selection at Scale

Given a task and a pool of N agent trajectories, `select` ranks them all and picks the best:

```python
import llm_verifier

problem = "Fix the failing test in utils.py."
trajectories = [traj_1, traj_2, traj_3, traj_4, traj_5]  # N trajectories

result = llm_verifier.select(
    problem=problem,
    candidates=trajectories,
    criteria={"Root cause": "Did the agent fix the real cause?",
              "Verification": "Did the agent confirm the fix?"},
    model="gemini-2.5-flash",          # verifier model (needs VERTEX_API_KEY for logprobs)
    n_evaluations=4,                 # repeated evaluations per criterion
    pivots=2,                          # pivots < N; reduced verification cost
)

print("Best trajectory:", result.index)           # result.best is the trajectory itself
print("Ranking:", result.ranking)                 # all trajectories, best-first
```

Under the hood, `select` runs the [Probabilistic Pivot Tournament](pivot_tournament.md) to rank all `N` trajectories using `O(Nk²)` pairwise verifications instead of a full `O(N²)` round-robin.
`pivots` trades cost for accuracy: more pivots = more comparisons = higher accuracy.

## Tournament controls

- `pivots`: pivots `k` in the tournament; keep `k` small relative to `N` — cost grows as `O(Nk²)`, and `k ≥ N` degenerates to a full round-robin.
- `seed`: identical inputs with the same seed run the identical tournament.
- `cache`: JSON score cache path; re-runs only score comparisons not seen before, so interrupted runs resume cheaply.
- `max_workers`: concurrency for verifier calls (default 50).

This is exactly how the [bundled benchmarks](../benchmarks/running_benchmarks.md) run: each swing task is one `select`-style tournament with cached scores.
