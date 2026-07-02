# Probabilistic Pivot Tournament

To pick the best of $N$ candidate trajectories, a round-robin tournament scores all $\binom{N}{2}$ pairs — $\mathcal{O}(N^2)$ verifier calls.
The **Probabilistic Pivot Tournament (PPT)** is a cost-efficient ranking algorithm in which every candidate is compared only against a small set of $k \ll N$ pivots, reducing the budget from $\mathcal{O}(N^2)$ to $\mathcal{O}(Nk^2)$.
The implementation lives in `llm_verifier/pivot_tournament.py`.

<img src="../_static/image/pivot_tournament.png" alt="Probabilistic Pivot Tournament" width="100%">

## The algorithm

1. **Candidates:** the pool $\{\tau_1,\dots,\tau_N\}$ to be ranked.
2. **Ring pass:** a random Hamiltonian cycle scores the $N$ adjacent pairs so every candidate appears once in the "A" slot and once in "B", canceling the model's positional bias.
3. **Pivot selection:** candidates are ranked by their ring-pass scores $w_{(i)}$, and the top-$k$ candidates form the pivot set $\mathcal{P}$.
4. **Pivot tournament:** every *non-pivot–vs–pivot* and *pivot–vs–pivot* pair is scored via the pairwise preference $p(a \succ b) = \sigma(R_a - R_b)$, concentrating the budget on uncertain top candidates and cutting cost from $\mathcal{O}(N^2)$ to $\mathcal{O}(Nk^2)$.
5. **Selection:** comparisons are aggregated into win mass $w_i$ and count $c_i$, and the candidate with the highest normalized $w_i/c_i$ is returned.

## Two-phase scoring

Scoring is two-phase because the pivots depend on the ring-pass results: `select` first scores all ring pairs, then chooses pivots, then scores the pivot rounds.
Only the directed pairs the tournament actually needs are scored, and every pair is cached (see the `cache` argument), so re-runs are incremental.

## Tuning

- **`pivots` (k):** trades cost for accuracy — more pivots means more comparisons and higher accuracy. Keep `k` small relative to `N`; `k ≥ N` degenerates to a full round-robin (`k` is clamped to `N`). The bundled benchmarks default to `pivots=2`.
- **`n_evaluations` (K):** repeats per criterion for each scored pair; the [repeated-evaluation scaling axis](verification_scaling.md). Benchmarks default to `K=8`.
- **`seed`:** seeds the random ring pass. Identical inputs with the same seed run the identical tournament, so results are reproducible.

## Cost model

For one task, the number of directed verifier comparisons is the $N$ ring pairs plus the pivot-round pairs, each scored $C \times K$ times (criteria × repeats).
`result.n_comparisons` reports how many directed comparisons were run:

```python
result = llm_verifier.select(problem, trajectories,
                             criteria="terminal_bench",
                             pivots=2, n_evaluations=8)
print(result.n_comparisons)
```
