# Pairwise Comparison

`select` is built on a pairwise reward model.
For the raw fine-grained rewards of a single comparison, call `compare`:

```python
import llm_verifier

problem = "Write a function that reverses a string."
candidates = [
    "def rev(s): return s[::-1]",              # correct reversal
    "def rev(s): return s",                    # identity — returns the string unchanged
    "def rev(s): return ''.join(sorted(s))",   # sorts the characters instead
]

reward_a, reward_b = llm_verifier.compare(
    problem, candidates[0], candidates[1],
    criteria={"Overall": "Does the code solve the problem?"},
    n_evaluations=1,                   # repeats K per criterion, averaged
    max_workers=50,                     # concurrency for the scoring calls
    model="gemini-2.5-flash",          # verifier model
)
print(reward_a, reward_b)   # fine-grained rewards in [0, 1]: 0.99994 0
```

The returned rewards are averaged over all criteria and `n_evaluations` repeats.

## Key arguments

- `criteria` *(required)*: same forms as `select`'s — a benchmark name, a criteria file path, a `{name: description}` dict, or a list of strings.
- `n_evaluations=1`: repeats `K` per criterion; averaging reduces per-call noise.
- `max_workers=8`: concurrency for the scoring calls.
- `model="gemini-2.5-flash"`: the verifier model.

See the [API reference](../references/api.md#compare) for all arguments.

## Positional bias

A single directed call does **not** cancel slot bias: LLMs systematically favor one slot, so `compare(problem, a, b)` and `compare(problem, b, a)` can disagree.
`select`'s ring pass places every candidate once in the "A" slot and once in "B" specifically to cancel this bias around the cycle.
If you use `compare` directly and care about unbiased preferences, score both directions and average:

```python
r_ab = llm_verifier.compare(problem, a, b, criteria=crits)
r_ba = llm_verifier.compare(problem, b, a, criteria=crits)
r_a = (r_ab[0] + r_ba[1]) / 2
r_b = (r_ab[1] + r_ba[0]) / 2
```

## Preference probabilities

The continuous rewards convert directly into a pairwise preference via the sigmoid of the reward gap:

$$
p(a \succ b) = \sigma(R_a - R_b)
$$

This is the preference the [Probabilistic Pivot Tournament](../advanced_features/pivot_tournament.md) aggregates when ranking N candidates.
