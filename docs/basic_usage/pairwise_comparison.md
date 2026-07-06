# Pairwise Comparison

`select` is built on a pairwise reward model.
For the raw fine-grained rewards of a single comparison, call `compare`:

```python
import llm_verifier

r_a, r_b = llm_verifier.compare(
    problem, trace_a, trace_b,
    criteria={"Overall": "Did the agent solve the task?"},
)
print(r_a, r_b)   # fine-grained rewards in [0, 1]
```

The verifier sees `trace_a` in slot A and `trace_b` in slot B; the returned rewards are averaged over all criteria and `n_evaluations` repeats.
To score against visual evidence, attach image(s) with `images=` — see [Multimodal Verification with Images](../multimodal/image_inputs.md).

## Arguments

```python
r_a, r_b = llm_verifier.compare(
    problem, trace_a, trace_b,
    criteria="swe_bench",        # same forms as select()
    images=None,                 # task-context image(s): a path, URL, or bytes — or a list
    ground_truth_note=None,      # optional note the verifier always sees
    n_evaluations=1,           # repeats K, averaged
    max_workers=8,               # concurrency for the K x C scoring calls
    model="gemini-2.5-flash",
    client=None,                 # pre-built google-genai client (optional)
)
```

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

```{note}
Unlike `select`, `compare` has no tie fallback — a failed verifier call raises.
It also raises `MissingAPIKeyError` if no credentials are found and no `client` is given.
```
