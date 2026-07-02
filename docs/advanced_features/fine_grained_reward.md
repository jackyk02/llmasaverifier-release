# Fine-Grained Reward Estimation

This page explains the core method behind LLM-as-a-Verifier.
The implementation lives in `llm_verifier/fine_grained_reward.py`.

## Motivation: generation is not the bottleneck — verification is

<img src="../_static/image/motivation.png" alt="Generation vs. verification" width="100%">

Most agents already "know" how to solve their tasks.
On Terminal-Bench, repeatedly sampling 100 trajectories per task can outperform frontier models and nearly solve the entire benchmark — but the agent doesn't know *which* trajectory is correct, particularly on long-horizon tasks.
Standard LLM-as-a-Judge scoring fails to provide sufficiently fine-grained feedback: when comparing complex solutions, judges often assign the same discrete score, resulting in a tie.
Coarse scoring leads to **27% ties** on Terminal-Bench V2.

## The scoring prompt

Given a task prompt $x$, a language model $p_\theta$, a criterion $c$, and two candidate trajectories $\tau_i$ and $\tau_j$, we construct scoring prompts and obtain the conditional distributions $p_{\theta}(v \mid x,c,\tau_i)$ and $p_{\theta}(v \mid x,c,\tau_j)$ by extracting the logprobs from the `<score_A>` and `<score_B>` tags:

```text
You are an expert [domain] reviewer. You will see a task description and two trajectories.

Evaluation Criteria: [domain specific criteria]

Task: {task prompt}
Trajectory A: {A}   Trajectory B: {B}

Carefully analyze each trajectory, then provide your final scores:

<score_A> INTEGER_1_TO_20 </score_A>
<score_B> INTEGER_1_TO_20 </score_B>

Rating Rules: Rate correctness on a 1–20 scale based on evaluation criteria
(1 = incorrect, 10 = borderline, 20 = correct)
```

```{note}
A letter-based scale (A–T) is used internally instead of digits so that each score level is a single token, enabling logprob extraction at any granularity.
```

## The reward

Rather than reducing each distribution into a single discrete score (as in LLM-as-a-Judge), LLM-as-a-Verifier approximates the reward of a trajectory $\tau$ on task $x$ as:

$$
R(x, \tau)
= \frac{1}{CK} \sum_{c=1}^{C} \sum_{k=1}^{K}
\sum_{g=1}^{G} p_{\theta}(v_g \mid x, c, \tau)\,\phi(v_g)
$$

where:

- $C$ = number of evaluation criteria
- $K$ = number of repeated verifications
- $G$ = number of score tokens (granularity level; the default `GRANULARITY` is 20)
- $p_{\theta}(v_g \mid x, c, \tau)$ = probability assigned by model $\theta$ to score token $v_g$
- $\phi(v_g)$ = maps each scoring token to a scalar value
- $V_{\text{score}} = \{v_1, \ldots, v_G\}$ = ordered set of discrete score tokens

This probabilistic formulation substantially reduces tie rates when comparing complex solutions: the continuous reward captures the verifier's *belief*, including its uncertainty, instead of collapsing it to one integer.

## From rewards to preferences

For ranking, the continuous rewards convert to a pairwise preference:

$$
p(a \succ b) = \sigma(R_a - R_b)
$$

These preferences are aggregated by the [Probabilistic Pivot Tournament](pivot_tournament.md) when selecting the best of N candidates.

## Score caching

Every scored `(criterion, task, A, B, repeat)` tuple is cacheable.
Pass `cache="path/to/cache.json"` to `select` (the bundled benchmarks each define their own cache under `cache/`): re-running with the same cache re-scores only comparisons not seen before, so interrupted runs resume cheaply and repeated experiments are free.
Failed calls scored as ties (`on_error="tie"`) are never persisted to the cache.
