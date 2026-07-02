# Verification as a Scaling Axis

Scaling pre-training, post-training, and test-time compute are the central paradigms for improving LLM capabilities.
LLM-as-a-Verifier identifies **verification** — the ability to determine the correctness of a solution — as a new scaling axis.
Verification accuracy consistently improves as we scale across three dimensions: (1) the granularity of score tokens, (2) the number of repeated evaluations, and (3) the decomposition of evaluation criteria.

<img src="../_static/image/granularity.png" alt="Verification scaling" width="100%">

## Score granularity

Finer score tokens give the decoder a finer space to project the model's belief, improving separation between correct and incorrect solutions.
Accuracy rises from **73.1% (G=1) to 77.5% (G=20)** on Terminal-Bench V2.

**What drives it: better separation between positive and negative solutions.**
Decompose the pairwise score gap between correct ($s_c$) and incorrect ($s_i$) trajectories into a signal and a noise component:

$$
\mathrm{SNR}(G) = \frac{\mathbb{E}[s_c - s_i]}{\sqrt{\mathrm{Var}(s_c - s_i)}}
$$

where $\mathbb{E}[s_c-s_i]$ captures how strongly the verifier prefers the correct trajectory (*signal strength*) and $\mathrm{Var}(s_c - s_i)$ captures how inconsistent that preference is across pairs (*noise*).
The signal-to-noise ratio grows with the number of scoring tokens $G$ (Terminal-Bench V2, $k{=}16$):

| Granularity $G$ | 1 | 4 | 16 | 20 |
|---|---|---|---|---|
| SNR (k=16) | 0.775 | 0.786 | 0.797 | **0.799** |

## Repeated evaluation

Averaging $K$ independent evaluations is a Monte Carlo estimator whose variance shrinks as $O(1/K)$, averaging out per-pass noise.
Accuracy rises from **74.7% (K=1) to 77.4% (K=16)**.

**What drives it: variance reduction.**
LLM-as-a-Verifier consistently outperforms LLM-as-a-Judge, achieving **77.4%** verification accuracy while eliminating ties entirely across all repeated verification budgets.
Even at $k = 16$, where repeated verification reduces judge ties, the verifier still maintains **7.2%** higher accuracy.

<img src="../_static/image/judge_vs_verifier.png" alt="Verifier vs Judge: accuracy and tie rate across repeated evaluations" width="100%">

## Criteria decomposition

Granularity and repeated evaluation both assume the rubric itself is adequate.
In long-horizon agentic tasks, a judgment like "is this trajectory correct?" conflates several logically distinct factors, and a verifier asked a compound question often latches onto whichever factor is most salient in the prompt.

**What drives it: complexity reduction.**
Replace the single monolithic rubric with an ensemble over $C$ simpler sub-criteria.
For code-agent trajectories, correctness decomposes into three factors that are each easier to verify — **Specification** (all task requirements satisfied), **Output** (final output format matches the expected result), and **Errors** (no failure signals in logs and tool outputs) — with the expected scores averaged across criteria.
Any single criterion alone reaches 75.2–76.4% accuracy; their ensemble reaches **78.3%**.

<img src="../_static/image/criteria_scaling_plot.png" alt="Criteria decomposition scaling" width="100%">

See [Writing Verifier Criteria](../basic_usage/criteria.md) for how to write decomposed criteria for your own task.

## Case study: Terminal-Bench `query-optimize`

To concretely illustrate how scaling granularity to $G{=}20$ and the probabilistic formulation sharpen the verifier's signal, consider a representative trajectory pair from the `query-optimize` task on Terminal-Bench V2.
The agent is given a slow SQL query and asked to produce an equivalent optimized version; both candidates run faster, but only one validates equivalence against the canonical database.

Over 100 repeated evaluations, a discrete 1–5 judge collapses these nuanced assessments into ties (88/100).
Taking the expectation over the *same* 5-point distribution eliminates ties entirely and ranks the correct trajectory higher in 69/100 runs; scaling granularity to $G{=}20$ sharpens the signal further, ranking it strictly higher in **77/100** runs.

| Method | correct > incorrect ✅ | correct = incorrect ⚖️ | correct < incorrect ❌ |
|---|---|---|---|
| Judge (discrete, G=5) | 12/100 | 88/100 | 0/100 |
| Verifier (continuous, G=5) | 69/100 | 0/100 | 31/100 |
| **Verifier (continuous, G=20)** | **77/100** | 0/100 | 23/100 |
