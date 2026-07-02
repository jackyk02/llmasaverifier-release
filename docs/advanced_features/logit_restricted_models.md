# Logit-Restricted Frontier Models

Some frontier models (e.g., GPT-5.5, Claude Opus) expose only sampled completions and **withhold token-level logprobs**, which the continuous reward requires.
A simple **two-stage workaround** recovers most of the calibrated signal: the closed model supplies domain-specific reasoning, and an open verifier supplies the calibrated probability distribution it withholds.

## The two-stage pipeline

```text
┌──────────────────────────┐    reasoning +   ┌──────────────────────────┐   expectation   ┌────────────────────┐
│  Closed Frontier Model   │   draft score    │      Open Verifier       │   over logits   │  Continuous Reward │
│  GPT-5.5 · Claude Opus   │ ───────────────► │  Gemini 2.5 Flash (G=20) │ ──────────────► │       R(x, τ)      │
│    logprobs withheld     │                  │ reads <score_A>/<score_B>│                 │                    │
│                          │                  │         logprobs         │                 │                    │
└──────────────────────────┘                  └──────────────────────────┘                 └────────────────────┘
```

1. **Closed frontier model**: produces the domain-specific reasoning and a draft score for the trajectory pair.
2. **Open verifier**: reads that reasoning and re-scores it, exposing `<score_A>` / `<score_B>` logprobs at granularity G=20.
3. **Continuous reward**: the expectation over the open verifier's logits yields $R(x,\tau)$ as usual.

## Results

On Terminal-Bench V2, routing GPT-5.5's reasoning through Gemini 2.5 Flash recovers a **+5.2-point** accuracy gain over directly using the closed model's integer scores (**80.1% vs. 74.9%**) and eliminates its 10.9% tie rate entirely — without any access to the frontier model's logits.

| $K$ | GPT-5.5 (Discrete) Accuracy (%) | GPT-5.5 (Discrete) Tie rate (%) | GPT-5.5 → Gemini 2.5 Flash (Continuous) Accuracy (%) | Tie rate (%) |
|---|---|---|---|---|
| 1 | 74.9 | 10.9 | **80.1** | **0.0** |
| 2 | 76.3 | 9.1 | **80.5** | **0.0** |
| 4 | 77.6 | 7.0 | **81.0** | **0.0** |
| 8 | 78.4 | 5.8 | **80.9** | **0.0** |
| 16 | 79.1 | 5.0 | **81.2** | **0.0** |

```{tip}
If your verifier model exposes logprobs (any Vertex AI Gemini model does), you don't need this workaround — pass it directly via the `model` argument of `select` / `compare` / `track`.
```
