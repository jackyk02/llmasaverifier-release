# Frequently Asked Questions

## Which verifier backends are supported?

The continuous reward is the expectation over the verifier's **token-level logprobs** at the `<score_A>` / `<score_B>` positions, so the backend must expose logprobs.
Two backends are picked automatically from the environment:

- **OpenAI-compatible server** (vLLM / SGLang / OpenAI) when `OPENAI_BASE_URL` is set — e.g. `vllm serve Qwen/Qwen3.5-9B` and `export OPENAI_BASE_URL=http://localhost:8000/v1`. The served model is auto-detected, so no `model=` argument is needed.
- **Gemini via Vertex AI** otherwise, from `VERTEX_API_KEY` (logprob extraction for Gemini requires the Vertex API).

You can also pass your own pre-built `openai` or `google-genai` client via the `client` argument.

## Can I use a local open model as the verifier?

Yes — serve it with vLLM (or SGLang) and point `OPENAI_BASE_URL` at it; see [Set up a verifier backend](../get_started/installation.md#set-up-a-verifier-backend).
On this backend the score tags are **prefilled** and the score position is constrained to the 20 scale letters via structured outputs, so the extracted distribution stays calibrated even for models that don't reliably follow the tag format.

## Can I use GPT or Claude as the verifier?

Not directly — those APIs withhold token-level logprobs, which the continuous reward requires.
Use the two-stage workaround described in [Logit-Restricted Frontier Models](../advanced_features/logit_restricted_models.md): the closed model supplies the reasoning, and an open verifier (e.g., Gemini 2.5 Flash) supplies the calibrated distribution.
On Terminal-Bench V2 this recovers a +5.2-point gain over the closed model's integer scores.

## How is this different from LLM-as-a-Judge?

A judge collapses its belief into one discrete score, which produces ties on complex solutions (27% ties on Terminal-Bench V2 with coarse scoring).
The verifier keeps the full distribution over score tokens and takes its expectation, yielding continuous scores with **zero ties**, and scales further along granularity, repeated evaluation, and criteria decomposition.
See [Verification as a Scaling Axis](../advanced_features/verification_scaling.md).

## How much does a `select` call cost?

`O(Nk²)` directed comparisons instead of `O(N²)`, each costing `C × K` verifier calls (criteria × `n_evaluations`) — check `result.n_comparisons` after a run.
Reduce cost with fewer `pivots`, fewer `n_evaluations`, or a score `cache`.

## How do I make runs reproducible and resumable?

- **Reproducible**: identical inputs with the same `seed` run the identical tournament.
- **Resumable**: pass `cache="path/to/cache.json"`. Every scored `(criterion, task, A, B, repeat)` tuple is cached; re-runs only score comparisons not seen before. Error ties are never persisted to the cache.

## What happens when a verifier call fails?

In `select`, the default `on_error="tie"` scores that comparison 0.5/0.5 for the current run only; pass `on_error="raise"` to fail fast.
`compare` and `track` always raise.

## Should I use `track` or `ProgressTracker`?

Use `track` for finished trajectories (one call per repeat, cheapest).
Use `ProgressTracker` when the agent is still running and you need scores that structurally cannot peek at the future — e.g., to abandon a hopeless rollout early.
See [Progress Tracking](../basic_usage/progress_tracking.md).

## My trajectories are very long. What should I do?

Steps and trajectories are plain strings, so truncate long tool observations yourself before passing them in.
For best-of-N selection, decomposed criteria (see [Writing Verifier Criteria](../basic_usage/criteria.md)) also help the verifier focus on the relevant evidence.

## Does the verifier work outside of coding?

Yes — the same framework is applied zero-shot to robotics (RoboRewardBench, 87.4% with a Qwen 3.6 35B VLM) and medical agent tasks (MedAgentBench, 73.3%), and as a dense RL reward on LIBERO and MATH.
See [Benchmark Results](../benchmarks/results.md).
