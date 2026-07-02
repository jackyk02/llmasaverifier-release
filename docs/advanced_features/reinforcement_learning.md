# Reinforcement Learning with Verifier Rewards

The fine-grained verifier score is a drop-in **dense reward** for both off-policy and on-policy RL, improving sample efficiency by ≈1.8× on LIBERO and ≈1.1× on MATH.

<img src="../_static/image/combined_libero_math.png" alt="RL sample efficiency" width="100%">

## Off-policy RL: dense progress rewards for SAC

When fine-tuning a $\pi_0$ policy on LIBERO with DSRL-SAC, each rollout is relabeled with the verifier's per-step progress score $\rho_t$ (see [Progress Tracking](../basic_usage/progress_tracking.md)) as a dense shaped reward; the transitions are stored in the replay buffer $\mathcal{D}$ and the SAC critic is trained on returns sampled from $\mathcal{D}$:

$$
r_t = r^{\text{env}}_t + \lambda\,\rho_t
\qquad\quad
\mathcal{D} \leftarrow \mathcal{D} \cup \{(s_t, a_t, r_t, s_{t+1})\}
$$

**Sample efficiency on LIBERO** ($\pi_0$ + DSRL-SAC): environment timesteps required to reach each target success rate (averaged over $n{=}5$ seeds).

| Target SR (%) | Sparse | LLM-as-a-Verifier | Steps saved | Speedup |
|---|---|---|---|---|
| 20 | 132,800 | 74,300 | 58,500 | **1.79×** |
| 40 | 309,400 | 168,500 | 140,900 | **1.84×** |
| 60 | 993,800 | 600,000 | 393,800 | **1.66×** |

## On-policy RL: dense reasoning rewards for GRPO

When fine-tuning Qwen3-8B on MATH with GRPO, sparse correctness rewards give no gradient when every sampled answer is wrong.
Each completion's reasoning trace is scored with the [Probabilistic Pivot Tournament](pivot_tournament.md), and this reasoning-quality score is added to the correctness and format rewards to provide additional signal:

$$
r_i = r_{\mathrm{correct},i} + r_{\mathrm{format},i} + \beta\, r_{\mathrm{reasoning},i}
$$

**Sample efficiency on MATH** (Qwen3-8B + GRPO): sampled completions required to reach each target success rate (averaged over $n{=}3$ seeds; each step samples 1024).

| Target SR (%) | Sparse | LLM-as-a-Verifier | Completions saved | Speedup |
|---|---|---|---|---|
| 20 | 40,890 | 36,010 | 4,880 | **1.14×** |
| 40 | 47,990 | 43,450 | 4,540 | **1.10×** |
| 60 | 55,860 | 50,780 | 5,080 | **1.10×** |

```{note}
Both integrations use the verifier zero-shot — no reward-model fine-tuning is involved. The progress reward comes from `llm_verifier.track` / `ProgressTracker`, and the reasoning reward from `llm_verifier.select`-style tournaments over sampled completions.
```
