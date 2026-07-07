---
orphan: true
---

# Benchmark Results

LLM-as-a-Verifier achieves state-of-the-art performance across coding, robotics, and medical domains: Terminal-Bench V2 (86.5%), SWE-Bench Verified (78.2%), RoboRewardBench (87.4%), and MedAgentBench (73.3%).

<img src="../_static/image/SOTA.png" alt="State-of-the-art across domains" width="100%">

## Test-time scaling

Across challenging benchmarks such as Terminal-Bench V2, SWE-Bench Verified, and MedAgentBench, LLM-as-a-Verifier — using **Gemini 2.5 Flash** as the verifier model — outperforms frontier models including Claude Opus 4.8, GPT-5.5, and Gemini models.
Results for Terminal-Bench and SWE-Bench are reported from the official leaderboards.

| Benchmark | Baseline #1 | Baseline #2 | Baseline #3 | Pass@1 | Oracle | Ours |
|---|---|---|---|---|---|---|
| Terminal-Bench V2 | GPT-5.5 (84.7%) | Opus 4.7 (80.2%) | Gemini 3.1 Pro (80.2%) | 83.1% | 92.1% | **86.5%** |
| SWE-Bench Verified | Opus 4.5 (76.8%) | Gemini 3 Flash (75.8%) | MiniMax M2.5 (75.8%) | 76.1% | 84.4% | **78.2%** |
| MedAgentBench | Opus 4.8 (70.2%) | Gemini 3.5 Flash (66.3%) | GPT-5.5 (65.1%) | 70.2% | 75.0% | **73.3%** |

## Preference accuracy on RoboRewardBench

For RoboRewardBench, pairs of rollout videos are curated that follow the same natural-language instruction but make different amounts of progress; the reward model must output a preference indicating which rollout makes more progress.
Applied **zero-shot** with a Qwen 3.6 35B VLM, LLM-as-a-Verifier outperforms reward models fine-tuned specifically on robotics data, and reduces MAE against human annotations from 1.11 to **0.72**.

| Method | Accuracy (%) |
|---|---|
| TOPReward | 74.7 |
| Robometer-4B | 78.8 |
| RoboReward-8B | 81.4 |
| LLM-as-a-Judge (Discrete) | 70.8 |
| **LLM-as-a-Verifier (Ours)** | **87.4** |

## Progress tracking

Progress tracking is quantified with the **Value-Order Correlation (VOC)** — the Spearman rank correlation between a step's chronological index and the verifier's predicted value for the prefix ending at that step.
A verifier that tracks progress assigns monotonically higher scores to later prefixes of a successful rollout ($\mathrm{VOC}\to 1$):

$$
\mathrm{VOC} = \mathrm{rank\text{-}correlation}\!\left( \mathrm{argsort}(s_{t_1}, s_{t_2}, \cdots, s_{t_K}),\ (t_1, t_2, \cdots, t_K) \right)
$$

**VOC on Terminal-Bench V2.** Successful rollouts show near-monotonic progress, while failed rollouts correlate more weakly:

| Trajectory outcome | Spearman VOC |
|---|---|
| Successful | **0.848** |
| Failed | 0.769 |
| Success − Failed (gap) | +0.079 |

**VOC on RoboRewardBench.** LLM-as-a-Verifier attains the highest correlation between step index and predicted progress, outperforming fine-tuned robotics reward models:

| Method | Spearman VOC |
|---|---|
| **LLM-as-a-Verifier (Qwen 3.6 35B)** | **0.966** |
| RoboReward-8B | 0.877 |
| Robometer-4B | 0.780 |
| TOPReward (Qwen 3.6) | 0.565 |

## Reinforcement learning

See [Reinforcement Learning with Verifier Rewards](../advanced_features/reinforcement_learning.md) for the full LIBERO (SAC, ≈1.8× sample efficiency) and MATH (GRPO, ≈1.1×) results.
