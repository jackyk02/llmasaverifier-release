LLM-as-a-Verifier Documentation
===============================

.. raw:: html

  <a class="github-button" href="https://github.com/llm-as-a-verifier/llm-as-a-verifier" data-size="large" data-show-count="true" aria-label="Star llm-as-a-verifier/llm-as-a-verifier on GitHub">Star</a>
  <a class="github-button" href="https://github.com/llm-as-a-verifier/llm-as-a-verifier/fork" data-icon="octicon-repo-forked" data-size="large" data-show-count="true" aria-label="Fork llm-as-a-verifier/llm-as-a-verifier on GitHub">Fork</a>
  <script async defer src="https://buttons.github.io/buttons.js"></script>
  <br></br>

.. image:: _static/image/llmoverview.png
   :width: 100%
   :alt: LLM-as-a-Verifier overview

LLM-as-a-Verifier is a general-purpose verification framework that provides fine-grained,
training-free feedback for any agent — any modality, many applications, one unified framework.
It identifies verification as a new scaling axis for LLMs: instead of prompting a judge for a
single discrete score, it takes the expectation over the full logprob distribution of score
tokens, and scales verification along score granularity, repeated evaluation, and criteria
decomposition.
Its core features include:

- **Fine-Grained Rewards**: Computes continuous rewards in [0, 1] as the expectation over the verifier's scoring-token logits, capturing evaluation uncertainty and eliminating the tie rates that plague discrete LLM-as-a-Judge scoring.
- **Cost-Efficient Best-of-N Selection**: The Probabilistic Pivot Tournament (PPT) ranks N candidate trajectories with O(Nk²) pairwise verifications instead of a full O(N²) round-robin, concentrating the budget on uncertain top candidates.
- **Progress Tracking**: The same fine-grained reward scores a trajectory at every step — offline over a finished run or online while the agent is still executing — enabling early stopping of hopeless rollouts and safe agent deployment.
- **Multimodal Inputs**: Every API accepts images — file paths, URLs, or raw bytes; one or many — so the verifier scores screenshots, rendered plots, and per-step camera frames instead of the agent's narration, on both the Gemini and OpenAI-compatible backends.
- **State-of-the-Art Results**: Achieves SOTA test-time scaling performance on Terminal-Bench V2 (86.5%), SWE-Bench Verified (78.2%), RoboRewardBench (87.4%), and MedAgentBench (73.3%).
- **RL-Ready Dense Rewards**: Serves as a drop-in dense reward for reinforcement learning, improving sample efficiency by ≈1.8× for SAC on LIBERO and ≈1.1× for GRPO on MATH.
- **Simple, Extensible API**: ``llm_verifier.select``, ``compare``, and ``track`` cover best-of-N selection, pairwise scoring, and progress curves in a few lines; new benchmarks plug into a typed registry with Markdown criteria files.

.. toctree::
   :maxdepth: 1
   :caption: Get Started

   get_started/install.md

.. toctree::
   :maxdepth: 1
   :caption: Basic Usage

   basic_usage/quickstart.md
   basic_usage/pairwise_comparison.md
   basic_usage/progress_tracking.md
   basic_usage/criteria.md

.. toctree::
   :maxdepth: 1
   :caption: Multimodal

   multimodal/image_inputs.md

.. toctree::
   :maxdepth: 1
   :caption: Advanced Features

   advanced_features/fine_grained_reward.md
   advanced_features/pivot_tournament.md
   advanced_features/verification_scaling.md
   advanced_features/logit_restricted_models.md
   advanced_features/reinforcement_learning.md

.. toctree::
   :maxdepth: 1
   :caption: Benchmarks

   benchmarks/running_benchmarks.md
   benchmarks/add_new_benchmark.md
   benchmarks/results.md

.. toctree::
   :maxdepth: 1
   :caption: References

   references/api.md
   references/directory_structure.md
   references/faq.md
   references/citation.md
