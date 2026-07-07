Welcome to LLM-as-a-Verifier
===============================

.. raw:: html

  <a class="github-button" href="https://github.com/llm-as-a-verifier/llm-as-a-verifier" data-size="large" data-show-count="true" aria-label="Star llm-as-a-verifier/llm-as-a-verifier on GitHub">Star</a>
  <a class="github-button" href="https://github.com/llm-as-a-verifier/llm-as-a-verifier/fork" data-icon="octicon-repo-forked" data-size="large" data-show-count="true" aria-label="Fork llm-as-a-verifier/llm-as-a-verifier on GitHub">Fork</a>
  <script async defer src="https://buttons.github.io/buttons.js"></script>
  <br></br>

.. image:: _static/image/llmoverview.png
   :width: 100%
   :alt: LLM-as-a-Verifier overview

LLM-as-a-Verifier is a general-purpose framework that provides fine-grained feedback for any agent without requiring additional training. By leveraging the full distribution of scoring-token logits, our method captures evaluation uncertainty and enables verification to scale along three dimensions: score granularity, repeated evaluation, and criteria decomposition. The resulting fine-grained feedback can be used for test-time scaling, progress tracking, and reinforcement learning.
Its core features include:

- **Simple, Extensible API**: ``llm_verifier.select``, ``compare``, and ``track`` cover best-of-N selection, pairwise scoring, and progress tracking in a few lines.
- **Fine-Grained Rewards**: Continuous rewards in [0, 1] by scaling verification across multiple dimensions.
- **Cost-Efficient Best-of-N Selection**: The Probabilistic Pivot Tournament (PPT) ranks N candidate trajectories with O(Nk²) pairwise verifications instead of a full O(N²) round-robin, concentrating the budget on uncertain top candidates.
- **Progress Tracking**: The same fine-grained reward scores a trajectory at every step — offline over a finished run or online while the agent is still executing.
- **Multimodal Inputs**: Every API supports optional image and video inputs, enabling the verifier to score images, VLM-based agents, and robotics rollouts.
- **State-of-the-Art Results**: Achieves SOTA on Terminal-Bench V2 (86.5%), SWE-Bench Verified (78.2%), RoboRewardBench (87.4%), and MedAgentBench (73.3%).
- **RL-Ready Dense Rewards**: Serves as a drop-in dense reward for reinforcement learning, improving sample efficiency across both off-policy and on-policy RL.

.. toctree::
   :maxdepth: 1
   :caption: Quickstart

   get_started/installation.md
   basic_usage/best_of_n_selection.md
   basic_usage/pairwise_comparison.md
   basic_usage/progress_tracking.md
   basic_usage/criteria.md

.. toctree::
   :maxdepth: 1
   :caption: Benchmarks

   benchmarks/running_benchmarks.md
   benchmarks/add_new_benchmark.md

.. commented out:
   benchmarks/results.md

.. toctree::
   :maxdepth: 1
   :caption: Multimodal

   multimodal/image_inputs.md

.. toctree::
   :maxdepth: 1
   :caption: References

   references/api.md
   references/directory_structure.md
   references/faq.md
   references/citation.md

.. toctree::
   :maxdepth: 1
   :caption: Advanced Features

   advanced_features/best_of_n_at_scale.md
   advanced_features/fine_grained_reward.md
   advanced_features/pivot_tournament.md
   advanced_features/verification_scaling.md
   advanced_features/progress_case_study.md
   advanced_features/logit_restricted_models.md
   advanced_features/reinforcement_learning.md
