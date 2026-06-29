"""LLM-as-a-Verifier: fine-grained reward + pivot tournament selection.

The high-level entry point is `verifier.select`: given a task and a list of
agent trajectories, it runs a Probabilistic Pivot Tournament scored by the
fine-grained Gemini reward and returns the best trajectory. For batch /
benchmark runs use the config-driven launcher in `run.py` instead.
"""

import random

from llm_verifier import pivot_tournament as ppt
from llm_verifier.fine_grained_reward import (
    GRANULARITY,
    LazyClient,
    directed_reward,
    load_dotenv,
    score_directed_pairs,
)
from llm_verifier.prompts import load_prompts

__all__ = ["select", "VerifierResult", "GRANULARITY", "load_dotenv"]


class VerifierResult:
    """Outcome of `select`.

    Attributes:
        index: index of the winning trajectory in the input list.
        best: the winning trajectory itself.
        scores: per-trajectory mean preference (w_i / c_i) from the tournament.
        n_comparisons: number of directed verifier comparisons that were run.
        criteria: the criterion ids used for scoring.
    """

    def __init__(self, index, best, scores, n_comparisons, criteria):
        self.index = index
        self.best = best
        self.scores = scores
        self.n_comparisons = n_comparisons
        self.criteria = criteria

    def __repr__(self):
        return (f"VerifierResult(index={self.index}, "
                f"n_comparisons={self.n_comparisons}, "
                f"criteria={self.criteria})")


def select(problem, trajectories, *, criteria, ground_truth_note=None,
           n_verifications=8, pivots=2, seed=0, max_workers=50, client=None):
    """Select the best of N agent trajectories for a single task.

    Scores directed pairs of trajectories with the fine-grained Gemini reward
    and aggregates them with a Probabilistic Pivot Tournament (PPT), so the cost is
    O(Nk²) verifier comparisons rather than the O(N²) of a full round-robin.

    Args:
        problem: the task description shown to the verifier.
        trajectories: list of N agent trajectories (strings) to rank.
        criteria: a bundled benchmark name (e.g. ``"swe_bench"``), a path to a
            ``*.md`` criteria file, or a list of
            ``{"id", "name", "description"}`` criterion dicts.
        ground_truth_note: optional note the verifier always sees; defaults to
            the note parsed from the prompt file (or empty).
        n_verifications: repeated verifications K per criterion.
        pivots: number of pivots k in the tournament.
        seed: seed for the random ring pass.
        max_workers: concurrency for verifier calls.
        client: a pre-built ``google-genai`` client (optional); by default one
            is created from ``VERTEX_API_KEY`` / ``GEMINI_API_KEY``.

    Returns:
        A `VerifierResult` whose ``.index`` / ``.best`` is the chosen trajectory.
    """
    if isinstance(criteria, str):
        note, crits = load_prompts(criteria)
    else:
        note, crits = "", list(criteria)
    if ground_truth_note is not None:
        note = ground_truth_note
    criteria_ids = [c["id"] for c in crits]

    n = len(trajectories)
    if n == 0:
        raise ValueError("need at least one trajectory")
    if n == 1:
        return VerifierResult(0, trajectories[0], [1.0], 0, criteria_ids)

    # One synthetic task holding the N candidate trajectories.
    task = "task"
    tasks = {task: [{"problem": problem, "trace": t, "reward": 0}
                    for t in trajectories]}

    lazy = LazyClient()
    if client is not None:
        lazy._client = client

    def directed_for(scores):
        return lambda a, b: directed_reward(
            scores, task, a, b, criteria_ids, n_verifications)

    # Phase A: ring pass (slot bias cancels around the cycle).
    rng = random.Random(seed)
    ring = ppt.ring_cycle(n, rng)
    scores = score_directed_pairs(
        lazy, tasks, {task: ring}, crits, note, n_verifications,
        max_workers, None)
    score = directed_for(scores)

    # Pivots = empirical leaders from the ring pass.
    w, c = [0.0] * n, [0] * n
    ppt.accumulate(ring, score, w, c)
    pivot_set = ppt.select_pivots(w, c, pivots)
    pr_pairs = ppt.pivot_round_pairs(n, pivot_set)

    # Phase B: score the pivot rounds, then aggregate everything.
    scores = score_directed_pairs(
        lazy, tasks, {task: pr_pairs}, crits, note, n_verifications,
        max_workers, None)
    score = directed_for(scores)

    w, c = [0.0] * n, [0] * n
    ppt.accumulate(ring, score, w, c)
    ppt.accumulate(pr_pairs, score, w, c)
    best = max(range(n), key=lambda i: (w[i] / c[i] if c[i] else 0.0, -i))
    mean_pref = [w[i] / c[i] if c[i] else 0.0 for i in range(n)]
    return VerifierResult(best, trajectories[best], mean_pref,
                          len(ring) + len(pr_pairs), criteria_ids)
