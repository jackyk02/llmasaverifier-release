"""
Progress tracking: fine-grained reward over trajectory prefixes.

The same logprob-expectation reward used for best-of-N selection can score a
trajectory *at every step*: in ONE verifier call we show the task, the full
trajectory with numbered agent steps, and a list of checkpoints, and ask
"would the agent's CURRENT state satisfy the task's hidden grader at this
point?" per checkpoint. Each answer letter (A = 0% progress ... T = 100%) is
decoded as the expectation over the verifier's top-20 logprobs at the answer
position, giving a continuous progress curve. On successful trajectories the
curve rises toward 1; on failing ones it plateaus or falls.

Two entry points:

- `llm_verifier.track(problem, steps, ...)` -> `ProgressResult` — offline:
  scores all checkpoints of a finished trajectory in one verifier call per
  repeat. The verifier sees the full trajectory while scoring each
  checkpoint (cheap, but earlier checkpoints could in principle be
  influenced by the visible ending).
- `llm_verifier.ProgressTracker(problem, ...)` — online: feed steps as the
  agent produces them; each `update(step)` scores only the prefix so far,
  so the verifier structurally cannot see the future. Costs one scoring
  call per step per repeat.
"""

from __future__ import annotations

import math
import re
import string
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

from llm_verifier.fine_grained_reward import (
    DEFAULT_MODEL,
    call_gemini,
    create_gemini_client,
)

# ---------------------------------------------------------------------------
# Letter scale — A = 0% progress, T = 100% progress.
# (Note: inverted relative to the pairwise reward scale, where A is best —
# for progress, later letters mean more progress.)
# ---------------------------------------------------------------------------

GRANULARITY = 20
_LETTERS = list(string.ascii_uppercase[:GRANULARITY])
LETTER_TO_VALUE = {
    **{_LETTERS[i]: i / (GRANULARITY - 1) for i in range(GRANULARITY)},
    **{_LETTERS[i].lower(): i / (GRANULARITY - 1) for i in range(GRANULARITY)},
}


@dataclass
class ProgressResult:
    """Outcome of `track`.

    Attributes:
        steps: 1-indexed agent-step numbers that were scored as checkpoints.
        scores: progress score in [0, 1] after each checkpoint step, averaged
            over `n_evaluations` repeats (same order as `steps`).
        per_rep_scores: the raw per-repeat curves (n_evaluations x
            len(steps)); entries are None where a repeat produced no
            readable score for that checkpoint.
    """

    steps: List[int]
    scores: List[float]
    per_rep_scores: List[List[Optional[float]]] = field(repr=False)

    @property
    def final(self) -> float:
        """Progress score at the last checkpoint."""
        return self.scores[-1]


def format_steps(steps: Sequence[str]) -> str:
    """Number the agent steps the way the checkpoint prompt refers to them."""
    parts = []
    for k, step in enumerate(steps, start=1):
        parts.append(f"=== Agent Step {k} ===")
        parts.append(str(step).strip())
        parts.append("")
    return "\n".join(parts)


def build_progress_prompt(problem: str, trajectory_text: str, n_steps: int,
                          checkpoint_steps: Sequence[int]) -> str:
    """Neutral progress-scoring prompt. It never reveals whether the
    trajectory eventually succeeded — successes and failures see the same
    template."""
    n = len(checkpoint_steps)
    out = [
        "You are a strict, skeptical evaluator of agent task attempts. "
        "Agents routinely declare victory while their environment still "
        "shows errors, edit the wrong target, or never actually run the "
        "verification the task asks for. Trust observed output — NOT the "
        "agent's narration.",
        "",
        "**Task instruction:**",
        problem.strip(),
        "",
        f"**Agent trajectory ({n_steps} agent steps; each step is one "
        "action by the agent, with its observed output):**",
        trajectory_text,
        "",
        f"You will score the trajectory at {n} CHECKPOINTS. The score "
        "measures exactly ONE thing:",
        "",
        "    \"Given everything the agent has done up to and including "
        "this step, would the agent's CURRENT state actually satisfy the "
        "task's hidden grader (i.e. produce the expected files / output / "
        "behavior the task requires)?\"",
        "",
        "Use the 20-letter A..T scale:",
        "  A = certainly NO — nothing useful done yet, or the agent is "
        "going down a clearly wrong path.",
        "  B-G = leans NO — partial work exists but key pieces are missing "
        "or broken.",
        "  H-M = uncertain — a plausible solution is taking shape, but no "
        "convincing verification yet.",
        "  N-S = leans YES — the right artifacts appear to be in place and "
        "partial verification has worked, with minor concerns.",
        "  T = essentially certain YES — the agent has run the relevant "
        "verification and the observed output literally matches what the "
        "task calls for, with no outstanding errors.",
        "",
        "CRITICAL CALIBRATION RULES:",
        "  * Effort, exploration, step count, and confident-sounding "
        "narration are NOT progress. An agent that ran 20 commands and "
        "still has not produced the right output deserves a score near A.",
        "  * Default to skepticism. The hidden grader is NOT visible to "
        "you. A result with no real verification step should not exceed "
        "~K, and even a verified-looking one should rarely exceed ~R "
        "unless the verification clearly matches the task's stated "
        "success criterion.",
        "  * Treat the agent's prose declarations (\"done!\", \"all tests "
        "pass\") as ZERO evidence. Ground your score in the actual actions "
        "and the actual output you can see.",
        "",
        "EXPECTED PATTERNS — successive checkpoints do NOT have to rise:",
        "  * On a trajectory that genuinely solves the task, scores "
        "typically rise from A toward T.",
        "  * On a trajectory committed to a WRONG approach, scores should "
        "PLATEAU once the wrong artifact is in place.",
        "  * If the agent regresses (breaks something that worked), scores "
        "should DECREASE.",
        "",
        "The N checkpoints to score are:",
    ]
    for i, k in enumerate(checkpoint_steps, start=1):
        out.append(f"  Checkpoint {i} = state right after Agent Step {k}")
    out.append("")
    out.append(
        "Score each checkpoint INDEPENDENTLY based on the agent's current "
        "best attempt at that point in the trajectory. Output EXACTLY N "
        "lines and nothing else, in the format:")
    for i in range(1, n + 1):
        out.append(f"<c{i}>LETTER</c{i}>")
    out.append("")
    out.append("where each LETTER is a single letter from A to T.")
    return "\n".join(out)


def _expected_value_from_alts(
        alts: Sequence[Tuple[str, float]]) -> Optional[float]:
    """Expectation over the letter values present in one position's top-K
    logprob alternatives; None if no scale letter appears."""
    vals_to_lp = {}
    for tok_str, lp in alts:
        t = (tok_str or "").lstrip()
        if not t:
            continue
        c = t[0]
        if c in LETTER_TO_VALUE:
            v = LETTER_TO_VALUE[c]
            if v not in vals_to_lp or lp > vals_to_lp[v]:
                vals_to_lp[v] = lp
    if not vals_to_lp:
        return None
    mx = max(vals_to_lp.values())
    probs = {v: math.exp(lp - mx) for v, lp in vals_to_lp.items()}
    total = sum(probs.values())
    return sum(v * p / total for v, p in probs.items())


def extract_progress_scores(text, tokens, position_logprobs,
                            n: int) -> List[Optional[float]]:
    """Decode the n checkpoint scores from one verifier response: logprob
    expectation at the answer position after each `<c{i}>` tag, with a
    text-parsing fallback."""
    scores: List[Optional[float]] = [None] * n
    if tokens and position_logprobs:
        joined = ""
        positions_after = []
        for j, tok in enumerate(tokens):
            joined += tok
            positions_after.append((len(joined), j + 1))
        for i in range(1, n + 1):
            tag = f"<c{i}>"
            idx = joined.find(tag)
            if idx < 0:
                continue
            target_char = idx + len(tag)
            for end_char, next_pos in positions_after:
                if end_char > target_char:
                    answer_pos = next_pos - 1
                    if 0 <= answer_pos < len(position_logprobs):
                        v = _expected_value_from_alts(
                            position_logprobs[answer_pos])
                        if v is not None:
                            scores[i - 1] = v
                    break
    # Fallback: tagged letters in the text, then bare one-letter lines.
    for i in range(1, n + 1):
        if scores[i - 1] is None:
            m = re.search(rf"<c{i}>\s*([A-Ta-t])\s*</c{i}>", text or "")
            if m:
                scores[i - 1] = LETTER_TO_VALUE[m.group(1)]
    if any(s is None for s in scores):
        bare = [ln.strip() for ln in (text or "").splitlines()
                if ln.strip() and len(ln.strip()) == 1
                and ln.strip() in LETTER_TO_VALUE]
        if len(bare) == n:
            for i, c in enumerate(bare):
                if scores[i] is None:
                    scores[i] = LETTER_TO_VALUE[c]
    return scores


def track(
    problem: str,
    steps: Sequence[str],
    *,
    checkpoint_steps: Optional[Sequence[int]] = None,
    n_evaluations: int = 1,
    max_workers: int = 8,
    model: str = DEFAULT_MODEL,
    client: Any = None,
) -> ProgressResult:
    """Score an agent trajectory's progress after each step.

    One verifier call scores every checkpoint (repeated `n_evaluations`
    times and averaged), so cost is O(K) calls regardless of trajectory
    length.

    Args:
        problem: the task instruction shown to the verifier.
        steps: the agent's steps, one string per step (action + observed
            output). Truncate very long observations yourself if needed.
        checkpoint_steps: 1-indexed step numbers to score. Defaults to the
            interior steps ``2 .. T-1`` (the first and last step anchor the
            scale), or every step for trajectories with fewer than 3 steps.
        n_evaluations: independent repeats K; the returned curve is their
            mean.
        max_workers: concurrency for the K repeats.
        model: verifier model name.
        client: a pre-built ``google-genai`` client (optional).

    Returns:
        A `ProgressResult` — ``.steps``, ``.scores`` (the progress curve in
        [0, 1]), ``.per_rep_scores``, ``.final``.

    Raises:
        MissingAPIKeyError: no credentials found and no `client` given.
    """
    t = len(steps)
    if t == 0:
        raise ValueError("need at least one step")
    if checkpoint_steps is None:
        checkpoint_steps = list(range(2, t)) if t > 2 else list(range(1, t + 1))
    else:
        checkpoint_steps = list(checkpoint_steps)
        bad = [k for k in checkpoint_steps if not 1 <= k <= t]
        if bad:
            raise ValueError(f"checkpoint_steps out of range 1..{t}: {bad}")
    if n_evaluations < 1:
        raise ValueError("n_evaluations must be >= 1")
    if client is None:
        client = create_gemini_client()

    n = len(checkpoint_steps)
    prompt = build_progress_prompt(
        problem, format_steps(steps), t, checkpoint_steps)

    def one_rep(_):
        text, tokens, position_logprobs = call_gemini(client, prompt, model)
        return extract_progress_scores(text, tokens, position_logprobs, n)

    if n_evaluations == 1:
        per_rep = [one_rep(0)]
    else:
        with ThreadPoolExecutor(
                max_workers=min(max_workers, n_evaluations)) as executor:
            per_rep = list(executor.map(one_rep, range(n_evaluations)))

    scores = []
    for i in range(n):
        vals = [v for rep in per_rep if (v := rep[i]) is not None]
        scores.append(sum(vals) / len(vals) if vals else 0.5)
    return ProgressResult(list(checkpoint_steps), scores, per_rep)


class ProgressTracker:
    """Online progress tracking for a still-running agent.

    Feed steps as the agent produces them; each `update` scores the
    trajectory prefix accumulated so far, so the verifier structurally
    cannot be influenced by future steps (unlike offline `track`, which
    shows the whole trajectory per call). Use it to stop hopeless rollouts
    early or to decide when to branch/resample.

    Cost: one verifier call per repeat per update — a T-step run costs
    T x n_evaluations calls, versus n_evaluations for offline `track`.

    Example:
        tracker = llm_verifier.ProgressTracker(problem, n_evaluations=4)
        for step in agent_steps():
            score = tracker.update(step)     # progress in [0, 1] so far
            if tracker.steps[-1] >= 8 and score < 0.05:
                break                        # abandon early
        result = tracker.result()            # same shape as track()'s

    Raises `MissingAPIKeyError` at construction if no credentials are found
    and no `client` is given.
    """

    def __init__(
        self,
        problem: str,
        *,
        n_evaluations: int = 1,
        max_workers: int = 8,
        model: str = DEFAULT_MODEL,
        client: Any = None,
    ) -> None:
        if n_evaluations < 1:
            raise ValueError("n_evaluations must be >= 1")
        self.problem = problem
        self.n_evaluations = n_evaluations
        self.max_workers = max_workers
        self.model = model
        self.client = client if client is not None else create_gemini_client()
        self.steps: List[int] = []
        self.scores: List[float] = []
        self._step_texts: List[str] = []
        self._per_step_reps: List[List[Optional[float]]] = []

    def update(self, step: str) -> float:
        """Append the agent's latest step and return the progress score of
        the trajectory so far (mean over `n_evaluations` repeats)."""
        self._step_texts.append(str(step))
        k = len(self._step_texts)
        prompt = build_progress_prompt(
            self.problem, format_steps(self._step_texts), k, [k])

        def one_rep(_):
            text, tokens, lps = call_gemini(self.client, prompt, self.model)
            return extract_progress_scores(text, tokens, lps, 1)[0]

        if self.n_evaluations == 1:
            reps = [one_rep(0)]
        else:
            with ThreadPoolExecutor(
                    max_workers=min(self.max_workers,
                                    self.n_evaluations)) as executor:
                reps = list(executor.map(one_rep,
                                         range(self.n_evaluations)))

        vals = [v for v in reps if v is not None]
        score = sum(vals) / len(vals) if vals else 0.5
        self.steps.append(k)
        self.scores.append(score)
        self._per_step_reps.append(reps)
        return score

    def result(self) -> ProgressResult:
        """The curve so far as a `ProgressResult` (same shape as `track`'s:
        `per_rep_scores` rows are repeats, columns are steps)."""
        if not self.steps:
            raise ValueError("no steps tracked yet")
        per_rep = [[self._per_step_reps[i][r]
                    for i in range(len(self.steps))]
                   for r in range(self.n_evaluations)]
        return ProgressResult(list(self.steps), list(self.scores), per_rep)
