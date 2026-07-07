# API Reference

The public API is exposed at the top of the `llm_verifier` package:

```python
from llm_verifier import (
    select, compare, track, ProgressTracker,
    VerifierResult, ProgressResult,
    MissingAPIKeyError, GRANULARITY, DEFAULT_MODEL, load_dotenv,
)
```

## `select`

```python
def select(
    problem: str,
    candidates: Sequence[str],
    *,
    criteria: CriteriaArg,
    images: Optional[ImagesArg] = None,
    ground_truth_note: Optional[str] = None,
    n_evaluations: int = 8,
    pivots: int = 2,
    seed: int = 0,
    max_workers: int = 50,
    model: str = DEFAULT_MODEL,
    cache: Optional[str] = None,
    progress: Optional[bool] = None,
    on_error: str = "tie",
    client: Any = None,
) -> VerifierResult
```

Select the best of N agent trajectories for a single task.
Scores directed pairs of trajectories with the fine-grained reward and aggregates them with a Probabilistic Pivot Tournament (PPT), so the cost is O(Nk²) verifier comparisons rather than the O(N²) of a full round-robin.
Identical inputs with the same `seed` run the identical tournament.

**Arguments**

- `problem`: the task description shown to the verifier.
- `candidates`: list of N agent trajectories (strings) to rank.
- `criteria`: a bundled benchmark name (e.g. `"swe_bench"`), a path to a `*.md` criteria file, a `{name: description}` dict, or a list of strings / `{"id", "name", "description"}` dicts.
- `images`: task-context image(s) the verifier sees with every comparison — a single image or a list ([`ImagesArg`](#constants-and-helpers)); requires a multimodal verifier model.
- `ground_truth_note`: optional note the verifier always sees; defaults to the note parsed from the prompt file (or empty).
- `n_evaluations`: repeated verifications K per criterion.
- `pivots`: number of pivots k in the tournament. Keep k small relative to `len(candidates)` — cost grows as O(Nk²), and k ≥ N degenerates to a full round-robin (k is clamped to N).
- `seed`: seed for the random ring pass.
- `max_workers`: concurrency for verifier calls.
- `model`: verifier model name (default `gemini-2.5-flash`).
- `cache`: optional path to a JSON score cache. Re-running with the same cache re-scores only the comparisons not seen before.
- `progress`: show a progress bar / log lines. Default (`None`) shows progress only when stderr is a TTY.
- `on_error`: `"tie"` scores a failed verifier call 0.5/0.5 for this run (never persisted to the cache); `"raise"` re-raises it.
- `client`: a pre-built `openai` or `google-genai` client (optional); by default the backend is picked from the environment (`OPENAI_BASE_URL`, else `VERTEX_API_KEY`) and must expose token-level logprobs.

**Returns** a [`VerifierResult`](#verifierresult) whose `.index` / `.best` is the chosen trajectory and `.ranking` orders all trajectories best-first.

**Raises** `MissingAPIKeyError` if there are no credentials and no `client` given (only when uncached comparisons actually need scoring); `ValueError` on an empty trajectory list.

## `compare`

```python
def compare(
    problem: str,
    trace_a: str,
    trace_b: str,
    *,
    criteria: CriteriaArg,
    images: Optional[ImagesArg] = None,
    ground_truth_note: Optional[str] = None,
    n_evaluations: int = 1,
    max_workers: int = 8,
    model: str = DEFAULT_MODEL,
    client: Any = None,
) -> Tuple[float, float]
```

Fine-grained rewards `(R_A, R_B)` in [0, 1] for one directed comparison.
The verifier sees `trace_a` in slot A and `trace_b` in slot B; rewards are averaged over all criteria and `n_evaluations` repeats.
`images` accepts the same forms as `select`'s (one image or a list — paths, URLs, or bytes) and is attached as task context.
This is the raw pairwise reward `select` is built on — note the single directed call does not cancel slot bias the way `select`'s ring pass does.
A failed verifier call raises (there is no tie fallback here).

## `track`

```python
def track(
    problem: str,
    steps: Sequence[str],
    *,
    images: Optional[ImagesArg] = None,
    checkpoint_steps: Optional[Sequence[int]] = None,
    n_evaluations: int = 1,
    max_workers: int = 8,
    model: str = DEFAULT_MODEL,
    client: Any = None,
) -> ProgressResult
```

Score an agent trajectory's progress after each step.
One verifier call scores every checkpoint (repeated `n_evaluations` times and averaged), so cost is O(K) calls regardless of trajectory length.

**Arguments**

- `problem`: the task instruction shown to the verifier.
- `steps`: the agent's steps, one string per step (action + observed output). Truncate very long observations yourself if needed.
- `images`: task-context image(s) attached to every scoring call ([`ImagesArg`](#constants-and-helpers)); for per-step frames, use `ProgressTracker` and pass images to each `update`.
- `checkpoint_steps`: 1-indexed step numbers to score. Defaults to the interior steps `2 .. T-1` (the first and last step anchor the scale), or every step for trajectories with fewer than 3 steps.
- `n_evaluations`: independent repeats K; the returned curve is their mean.
- `max_workers`: concurrency for the K repeats.

**Returns** a [`ProgressResult`](#progressresult) — `.steps`, `.scores` (the progress curve in [0, 1]), `.per_rep_scores`, `.final`.

## `ProgressTracker`

```python
class ProgressTracker:
    def __init__(
        self,
        problem: str,
        *,
        images: Optional[ImagesArg] = None,
        n_evaluations: int = 1,
        max_workers: int = 8,
        model: str = DEFAULT_MODEL,
        client: Any = None,
    ) -> None
```

Online progress tracking for a still-running agent.
Feed steps as the agent produces them; each `update` scores the trajectory prefix accumulated so far, so the verifier structurally cannot be influenced by future steps.
Cost: one verifier call per repeat per update — a T-step run costs `T × n_evaluations` calls, versus `n_evaluations` for offline `track`.
Raises `MissingAPIKeyError` at construction if no credentials are found and no `client` is given.

Pass `images` at construction for task-context image(s) — e.g. a goal image or reference screenshot.

**Methods and attributes**

- `update(step: str, images=None) -> float`: append the agent's latest step and return the progress score of the trajectory so far (mean over `n_evaluations` repeats). `images` (one image or a list — e.g. a camera frame after this step) is attached to this step: the step text gets an `[Image i attached]` marker and the image stays part of the trajectory for all later updates.
- `result() -> ProgressResult`: the curve so far, same shape as `track()`'s.
- `steps` / `scores`: the checkpoint indices and scores accumulated so far.

## `VerifierResult`

```python
@dataclass
class VerifierResult:
    index: int            # index of the winning trajectory in the input list
    best: str             # the winning trajectory itself
    scores: list[float]   # per-trajectory mean preference (w_i / c_i)
    n_comparisons: int    # number of directed verifier comparisons run
    criteria: list[str]   # the criterion ids used for scoring

    @property
    def ranking(self) -> list[int]:
        """Trajectory indices sorted best-first by tournament score."""
```

## `ProgressResult`

```python
@dataclass
class ProgressResult:
    steps: list[int]                        # checkpoint step numbers
    scores: list[float]                     # mean progress per checkpoint, in [0, 1]
    per_rep_scores: list[list[float|None]]  # rows are repeats, columns are steps

    @property
    def final(self) -> float:
        """Progress score at the last checkpoint."""
```

## Constants and helpers

- `ImagesArg` — the type of every `images` argument: one image or a sequence, each a local file path, an http(s) URL, or raw image bytes; attached after the text prompt, in order. See [Multi-Modal Supports](../multimodal/image_inputs.md).
- `DEFAULT_MODEL = "gemini-2.5-flash"` — the default verifier model.
- `GRANULARITY = 20` — the number of score tokens G (the letter scale A–T).
- `load_dotenv(root_dir=None)` — load `VERTEX_API_KEY` / `OPENAI_BASE_URL` (and friends) from a `.env` file.
- `MissingAPIKeyError` — raised when a verifier call needs a backend and neither `OPENAI_BASE_URL` nor `VERTEX_API_KEY` is configured.

## Command line

```bash
python -m llm_verifier <criteria.md | benchmark_name>   # preview criteria (no API key needed)
python run.py [benchmark] [--pivots K] [--n-evaluations K] [--seed S] [--max-workers W]
```
