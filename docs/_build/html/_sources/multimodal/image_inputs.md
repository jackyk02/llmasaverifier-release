# Multimodal Verification with Images

Every LLM-as-a-Verifier entry point — `select`, `compare`, `track`, and `ProgressTracker` — accepts image inputs alongside the text trajectory.
The images are attached to the verifier message itself, so the fine-grained logprob reward works unchanged: the verifier *looks* at the evidence (a screenshot, a rendered plot, a camera frame) instead of trusting the agent's textual claim about it.

Typical uses:

- **GUI / browser agents** — attach screenshots so the verifier scores what is actually on screen, not what the agent narrates.
- **Robotics** — attach per-step camera frames and track progress toward a visual goal (this is the zero-shot setup behind the RoboRewardBench results in [Benchmark Results](../benchmarks/results.md)).
- **Data analysis & charts** — attach the rendered figure a task asked for and let the verifier check it against the instruction.
- **Visual ground truth** — attach a goal or reference image the trajectories are supposed to reproduce.

## The `images` argument

All APIs take the same `images` keyword (type alias `ImagesArg`):

```python
images="frame.png"                      # a single image ...
images=["before.png", "after.png"]      # ... or several, attached in order
```

Each image may be:

- a **local file path** (`str` or `os.PathLike`) — read from disk;
- an **http(s) URL** — fetched once when the call is made;
- **raw bytes** — e.g. a frame you already hold in memory.

The MIME type is sniffed from the image bytes (PNG, JPEG, GIF, WebP).
Images are attached to the verifier message *after* the text prompt, in the order given, and the prompt gains a one-line `**Attached images:** N image(s)…` note so the verifier knows to use them; text-only calls build byte-identical prompts to before.

## Verifier backends

Image inputs require a **multimodal verifier model** on either backend:

| Backend | How | Example |
|---|---|---|
| Gemini (default) | images become inline parts of the request | `gemini-2.5-flash` (the default model) is multimodal out of the box |
| OpenAI-compatible (vLLM / SGLang) | images become base64 `image_url` content parts | serve a multimodal model, e.g. `vllm serve Qwen/Qwen3.5-9B` |

For the OpenAI-compatible backend everything from [Serving Open Models](../get_started/install.md) carries over unchanged — the same `vllm serve Qwen/Qwen3.5-9B` used for text verification handles images too (Qwen3.5 models are natively multimodal), `OPENAI_BASE_URL` selects the backend, the served model is auto-detected, and the [constrained score-tag prefill](../advanced_features/logit_restricted_models.md) keeps the fine-grained reward exact.
The prefill grammar accepts the score letter with or without a leading space; multimodal tokenizers (e.g. Qwen's) put nearly all probability mass on the space-prefixed spelling, and without that allowance the grammar mask would silently discard the model's real distribution.

A text-only model behind the server will reject (or worse, ignore) image content — if scores stop tracking the images, check the served model is actually multimodal.

## The example images

The snippets below use tiny solid-color squares so they run verbatim and only the *image* reveals the right answer — generate them with Pillow:

```python
from PIL import Image

Image.new("RGB", (64, 64), (220, 30, 30)).save("red.png")
Image.new("RGB", (64, 64), (128, 30, 160)).save("purple.png")
Image.new("RGB", (64, 64), (30, 30, 220)).save("blue.png")
```

These exact squares produced every number quoted on this page.

## Best-of-N selection with images

`images` on `select` is task context: every pairwise comparison in the tournament sees the same image(s):

```python
import llm_verifier

result = llm_verifier.select(
    problem="Two images are attached: first a square, then another square. "
            "Report both colors in order.",
    candidates=["First red, then blue.",
                "First blue, then red.",
                "Both squares are green."],
    criteria={"Correctness": "Do the reported colors match the attached "
                             "images, in order?"},
    images=["red.png", "blue.png"],
    n_evaluations=4,
)
print(result.index)   # 0 — only the image reveals the right answer
```

The same applies to `compare`:

```python
r_a, r_b = llm_verifier.compare(
    "Report the dominant color of the square in the attached image.",
    "The square is red.",           # matches the attached image
    "The square is blue.",
    criteria={"Correctness": "Does the answer match the attached image?"},
    images="red.png",
)
# Gemini 2.5 Flash:      R_A = 1.000, R_B = 0.000
# Qwen3.5-9B via vLLM:   R_A = 0.993, R_B = 0.158
```

## Progress tracking with per-step frames

For progress tracking, images can be **task context** (a goal image, attached to every scoring call) or **per-step evidence** (a camera frame after each action):

- `track(problem, steps, images=...)` and `ProgressTracker(problem, images=...)` attach task-context image(s) to every scoring call.
- `ProgressTracker.update(step, images=...)` attaches image(s) to *that step*: the step text gets an `[Image i attached]` marker, the frame is appended to the message, and it stays part of the trajectory for all later updates — so the verifier always sees the full visual history of the prefix.

```python
tracker = llm_verifier.ProgressTracker(
    "Change the on-screen square's color from red to blue. The attached "
    "frame after each step shows the current screen.")

for step, frame in agent_run():          # frame: path, URL, or raw bytes
    score = tracker.update(step, images=frame)
```

On the red → purple → blue toy task above, both backends produce the expected rising curve from the frames alone:

| Backend | step 1 (red) | step 2 (purple) | step 3 (blue) |
|---|---|---|---|
| Gemini 2.5 Flash | 0.000 | 0.175 | 1.000 |
| Qwen3.5-9B via vLLM | 0.002 | 0.263 | 1.000 |

When images are attached, the [progress-scoring prompt](../basic_usage/progress_tracking.md) gains an `**Attached images:**` line explaining the markers; the template is otherwise unchanged.

## Practical notes

- **Every scoring call resends the images.** `select` runs O(Nk²) comparisons and an online tracker resends the accumulated frames on every update — keep images small (downscale screenshots; a 64–512 px side is usually plenty for color/layout judgments).
- **Small open models are noisier.** As with text, average out per-call noise with `n_evaluations` (the two-image `select` above is reliable at `n_evaluations=4` on Qwen3.5-9B; Gemini 2.5 Flash needs less).
- **Video**: sample frames yourself and pass them as ordered images — one frame per step through `ProgressTracker.update` mirrors the RoboRewardBench setup.
- Under the hood the plumbing lives in `llm_verifier/fine_grained_reward.py` (`as_image_list`, `load_image`, and the `images=` parameter of `call_verifier`).
