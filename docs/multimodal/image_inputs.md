# Multi-Modal Supports

Every entry point — `select`, `compare`, `track`, and `ProgressTracker` — accepts images alongside the text trajectory.

## The `images` argument

All APIs take the same `images` keyword — a single image or a list, each a **local file path**, an **http(s) URL**, or **raw bytes**:

```python
images="frame.png"                      # a single image ...
images=["before.png", "after.png"]      # ... or several, attached in order
```

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
# R_A = 0.993, R_B = 0.158
```

## Progress tracking with per-step frames

On a red → purple → blue toy task, the progress curve rises from the frames alone (scores below are Gemini 2.5 Flash):

```python
tracker = llm_verifier.ProgressTracker(
    "Change the on-screen square's color from red to blue. The attached "
    "frame after each step shows the current screen.")

score = tracker.update("Opened the color panel.", images="red.png")            # 0.000
score = tracker.update("Dragged the hue slider halfway.", images="purple.png") # 0.175
score = tracker.update("Saved; the square renders blue.", images="blue.png")   # 1.000
```

## Verifier backends

Image inputs require a **multimodal verifier model** on either backend:

| Backend | How | Example |
|---|---|---|
| Gemini (default) | images become inline parts of the request | `gemini-2.5-flash` (the default model) is multimodal out of the box |
| OpenAI-compatible (vLLM / SGLang) | images become base64 `image_url` content parts | serve a multimodal model, e.g. `vllm serve Qwen/Qwen3.5-9B` |
