#!/usr/bin/env python3
"""Render a SUCCESS-vs-FAILED progress-trace figure from saved track() curves.

Reproduces the format of `figures/progress_pytorch_model_cli.png` (minus the
per-step annotation boxes, which were added by hand for the README): steps
normalized to [0, 1], scores normalized by the global max of the two mean
curves, and a ±1 std band over the `n_verifications` repeats.

Usage:
    python plot_progress.py [curves.json] [out.png]

Defaults to the shipped example (`cache/progress_pytorch-model-cli_k16.json`)
and writes `progress_traces.png`. The JSON holds one entry per curve —
`{"succ": {...}, "fail": {...}}` — each with `checkpoint_steps` and
`per_rep_scores` (save them from a `llm_verifier.track()` result:
`result.steps` and `result.per_rep_scores`). Requires matplotlib.
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IN = os.path.join(ROOT_DIR, "cache",
                          "progress_pytorch-model-cli_k16.json")
GREEN, RED = "#3a7d34", "#c0392b"


def curve(entry):
    """(normalized steps, mean over reps, std over reps) for one run."""
    reps = np.array([[0.0 if v is None else v for v in rep]
                     for rep in entry["per_rep_scores"]], float)
    steps = np.array(entry["checkpoint_steps"], float)
    x = (steps - steps.min()) / max(steps.max() - steps.min(), 1.0)
    return x, reps.mean(0), reps.std(0)


def main():
    in_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IN
    out_png = sys.argv[2] if len(sys.argv) > 2 else "progress_traces.png"
    with open(in_path) as f:
        payload = json.load(f)

    runs = []
    if "succ" in payload:
        runs.append(("SUCCESS", GREEN, "-o", 0.18, curve(payload["succ"])))
    if "fail" in payload:
        runs.append(("FAILED", RED, "-s", 0.15, curve(payload["fail"])))
    if not runs:
        sys.exit(f"no 'succ' or 'fail' entry in {in_path}")

    # Normalize scores by the global max of the mean curves.
    g = max(m.max() for _, _, _, _, (_, m, _) in runs) or 1.0

    plt.rcParams.update({"font.size": 13, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for label, color, style, band_alpha, (x, m, s) in runs:
        m, s = m / g, s / g
        ax.plot(x, m, style, color=color, lw=2.4, ms=7, label=label, zorder=3)
        ax.fill_between(x, np.clip(m - s, 0, None), m + s, color=color,
                        alpha=band_alpha, zorder=1)

    ax.set_xlabel("Normalized Step", fontsize=16, fontweight="bold")
    ax.set_ylabel("Normalized Verifier Score", fontsize=16, fontweight="bold")
    ax.set_xlim(-0.03, 1.05)
    ax.set_ylim(-0.05, 1.22)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
