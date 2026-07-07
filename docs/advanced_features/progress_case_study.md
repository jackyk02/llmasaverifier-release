# Progress Tracking Case Study

The progress curve separates successful from failed runs long before the task's grader does — a signal you can act on mid-rollout.

Below, we track two Terminus-2 runs of the Terminal-Bench task `pytorch-model-cli`. The successful trajectory exhibits consistently increasing verifier scores, whereas the failed trajectory is characterized by erroneous behaviors, resulting in lower scores throughout the execution. Reproduce it with:

```bash
python terminal_bench_progress.py    # scores both runs then plots
```

<img src="../_static/image/progress_pytorch_model_cli.png" alt="Progress curves for two pytorch-model-cli runs" width="100%">

See [Progress Tracking](../basic_usage/progress_tracking.md) for the `track` / `ProgressTracker` API used here.
