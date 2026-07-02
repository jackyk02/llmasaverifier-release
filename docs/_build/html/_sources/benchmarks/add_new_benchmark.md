# Adding a New Benchmark

Use the verifier for your own task in three steps — Claude Code does the rest (generates the criteria, writes a runner, and selects the best-of-N for you):

1. **Add your data.** Copy your agent trajectories into `data/task_name_trajs/`.
2. **Update naming.** Replace every `task_name` in `add_new_benchmark.md` (at the repository root) with the name of your task.
3. **Spin up Claude Code in the repo** (or Codex, or whatever you like — with permissions disabled) and paste the contents of `add_new_benchmark.md` to let it run.

The rest of this page describes what that workflow produces, in case you prefer to do it by hand.

## 1. Lay out the data

The candidate trajectories, each with its ground-truth reward (success / failure), live in `data/task_name_trajs/`.
The layout should make three things discoverable: the task/problem prompt, each candidate trajectory, and the success/failure label.

## 2. Generate criteria

Write `criteria/task_name.md` using the exact format the loader expects (see [Writing Verifier Criteria](../basic_usage/criteria.md)):

- `## Ground Truth Note`
- `## Criteria`
- one `### <id> — <Name>` block per criterion

Aim for **2–3 criteria** that are *decidable from the trajectory alone* and that target the task's common failure modes.
The criteria should avoid label leakage or reward hacking: they should evaluate observable behavior in the trajectory.

Preview the result without an API key:

```bash
python -m llm_verifier criteria/task_name.md
```

## 3. Write a runner

Create `adapt_run.py` that:

- loads the candidate trajectories from `data/task_name_trajs/` into a list of strings,
- sets `problem` to the task prompt,
- calls:

```python
llm_verifier.select(
    problem, trajectories,
    criteria="task_name",
    n_evaluations=8,   # K
    pivots=2,            # k
    seed=0,
    max_workers=50,
)
```

- prints `result.index`, `result.best`, `result.scores`, and `result.n_comparisons`.

Since ground-truth rewards are available, also report whether the selected trajectory was actually a success (look up the label for `result.index`), and save the run summary (chosen index, per-trajectory scores) for later comparison.

## 4. (Optional) Register it

To make the task runnable as `python run.py task_name`, add an entry to `BENCHMARKS` in `llm_verifier/benchmarks.py` with a loader in `llm_verifier/loaders.py`:

```python
"task_name": Benchmark(
    name="MY TASK  (agent, xN)",
    loader="my_loader",
    prompts="task_name",
    criteria=["criterion_1", "criterion_2"],
    cache="cache/cache_task_name.json",
    results="results/task_name.txt",
    data={"trajs_dir": "data/task_name_trajs"},
),
```
