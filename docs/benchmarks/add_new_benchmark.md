# Adding a New Benchmark

Use the verifier for your own task in three steps — Claude Code does the rest (generates the criteria, writes a runner, and selects the best-of-N for you):

1. **Add your data.** Copy your agent trajectories into `data/task_name_trajs/`.
2. **Update naming.** Replace every `task_name` in `add_new_benchmark.md` (at the repository root) with the name of your task.
3. **Spin up Claude Code in the repo** (or Codex, or whatever you like — with permissions disabled) and paste the contents of `add_new_benchmark.md` to let it run.