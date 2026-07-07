# Writing Verifier Criteria

Every verification is scored against explicit **criteria**. The verifier scores each criterion independently and averages the expected scores. 

## Accepted forms

The `criteria` argument of `select` / `compare` accepts any of:

```python
# 1. A bundled benchmark name (loads criteria/<name>.md from the repo)
llm_verifier.select(..., criteria="swe_bench")

# 2. A path to your own criteria file
llm_verifier.select(..., criteria="path/to/your_file.md")

# 3. A {name: description} mapping
llm_verifier.select(..., criteria={
    "Root cause": "Did the agent fix the real cause?",
    "Verification": "Did the agent confirm the fix?",
})

# 4. A list of strings, or of {"id", "name", "description"} dicts
llm_verifier.select(..., criteria=["Did the agent solve the task?"])
```

## Criteria files

A criteria file is a Markdown file with two sections.
Copy `criteria/TEMPLATE.md` and keep the heading structure; HTML comments are stripped before the verifier sees anything.

```markdown
# <Your Task> — Verifier Criteria

## Ground Truth Note

Do NOT trust the agent's self-assessment or claims of success.

## Criteria

### Final Answer Correctness

Compare the agent's final answer against what the task actually asked for.
Check the answer's content, type, and format. Score HIGH only when the answer
is fully supported by output observed in the trajectory; score LOW when it is
asserted without supporting evidence, contradicts the observed output, or
answers a different question. Ignore code style and efficiency.

### Empirical Verification

Look at the commands the agent actually ran and what they printed — not what
the agent claimed in its narration. Reward agents that reproduced the problem,
observed the fix working, and re-ran relevant checks afterwards. Penalize
agents that declared success without running anything, misread their own
output, or made further edits after the last successful check so the final
state is unverified.
```

### Ground Truth Note

Optional, but recommended: one paragraph the verifier sees on **every** comparison.
Best used to tell the verifier which evidence to trust (raw tool output, not the agent's narration) and what a correct answer may look like.
You can also override it per call with the `ground_truth_note` argument.

### Criterion blocks

One `### Criterion Name` heading per criterion; everything until the next heading is its instruction.
Write each so a stranger could score with it:

## Preview what the verifier sees

Preview any criteria file or bundled benchmark exactly as the verifier will see it — no API key needed:

```bash
python -m llm_verifier path/to/your_file.md
python -m llm_verifier swe_bench
```

## Existing criteria

| Benchmark | File | Criterion ids |
|---|---|---|
| Terminal-Bench 2.0 | `criteria/terminal_bench.md` | `specification`, `output_match`, `error_signals` |
| SWE-bench Verified | `criteria/swe_bench.md` | `root_cause`, `code_review`, `verification` |
| MedAgentBench | `criteria/medagentbench.md` | `query`, `consistency`, `structure` |