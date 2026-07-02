# <Your Task> — Verifier Criteria

<!-- Copy this file and replace the content; keep the heading structure.
     HTML comments like this one are stripped before the verifier sees
     anything. Load your file with:

         llm_verifier.select(..., criteria="path/to/your_file.md")

     Preview what the verifier will see (no API key needed):

         python -m llm_verifier path/to/your_file.md                  -->

## Ground Truth Note

<!-- Optional, but recommended: one paragraph the verifier sees on EVERY
     comparison. Best used to tell the verifier which evidence to trust
     (raw tool output, not the agent's narration) and what a correct
     answer may look like. Delete the whole section if you don't need it. -->

Do NOT trust the agent's self-assessment or claims of success.

## Criteria

<!-- One `### id — Name` heading per criterion (em dash, en dash, or hyphen
     between id and name all work; the id becomes part of the score-cache
     key). Everything until the next heading is that criterion's
     instruction. The verifier scores each criterion INDEPENDENTLY, so 2-4
     narrow criteria beat one broad one.

     Write each instruction so a stranger could score with it:
       - say exactly WHERE to look (which commands, fields, files, outputs)
       - say what should score HIGH and what should score LOW
       - say what to IGNORE, so one criterion doesn't leak into another   -->

### correctness — Final Answer Correctness

Compare the agent's final answer against what the task actually asked for.
Check the answer's content, type, and format. Score HIGH only when the answer
is fully supported by output observed in the trajectory; score LOW when it is
asserted without supporting evidence, contradicts the observed output, or
answers a different question. Ignore code style and efficiency.

### verification — Empirical Verification

Look at the commands the agent actually ran and what they printed — not what
the agent claimed in its narration. Reward agents that reproduced the problem,
observed the fix working, and re-ran relevant checks afterwards. Penalize
agents that declared success without running anything, misread their own
output, or made further edits after the last successful check so the final
state is unverified.
