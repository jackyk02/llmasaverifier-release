"""
Load verifier criteria + ground-truth note from a benchmark prompt file.

Prompt files live in prompts/<benchmark>.md so they are easy to read and edit
without touching code. Expected layout:

    # <title>

    ## Ground Truth Note

    <one paragraph the verifier always sees>

    ## Criteria

    ### <criterion_id> — <Criterion Name>

    <criterion description, any number of paragraphs>

    ### <criterion_id> — <Criterion Name>

    ...

The section headings ("Ground Truth Note", "Criteria") are matched
case-insensitively. Each `### id — Name` heading accepts an em dash ("—"),
en dash ("–"), or hyphen ("-") between the id and the name.
"""

import re

_CRIT_HEADING = re.compile(r"^(.+?)\s*[—–-]\s*(.+)$")


def load_prompts(path):
    """Return (ground_truth_note, criteria) where criteria is a list of
    {"id", "name", "description"} dicts in file order."""
    with open(path) as f:
        lines = f.read().splitlines()

    ground_truth_note = ""
    criteria = []

    section = None      # "ground_truth" | "criteria" | None
    cur = None          # current criterion dict
    buf = []            # accumulated body lines

    def flush():
        nonlocal ground_truth_note, cur, buf
        text = "\n".join(buf).strip()
        if section == "ground_truth" and not ground_truth_note:
            ground_truth_note = text
        elif cur is not None:
            cur["description"] = text
            criteria.append(cur)
            cur = None
        buf = []

    for line in lines:
        if line.startswith("## ") and not line.startswith("### "):
            flush()
            heading = line[3:].strip().lower()
            if "ground truth" in heading:
                section = "ground_truth"
            elif "criteri" in heading:
                section = "criteria"
            else:
                section = None
        elif line.startswith("### ") and section == "criteria":
            flush()
            heading = line[4:].strip()
            m = _CRIT_HEADING.match(heading)
            if m:
                cur = {"id": m.group(1).strip(), "name": m.group(2).strip()}
            else:
                cur = {"id": heading, "name": heading}
        elif line.startswith("# "):
            continue
        else:
            buf.append(line)
    flush()

    return ground_truth_note, criteria


def select_criteria(criteria, ids):
    """Subset + order `criteria` by the given list of ids. If `ids` is falsy,
    return all criteria in file order."""
    if not ids:
        return list(criteria)
    by_id = {c["id"]: c for c in criteria}
    missing = [cid for cid in ids if cid not in by_id]
    if missing:
        raise KeyError(f"criteria not found in prompt file: {missing}")
    return [by_id[cid] for cid in ids]
