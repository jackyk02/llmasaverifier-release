"""
Load verifier criteria + ground-truth note from a benchmark prompt file.

Prompt files live in llm_verifier/criteria/<benchmark>.md (shipped with the
package) so they are easy to read and edit without touching code. Expected
layout:

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

import os
import re
from importlib import resources

_CRIT_HEADING = re.compile(r"^(.+?)\s*[—–-]\s*(.+)$")


def _read_criteria(path):
    """Resolve a criteria argument to its file contents.

    Accepts either an existing filesystem path (your own ``my_criteria.md``, or
    an absolute path) or a bare benchmark name (``"swe_bench"``) that maps to a
    prompt bundled inside the installed package — so a plain ``pip install``
    user does not need any files on disk.
    """
    if os.path.isfile(path):
        with open(path) as f:
            return f.read()
    name = os.path.basename(path)
    if not name.endswith(".md"):
        name += ".md"
    bundled = resources.files(__package__).joinpath("criteria", name)
    try:
        return bundled.read_text()
    except (FileNotFoundError, OSError):
        raise FileNotFoundError(
            f"criteria {path!r} not found: not a file on disk, and no bundled "
            f"prompt named {name!r} in llm_verifier/criteria/"
        )


def load_prompts(path):
    """Return (ground_truth_note, criteria) where criteria is a list of
    {"id", "name", "description"} dicts in file order.

    ``path`` may be a filesystem path or a bundled benchmark name (see
    :func:`_read_criteria`)."""
    lines = _read_criteria(path).splitlines()

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
