"""
Fine-grained reward model.

Rather than collapsing a verifier's judgement into one discrete label (as in
LLM-as-a-Judge), LLM-as-a-Verifier reads the model's probability distribution
over an ordered set of score tokens and takes its expectation. The reward of a
trajectory tau on task t is

    R(t, tau) = (1 / C K) * sum_c sum_k sum_g  p_theta(v_g | t, c, tau) * phi(v_g)

  C  = number of evaluation criteria
  K  = number of repeated verifications
  G  = number of ordered score tokens (granularity)
  v_g = the g-th score token,  phi(v_g) = its scalar value
  p_theta = probability the verifier assigns to token v_g (from logprobs)

This module provides the granularity-20 scale, the Gemini logprob client, the
score-token expectation `extract_score`, the pairwise prompt, and a cached
batch scorer that only scores the pairs a pivot tournament actually needs.
"""

import json
import math
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed


# ---------------------------------------------------------------------------
# g = 20 score-token scale
# ---------------------------------------------------------------------------

GRANULARITY = 20

SCALE = {
    "scale_description": (
        "Rate how likely the agent correctly solved the task on a "
        "20-point scale using letters A through T:\n"
        "  A = clearly and completely succeeded with verified output (best)\n"
        "  B-D = succeeded with only minor issues\n"
        "  E-G = above average, mostly correct with some issues\n"
        "  H-J = uncertain, leans toward success\n"
        "  K-M = uncertain, leans toward failure\n"
        "  N-P = below average, significant issues remain\n"
        "  Q-S = failed with some partial progress\n"
        "  T = clearly and completely failed (worst)"
    ),
    "score_format": "LETTER_A_TO_T",
    "valid_tokens": {
        **{chr(65 + i): float(GRANULARITY - i) for i in range(GRANULARITY)},
        **{chr(97 + i): float(GRANULARITY - i) for i in range(GRANULARITY)},
    },
}


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

def load_dotenv(root_dir):
    env_path = os.path.join(root_dir, ".env")
    if os.path.exists(env_path):
        for line in open(env_path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def create_gemini_client():
    from google import genai
    vertex_key = os.environ.get("VERTEX_API_KEY")
    if vertex_key:
        return genai.Client(vertexai=True, api_key=vertex_key)
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        return genai.Client(api_key=api_key)
    print("Error: set GEMINI_API_KEY or VERTEX_API_KEY in .env or environment")
    sys.exit(1)


def call_gemini(client, prompt, top_logprobs=20):
    """Call Gemini 2.5 Flash with logprobs.
    Returns (text, tokens, position_logprobs)."""
    from google.genai.types import (
        Content, GenerateContentConfig, Part, ThinkingConfig)

    config = GenerateContentConfig(
        max_output_tokens=4096,
        temperature=1.0,
        response_logprobs=True,
        logprobs=top_logprobs,
        thinking_config=ThinkingConfig(thinking_budget=0),
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[Content(role="user", parts=[Part(text=prompt)])],
        config=config,
    )

    text = response.text or ""
    tokens = None
    position_logprobs = None

    candidate = response.candidates[0]
    if candidate.logprobs_result and candidate.logprobs_result.top_candidates:
        position_logprobs = []
        for pos in candidate.logprobs_result.top_candidates:
            alts = [(lp.token, lp.log_probability) for lp in pos.candidates]
            position_logprobs.append(alts)
        if candidate.logprobs_result.chosen_candidates:
            tokens = [c.token
                      for c in candidate.logprobs_result.chosen_candidates]

    return text, tokens, position_logprobs


# ---------------------------------------------------------------------------
# Score-token expectation:  sum_g p(v_g) * phi(v_g), normalized to [0, 1]
# ---------------------------------------------------------------------------

def _find_tag_logprobs(tokens, position_logprobs, tag):
    if not tokens or not position_logprobs:
        return None
    text_so_far = ""
    for i, tok in enumerate(tokens):
        text_so_far += tok
        if text_so_far.rstrip().endswith(tag):
            if i + 1 < len(position_logprobs):
                return position_logprobs[i + 1]
    return None


def extract_score(text, tokens, position_logprobs, tag):
    """Expected score over the verifier's token distribution at `tag`,
    normalized to [0, 1]. Falls back to parsing the literal text token."""
    valid_tokens = SCALE["valid_tokens"]

    tag_lp = _find_tag_logprobs(tokens, position_logprobs, tag)
    probs = {}
    if tag_lp:
        for tok_str, logprob in tag_lp:
            tok = tok_str.strip()
            if tok in valid_tokens:
                val = valid_tokens[tok]
                p = math.exp(logprob)
                probs[val] = max(probs.get(val, 0.0), p)

    if probs:
        unique_vals = sorted(set(valid_tokens.values()))
        min_val, max_val = min(unique_vals), max(unique_vals)
        total_p = sum(probs.values())
        expected = sum(v * p for v, p in probs.items()) / total_p
        return (expected - min_val) / (max_val - min_val) \
            if max_val > min_val else 0.5

    tag_name = tag.strip("<>")
    pattern = rf"<{re.escape(tag_name)}>\s*(.+?)\s*</{re.escape(tag_name)}>"
    match = re.search(pattern, text or "", re.IGNORECASE)
    if match:
        tok = match.group(1).strip()
        raw_val = valid_tokens.get(tok)
        if raw_val is None:
            for vt, val in valid_tokens.items():
                if tok.lower() == vt.lower():
                    raw_val = val
                    break
        if raw_val is not None:
            unique_vals = sorted(set(valid_tokens.values()))
            min_val, max_val = min(unique_vals), max(unique_vals)
            return (raw_val - min_val) / (max_val - min_val) \
                if max_val > min_val else 0.5

    return 0.5


# ---------------------------------------------------------------------------
# Pairwise prompt + single-criterion scoring
# ---------------------------------------------------------------------------

def build_prompt(problem, trace_a, trace_b, criterion, ground_truth_note):
    """One pairwise prompt focused on a single evaluation criterion."""
    return (
        "You are an expert evaluator of AI coding agents. "
        "You will see a task description and two agent trajectories. "
        f"Your job is to evaluate them on ONE specific criterion: "
        f"**{criterion['name']}**.\n\n"
        f"{ground_truth_note}\n\n"
        f"**Task:**\n{problem}\n\n"
        f"**Trajectory A:**\n{trace_a}\n\n"
        f"**Trajectory B:**\n{trace_b}\n\n"
        f"**Evaluation Guideline — {criterion['name']}:**\n"
        f"{criterion['description']}\n\n"
        f"Score each trajectory ONLY on this specific criterion. Ignore other "
        f"aspects of the trajectory that are not relevant to "
        f"\"{criterion['name']}\".\n\n"
        f"**Rating Scale:**\n{SCALE['scale_description']}\n\n"
        "Then output your final scores:\n"
        f"<score_A>{SCALE['score_format']}</score_A>\n"
        f"<score_B>{SCALE['score_format']}</score_B>\n\n"
        "Begin your analysis now."
    )


def score_pair_criterion(client, problem, trace_a, trace_b, criterion,
                         ground_truth_note):
    """Score (A, B) for a single criterion, returning fine-grained
    rewards (R_A, R_B) in [0, 1]."""
    prompt = build_prompt(
        problem, trace_a, trace_b, criterion, ground_truth_note)
    text, tokens, position_logprobs = call_gemini(client, prompt)
    ra = extract_score(text, tokens, position_logprobs, "<score_A>")
    rb = extract_score(text, tokens, position_logprobs, "<score_B>")
    return ra, rb


# ---------------------------------------------------------------------------
# Directed cache key + the per-comparison reward used by the tournament
# ---------------------------------------------------------------------------
#
# Comparisons are *directed*: candidate `a` is shown in slot A and `b` in slot
# B of the verifier prompt. PPT's ring pass relies on this — each candidate is
# placed in A exactly once and in B exactly once around the cycle, so the
# verifier's slot bias cancels. The cache key therefore records the ordered
# pair (a, b); (a, b) and (b, a) are distinct entries.

def cache_key(crit_id, task_name, a, b, rep):
    return f"{crit_id}|{task_name}|{a},{b}|{rep}"


def directed_reward(scores, task_name, a, b, criteria_ids, n_reps):
    """Fine-grained rewards (R_a, R_b) for the directed comparison (a in slot
    A, b in slot B), averaged over all criteria and repeated verifications.
    Missing entries default to a tie (0.5)."""
    if a == b:
        return 0.5, 0.5
    sa = sb = 0.0
    cnt = 0
    for cid in criteria_ids:
        for rep in range(n_reps):
            entry = scores.get(cache_key(cid, task_name, a, b, rep), {})
            sa += entry.get("score_A", 0.5)
            sb += entry.get("score_B", 0.5)
            cnt += 1
    return (sa / cnt, sb / cnt) if cnt else (0.5, 0.5)


class LazyClient:
    """Create the Gemini client on first use, so reproducing a fully cached
    run never needs an API key."""

    def __init__(self):
        self._client = None

    def get(self):
        if self._client is None:
            self._client = create_gemini_client()
        return self._client


# ---------------------------------------------------------------------------
# Cached batch scoring — only the directed pairs PPT actually needs
# ---------------------------------------------------------------------------

def score_directed_pairs(lazy_client, tasks, needed_pairs, criteria,
                         ground_truth_note, n_reps, max_workers, cache_file):
    """Score every (criterion, rep) for the requested directed (task, a, b)
    pairs and merge into the cache on disk.

    `needed_pairs` maps task_name -> iterable of directed (a, b) comparisons.
    Only comparisons missing from the cache trigger API calls, so the cost of a
    run scales with the PPT comparison count, not C(N, 2). Returns the cache."""
    cached = {}
    if cache_file and os.path.exists(cache_file):
        with open(cache_file) as f:
            cached = json.load(f)

    jobs = []
    for task_name, pairs in needed_pairs.items():
        trials = tasks[task_name]
        for a, b in pairs:
            for crit in criteria:
                for rep in range(n_reps):
                    key = cache_key(crit["id"], task_name, a, b, rep)
                    if key not in cached:
                        jobs.append((key, trials[a]["problem"],
                                     trials[a]["trace"], trials[b]["trace"],
                                     crit))

    if not jobs:
        print(f"  All scores cached ({len(cached)} entries)")
        return cached

    print(f"  {len(jobs)} scoring jobs ({len(cached)} cached)")

    from tqdm import tqdm
    client = lazy_client.get()
    errors = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(score_pair_criterion, client, prob, ta, tb, crit,
                            ground_truth_note): key
            for key, prob, ta, tb, crit in jobs
        }
        pbar = tqdm(as_completed(futures), total=len(futures), desc="Scoring")
        save_every = max(1, len(futures) // 20)
        done = 0

        for future in pbar:
            key = futures[future]
            try:
                ra, rb = future.result()
                cached[key] = {"score_A": ra, "score_B": rb}
            except Exception as e:
                errors += 1
                cached[key] = {"score_A": 0.5, "score_B": 0.5}
                if errors <= 3:
                    print(f"\n  Error: {e}")
            done += 1
            pbar.set_postfix(errors=errors)
            if cache_file and done % save_every == 0:
                with open(cache_file, "w") as f:
                    json.dump(cached, f)

    if cache_file:
        with open(cache_file, "w") as f:
            json.dump(cached, f)

    print(f"  Done ({errors} errors)")
    return cached
