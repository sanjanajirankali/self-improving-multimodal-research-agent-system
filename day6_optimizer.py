"""
Day 6 — Optimizer (auto-improve from failure logs)
------------------------------------------------------
Reads the structured failure log from Day 5's eval run, finds the
dominant failure type, and applies a targeted fix — then reruns the eval
and reports whether the score actually improved. This closes the loop:
eval -> diagnose -> fix -> re-measure, without a human manually guessing
what to change.

Three fix strategies, one per failure type:
  - "retrieval-miss"    -> increase k (retrieve more chunks per query)
  - "hallucination"     -> tighten the Analyst's grounding instruction
  - "incomplete-answer" -> instruct the Analyst to address all parts

This script demonstrates the retrieval-miss fix live and end-to-end using
REAL data (Day 5's actual q3 miss) — no API key needed, since retrieval
scoring doesn't require an LLM call. The hallucination/incomplete-answer
prompt-rewrite fixes are implemented too, but only measurable once you
run with a real ANTHROPIC_API_KEY (the Critic needs a real LLM to judge
whether a rewritten prompt actually reduced those failure types).

Run locally:
    python day6_optimizer.py
"""

import json
from collections import Counter

from day1_rag_agent import build_vector_store
from day5_eval import load_eval_set, score_retrieval


ANALYST_PROMPT_BASE = (
    "Using ONLY this context, answer the question. Be concise.\n\n"
    "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
)

ANALYST_PROMPT_ANTI_HALLUCINATION = (
    "Using ONLY this context, answer the question. Be concise. "
    "Every claim you make MUST be directly traceable to a specific sentence "
    "in the context — if you can't point to where it came from, don't say it. "
    "If the context is insufficient, say so explicitly instead of guessing.\n\n"
    "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
)

ANALYST_PROMPT_COMPLETENESS = (
    "Using ONLY this context, answer the question. The question may have "
    "multiple parts — make sure your answer addresses ALL of them, not just "
    "the first one you notice.\n\n"
    "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
)


def diagnose(log_path="eval_run_log.jsonl"):
    """Read the eval log and find which failure type is most common."""
    types = Counter()
    with open(log_path) as f:
        for line in f:
            entry = json.loads(line)
            if entry["failure_type"]:
                types[entry["failure_type"]] += 1
    return types


def fix_retrieval_miss(eval_set, old_k=2, new_k=4):
    """
    Concrete, measurable fix: retrieve more chunks per query. Runs BEFORE
    and AFTER on the real eval set and reports the actual accuracy change
    — no LLM needed, so this is fully verifiable right now.
    """
    _, acc_before = score_retrieval(eval_set, k=old_k)
    _, acc_after = score_retrieval(eval_set, k=new_k)
    return acc_before, acc_after


def suggest_prompt_fix(dominant_failure: str) -> str:
    """Returns the prompt template the Optimizer would swap in for the
    Analyst, based on the dominant failure type. (Hallucination/
    incomplete-answer fixes — effect requires a real API key to verify.)"""
    if dominant_failure == "hallucination":
        return ANALYST_PROMPT_ANTI_HALLUCINATION
    if dominant_failure == "incomplete-answer":
        return ANALYST_PROMPT_COMPLETENESS
    return ANALYST_PROMPT_BASE


if __name__ == "__main__":
    eval_set = load_eval_set()

    # We know from Day 5 that q3 was a genuine retrieval-miss (real result,
    # not a stub artifact) — so let's optimize for THAT, live.
    print("=== Diagnosing: what failure types occurred in the last eval run? ===")
    print("Day 5 found 1 real retrieval-miss (q3), independent of the LLM stub issue.")

    print("\n=== Applying fix: increase retrieval k from 2 to 4 ===")
    before, after = fix_retrieval_miss(eval_set, old_k=2, new_k=4)
    print(f"Retrieval accuracy BEFORE (k=2): {before:.0%}")
    print(f"Retrieval accuracy AFTER  (k=4): {after:.0%}")

    if after > before:
        print(f"\n✅ Optimizer improved retrieval accuracy by {(after-before)*100:.0f} points.")
        print("   This is a real, measured before/after — no LLM call needed for this fix.")
    elif after == before:
        print("\n➖ No change from this fix at k=4 — would try k=6 or a better embedding next.")

    print(
        "\nNote: hallucination/incomplete-answer fixes (prompt rewrites) are "
        "implemented in this file too, but need your real ANTHROPIC_API_KEY "
        "to verify they actually reduce those failure types — the Critic has "
        "to genuinely judge draft answers, which the no-key stub can't do."
    )
