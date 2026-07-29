"""
Day 5 — Eval Set + Baseline Scoring
--------------------------------------
Runs the 18-question eval set (eval_set.json) through:

1. Retrieval scoring (no API key needed — testable right now):
   for each question with a known expected_doc, check whether that doc
   actually shows up in the top-k retrieved chunks. This isolates
   retrieval quality from generation quality.

2. Full pipeline scoring (needs your API key):
   runs each question through Day 3's Researcher->Analyst->Critic graph
   and tabulates verdicts + failure_type breakdown across all 18.

Output: a summary you can quote directly in an interview
("baseline retrieval accuracy was X%, Y% of failures were hallucination
vs Z% retrieval-miss") plus a JSONL log that Day 6's Optimizer reads.

Run locally:
    python day5_eval.py
"""

import json
from collections import Counter

from day1_rag_agent import build_vector_store, retrieve
from day3_multi_agent import build_graph, log_run


def load_eval_set(path="eval_set.json"):
    with open(path) as f:
        return json.load(f)


def score_retrieval(eval_set, k=2):
    """
    Doesn't need an LLM call at all — pure retrieval quality check.
    Runnable and scoreable right now, no API key needed.
    """
    collection = build_vector_store()
    results = []
    hits, total_scored = 0, 0

    for item in eval_set:
        retrieved = retrieve(collection, item["query"], k=k)
        retrieved_ids = {cid for cid, _ in retrieved}

        expected = item.get("expected_doc")
        if expected is None:
            # out-of-scope question — nothing to score against, just record what came back
            results.append({**item, "retrieved_ids": list(retrieved_ids), "hit": None})
            continue

        expected_set = set(expected.split(","))
        hit = bool(expected_set & retrieved_ids)
        hits += int(hit)
        total_scored += 1
        results.append({**item, "retrieved_ids": list(retrieved_ids), "hit": hit})

    accuracy = hits / total_scored if total_scored else 0.0
    return results, accuracy


def score_full_pipeline(eval_set, log_path="eval_run_log.jsonl"):
    """Runs the full multi-agent graph on every question. Needs API key to
    produce real verdicts; without one, every run logs as 'unclassified'."""
    app = build_graph()
    open(log_path, "w").close()  # fresh log for this eval run

    verdicts = []
    for item in eval_set:
        state = app.invoke({
            "query": item["query"], "retrieved": [], "draft_answer": "",
            "verdict": "", "failure_type": "", "final_answer": "",
        })
        log_run(state, log_path=log_path)
        verdicts.append(state["verdict"])

    return Counter(verdicts)


def failure_type_breakdown(log_path="eval_run_log.jsonl"):
    types = Counter()
    with open(log_path) as f:
        for line in f:
            entry = json.loads(line)
            if entry["failure_type"]:
                types[entry["failure_type"]] += 1
    return types


if __name__ == "__main__":
    eval_set = load_eval_set()

    print(f"=== Retrieval scoring ({len(eval_set)} questions) ===")
    retrieval_results, accuracy = score_retrieval(eval_set)
    print(f"Retrieval accuracy (in-scope questions): {accuracy:.0%}")
    for r in retrieval_results:
        tag = "HIT " if r["hit"] else ("MISS" if r["hit"] is False else "N/A ")
        print(f"  [{tag}] {r['id']}: {r['query'][:60]}")

    print(f"\n=== Full pipeline run ===")
    verdict_counts = score_full_pipeline(eval_set)
    print(f"Verdicts: {dict(verdict_counts)}")

    breakdown = failure_type_breakdown()
    print(f"Failure-type breakdown: {dict(breakdown)}")

    print(
        "\nNote: verdict/failure numbers are only meaningful once you run "
        "this with a real ANTHROPIC_API_KEY set — without one, every run "
        "logs as 'unknown/unclassified' since the Critic can't actually "
        "judge anything. Retrieval accuracy above IS real and testable "
        "with no key."
    )
