"""
Day 3 — Multi-Agent Split (Researcher -> Analyst -> Critic)
-------------------------------------------------------------
Splits the single RAG agent from Day 1/2 into three roles, orchestrated as
a LangGraph state graph:

    Researcher  -> retrieves relevant chunks for the query
    Analyst     -> drafts an answer using ONLY those chunks
    Critic      -> checks the draft against the retrieved chunks. If any
                   claim in the draft isn't supported, it's rejected with
                   a structured failure reason (not just "wrong") — this
                   failure-type log is what Day 6's Optimizer will read to
                   auto-improve prompts.

Run locally:
    pip install langgraph chromadb anthropic scikit-learn numpy
    export ANTHROPIC_API_KEY=sk-...
    python day3_multi_agent.py --query "..."
"""

import os
import json
import argparse
from typing import TypedDict, List, Tuple

from langgraph.graph import StateGraph, END

from day1_rag_agent import build_vector_store, retrieve


# ---------- Shared state passed between agents ----------
class AgentState(TypedDict):
    query: str
    retrieved: List[Tuple[str, str]]   # [(chunk_id, text), ...]
    draft_answer: str
    verdict: str                       # "approved" | "rejected"
    failure_type: str                  # "" | "hallucination" | "retrieval-miss" | "incomplete-answer"
    final_answer: str


def call_llm(prompt: str) -> str:
    """Single shared LLM call used by all three agents."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return "[NO API KEY] " + prompt[:200] + "..."
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


# ---------- Agent nodes ----------
def researcher_node(state: AgentState) -> AgentState:
    collection = build_vector_store()
    retrieved = retrieve(collection, state["query"], k=2)
    return {**state, "retrieved": retrieved}


def analyst_node(state: AgentState) -> AgentState:
    context = "\n\n".join(f"[{cid}] {text}" for cid, text in state["retrieved"])
    prompt = (
        f"Using ONLY this context, answer the question. Be concise.\n\n"
        f"Context:\n{context}\n\nQuestion: {state['query']}\n\nAnswer:"
    )
    draft = call_llm(prompt)
    return {**state, "draft_answer": draft}


def critic_node(state: AgentState) -> AgentState:
    """
    Verifies the draft against retrieved sources. Rather than a plain
    pass/fail, it classifies WHY a rejection happened, into one of:
      - "retrieval-miss"    : sources don't contain info needed to answer
      - "hallucination"     : draft makes claims not in the sources
      - "incomplete-answer" : draft is grounded but doesn't fully answer
      - ""                  : approved, no failure
    This structured log is what the Day 6 Optimizer reads.
    """
    context = "\n\n".join(f"[{cid}] {text}" for cid, text in state["retrieved"])
    prompt = (
        "You are a strict fact-checker. Given the SOURCES and a DRAFT ANSWER, "
        "classify the draft into exactly one of these labels:\n"
        "- approved (fully supported by sources and answers the question)\n"
        "- retrieval-miss (sources don't contain the info needed)\n"
        "- hallucination (draft claims things not present in sources)\n"
        "- incomplete-answer (grounded, but doesn't fully answer the question)\n\n"
        f"Sources:\n{context}\n\n"
        f"Draft answer: {state['draft_answer']}\n\n"
        "Respond with ONLY the label, nothing else."
    )
    label = call_llm(prompt).strip().lower()

    if "approved" in label:
        return {**state, "verdict": "approved", "failure_type": ""}
    for tag in ("retrieval-miss", "hallucination", "incomplete-answer"):
        if tag in label:
            return {**state, "verdict": "rejected", "failure_type": tag}
    # LLM didn't return a clean label (or no API key) — log as such rather than guessing
    return {**state, "verdict": "unknown", "failure_type": "unclassified"}


def finalize_node(state: AgentState) -> AgentState:
    if state["verdict"] == "approved":
        final = state["draft_answer"]
    else:
        final = (
            f"[REJECTED by Critic — reason: {state['failure_type']}] "
            f"Draft was: {state['draft_answer']}"
        )
    return {**state, "final_answer": final}


# ---------- Build the graph ----------
def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("researcher", researcher_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("critic", critic_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("researcher")
    graph.add_edge("researcher", "analyst")
    graph.add_edge("analyst", "critic")
    graph.add_edge("critic", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


def log_run(state: AgentState, log_path: str = "run_log.jsonl"):
    """Append this run's outcome as a structured log line (feeds Day 5/6)."""
    entry = {
        "query": state["query"],
        "retrieved_ids": [cid for cid, _ in state["retrieved"]],
        "verdict": state["verdict"],
        "failure_type": state["failure_type"],
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    args = parser.parse_args()

    app = build_graph()
    result = app.invoke({
        "query": args.query,
        "retrieved": [],
        "draft_answer": "",
        "verdict": "",
        "failure_type": "",
        "final_answer": "",
    })

    print("=== Retrieved ===")
    for cid, text in result["retrieved"]:
        print(f"- ({cid}) {text[:80]}...")
    print(f"\n=== Critic verdict: {result['verdict']} (failure_type: '{result['failure_type']}') ===")
    print(f"\n=== Final answer ===\n{result['final_answer']}")

    log_run(result)
    print("\n(logged to run_log.jsonl)")
