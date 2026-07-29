"""
Day 4 — Live Tool-Calling (Web Search)
-----------------------------------------
Gives the Researcher agent a second information source: live web search,
using Claude's built-in server-side web_search tool (no separate search
API/key needed — it's the same Anthropic API call, just with a tool
attached).

The Researcher now decides PER QUERY whether local docs are enough, or
whether it needs to search the web too — e.g. "what's today's date" or
"latest news on X" can't be answered from a static local doc set.

Run locally:
    pip install anthropic chromadb scikit-learn numpy langgraph
    export ANTHROPIC_API_KEY=sk-...
    python day4_tools.py --query "..."
"""

import os
import argparse

from day1_rag_agent import build_vector_store, retrieve


def researcher_with_web_search(query: str, local_chunks: list):
    """
    Calls Claude with the web_search tool attached. The model itself
    decides whether the local context is sufficient or whether it should
    issue a web search — we don't hardcode that decision.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return "[NO API KEY] Would call Claude with web_search tool attached here."

    import anthropic
    client = anthropic.Anthropic()

    local_context = "\n\n".join(f"[{cid}] {text}" for cid, text in local_chunks)

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=800,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{
            "role": "user",
            "content": (
                f"You have local reference material below, and a web_search tool. "
                f"Use the local material if it's sufficient. Use web_search only if "
                f"the question needs current/live information the local material "
                f"doesn't cover.\n\n"
                f"Local material:\n{local_context}\n\n"
                f"Question: {query}"
            ),
        }],
    )

    # Response may contain a mix of tool_use, tool_result, and text blocks —
    # collect the text parts for the final answer, and note if search was used.
    used_search = any(block.type == "server_tool_use" for block in resp.content)
    text_parts = [block.text for block in resp.content if block.type == "text"]

    return {
        "used_web_search": used_search,
        "answer": "\n".join(text_parts),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    args = parser.parse_args()

    collection = build_vector_store()
    local_chunks = retrieve(collection, args.query, k=2)

    print("=== Local chunks retrieved ===")
    for cid, text in local_chunks:
        print(f"- ({cid}) {text[:80]}...")

    result = researcher_with_web_search(args.query, local_chunks)
    print(f"\n=== Result ===")
    print(result)
