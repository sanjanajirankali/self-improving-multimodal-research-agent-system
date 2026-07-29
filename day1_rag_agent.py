"""
Day 1 — Single Agent + RAG
--------------------------
Minimal, working RAG pipeline:
1. Load a small doc set into Chroma (local vector DB)
2. Retrieve relevant chunks for a query
3. Pass retrieved chunks + query to an LLM to generate a grounded answer

Run locally with your own API key:
    export ANTHROPIC_API_KEY=sk-...      (or OPENAI_API_KEY)
    python day1_rag_agent.py
"""

import os
import numpy as np
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfEmbeddingFunction(EmbeddingFunction):
    """
    Lightweight, fully local/offline embedding function using TF-IDF.
    Good enough for small demo doc sets and avoids needing to download a
    model or call an embeddings API. On your own machine with normal
    internet access, feel free to swap this for
    chromadb.utils.embedding_functions.DefaultEmbeddingFunction() (a real
    sentence-transformer model) or an OpenAI/Voyage embeddings API for
    much better semantic retrieval quality.
    """

    def __init__(self, corpus: list):
        self.vectorizer = TfidfVectorizer(max_features=512)
        self.vectorizer.fit(corpus)

    def __call__(self, input: Documents) -> Embeddings:
        matrix = self.vectorizer.transform(input)
        return [row.toarray().flatten().tolist() for row in matrix]

# ---------- 1. Sample knowledge base (replace with your own docs later) ----------
DOCS = [
    {
        "id": "doc1",
        "text": "Retrieval-Augmented Generation (RAG) combines a retriever, which fetches "
                "relevant text chunks from an external knowledge base, with a generator "
                "(an LLM) that uses those chunks as context to produce a grounded answer. "
                "This reduces hallucination compared to relying on the model's parametric "
                "memory alone.",
    },
    {
        "id": "doc2",
        "text": "Multi-agent systems split a task across specialized agents, e.g. a Researcher "
                "that retrieves information, an Analyst that synthesizes an answer, and a "
                "Critic that verifies the answer against sources. Orchestration frameworks "
                "like LangGraph model this as a graph of nodes and edges with shared state.",
    },
    {
        "id": "doc3",
        "text": "Automatic prompt optimization refers to systems that iteratively rewrite "
                "their own prompts based on evaluation feedback, rather than relying on a "
                "human to hand-tune them. DSPy is a well-known framework for this. The core "
                "loop is: run eval, log failure types, propose a prompt edit, re-run eval, "
                "keep the edit if the score improves.",
    },
    {
        "id": "doc4",
        "text": "A vector database stores text as high-dimensional embeddings and retrieves "
                "the most semantically similar chunks to a query embedding using approximate "
                "nearest-neighbor search. Chroma and FAISS are common lightweight choices for "
                "prototyping RAG systems.",
    },
]


def build_vector_store(persist_path: str = "./chroma_store"):
    """Create (or load) a Chroma collection using its built-in local embedding function."""
    client = chromadb.PersistentClient(path=persist_path)
    embed_fn = TfidfEmbeddingFunction(corpus=[d["text"] for d in DOCS])

    # Fresh collection each run since the TF-IDF vocabulary is fit per-run
    try:
        client.delete_collection("research_docs")
    except Exception:
        pass
    collection = client.create_collection(
        name="research_docs",
        embedding_function=embed_fn,
    )

    # Only add if empty (avoid duplicate inserts on repeated runs)
    if collection.count() == 0:
        collection.add(
            ids=[d["id"] for d in DOCS],
            documents=[d["text"] for d in DOCS],
        )
    return collection


def retrieve(collection, query: str, k: int = 2):
    """Return the top-k most relevant chunks for a query."""
    results = collection.query(query_texts=[query], n_results=k)
    chunks = results["documents"][0]
    ids = results["ids"][0]
    return list(zip(ids, chunks))


def generate_answer(query: str, retrieved_chunks: list):
    """
    Call an LLM with the retrieved context. Supports Anthropic or OpenAI —
    picks whichever API key is set in your environment.
    """
    context = "\n\n".join(f"[{cid}] {text}" for cid, text in retrieved_chunks)
    prompt = (
        f"Answer the question using ONLY the context below. If the context "
        f"doesn't contain the answer, say so — do not make things up.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )

    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

    elif os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
        )
        return resp.choices[0].message.content

    else:
        return (
            "[NO API KEY FOUND] Set ANTHROPIC_API_KEY or OPENAI_API_KEY to generate "
            "an answer. Retrieval-only mode — here's what would be sent to the LLM:\n\n"
            + prompt
        )


if __name__ == "__main__":
    collection = build_vector_store()

    test_query = "What does a Critic agent do in a multi-agent system?"
    retrieved = retrieve(collection, test_query, k=2)

    print("=== Retrieved chunks ===")
    for cid, text in retrieved:
        print(f"- ({cid}) {text[:100]}...")

    print("\n=== Answer ===")
    print(generate_answer(test_query, retrieved))
