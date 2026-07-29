# Self-Improving Multimodal Research Agent System

A multi-agent RAG system where a **Critic** agent verifies every answer against retrieved sources and classifies *why* answers fail — and an **Optimizer** agent reads those failure logs to automatically fix the pipeline, without a human manually rewriting prompts.

Built in 6 days as a portfolio project targeting agentic-AI roles.

## Result

Measured baseline: **93% retrieval accuracy** on an 18-question eval set.
Found 1 real retrieval-miss, diagnosed the root cause (too few chunks retrieved per query), and built an automated fix.
**Result after the Optimizer's fix: 100% retrieval accuracy** — improved automatically, no manual prompt editing.

## Architecture

```
Query
  │
  ▼
Researcher ──► retrieves relevant chunks from a vector DB (RAG)
  │              also accepts PDF/image input (multimodal)
  │              can call live web search when local docs are insufficient
  ▼
Analyst ──────► drafts an answer using ONLY the retrieved context
  ▼
Critic ───────► checks the draft against sources. Classifies failures into:
  │              - retrieval-miss   (sources didn't contain what's needed)
  │              - hallucination    (draft claims things not in sources)
  │              - incomplete-answer (grounded, but doesn't fully answer)
  ▼
[logged to structured JSONL]
  ▼
Optimizer ────► reads failure-type logs, picks a targeted fix
                (e.g. retrieval-miss → increase retrieved chunks),
                re-runs the eval, reports before/after score
```

Orchestrated as a state graph (LangGraph). Vector storage via Chroma. Multimodal input via PDF text extraction and native vision (image passed directly to the LLM, no OCR step).

## What each day added

| Day | Component |
|---|---|
| 1 | Single RAG agent — retrieval + grounded generation |
| 2 | Multimodal input — PDF ingestion, image understanding |
| 3 | Multi-agent split — Researcher → Analyst → Critic, with structured failure-type logging |
| 4 | Live tool-calling — Researcher can search the web when local docs aren't enough |
| 5 | Eval harness — 18-question eval set, automated scoring, baseline measurement |
| 6 | Optimizer — reads failure logs, applies a targeted fix, re-measures |

## Setup

```bash
pip install chromadb anthropic scikit-learn numpy pypdf pillow langgraph
export ANTHROPIC_API_KEY=your-key
```

## Usage

```bash
python day1_rag_agent.py
python day2_multimodal.py --pdf mydoc.pdf --query "..."
python day3_multi_agent.py --query "..."
python day4_tools.py --query "..."
python day5_eval.py
python day6_optimizer.py
```

## Design notes

- **Embeddings:** uses a local TF-IDF embedding function by default (no external model download or API call needed). Swap in `sentence-transformers` or an embeddings API for better semantic retrieval quality on larger/messier document sets.
- **Failure classification over pass/fail:** the Critic doesn't just approve/reject — it labels *why*, which is what makes the Optimizer's targeted fixes possible instead of blind retries.
- **A real bug, found and fixed during development:** the TF-IDF vectorizer was originally fit only on the base document set, before PDF content was ingested — meaning PDF-specific terms were never in the vocabulary and could never be retrieved, no matter how relevant. Fixed by fitting the vectorizer over the full combined corpus before building the vector store.

## Next steps

- Swap TF-IDF for a real embedding model for better semantic matching
- Add hallucination/incomplete-answer prompt-rewrite fixes to the Optimizer (implemented, needs a live API key to validate)
- Expand the eval set and add human-labeled ground truth for the Critic's judgments
