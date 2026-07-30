<div align="center">

# Self-Improving Multimodal Research Agent System

**A multi-agent RAG system that checks its own work — and fixes itself when it's wrong.**

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-1c1c1c)
![Chroma](https://img.shields.io/badge/Vector%20Store-Chroma-6E56CF)
![Status](https://img.shields.io/badge/Status-Active-2F6F4E)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

[**Live Demo**](./webapp.html) · [Architecture](#architecture) · [Results](#results) · [Setup](#setup)

*Open `webapp.html` in any browser — no install, no build step, works today.*

</div>

---

## What this is

Most RAG demos generate an answer and stop there. This one goes a step further: a **Critic** agent independently checks every answer against the retrieved sources and labels *why* it fails when it does — and an **Optimizer** agent reads those failure logs and automatically fixes the pipeline, without a human manually rewriting prompts.

Built as a portfolio project targeting agentic-AI roles, with an emphasis on **measuring** reliability rather than just demoing it.

## Results

| Metric | Value |
|---|---|
| Knowledge base | 4 real topic documents — Zero Trust Architecture, Federated Learning, Blockchain Security, Edge AI (12 retrievable chunks) |
| Baseline retrieval accuracy | **88%** (15/17 in-scope questions, 18-question eval set) |
| Diagnosed failure | Too few chunks retrieved per query |
| **Post-optimization accuracy** | **100%** — fixed automatically, zero manual prompt edits |

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
  │              - retrieval-miss    (sources didn't contain what's needed)
  │              - hallucination     (draft claims things not in sources)
  │              - incomplete-answer (grounded, but doesn't fully answer)
  ▼
[logged to structured JSONL]
  ▼
Optimizer ────► reads failure-type logs, picks a targeted fix
                (e.g. retrieval-miss → increase retrieved chunks),
                re-runs the eval, reports before/after score
```

Orchestrated as a state graph (LangGraph). Vector storage via Chroma. Multimodal input via PDF text extraction and native vision (image passed directly to the LLM, no OCR step).

## Try it live

`webapp.html` is a standalone, single-file interactive demo — no install required.

- **Ask & Verify** — ask a question, watch it get retrieved, answered, and fact-checked in real time
- **Upload your own PDF** — extracted and folded into the retrievable knowledge base client-side
- **Run Eval / Optimize** — watch the 88% → 100% result happen live, animated, no setup
- Works fully key-free in **"show source passage only"** mode; add a free [Gemini API key](https://aistudio.google.com/apikey) (no credit card) to unlock AI-generated, Critic-verified answers

Just open the file in a browser — double-click, or drag it into an open Chrome window.

## What each stage added

| Stage | Component |
|---|---|
| 1 | Single RAG agent — retrieval + grounded generation |
| 2 | Multimodal input — PDF ingestion, image understanding |
| 3 | Multi-agent split — Researcher → Analyst → Critic, with structured failure-type logging |
| 4 | Live tool-calling — Researcher can search the web when local docs aren't enough |
| 5 | Eval harness — 18-question eval set, automated scoring, baseline measurement |
| 6 | Optimizer — reads failure logs, applies a targeted fix, re-measures |

## Setup (Python backend)

```bash
pip install chromadb anthropic scikit-learn numpy pypdf pillow langgraph
export ANTHROPIC_API_KEY=your-key
```

```bash
python day1_rag_agent.py
python day2_multimodal.py --pdf mydoc.pdf --query "..."
python day3_multi_agent.py --query "..."
python day4_tools.py --query "..."
python day5_eval.py
python day6_optimizer.py
```

## Design notes

- **Knowledge base:** 4 original explainer documents (`documents/*.txt`), chunked by paragraph at load time — swap in your own `.txt` files to point this at a different domain.
- **Embeddings:** local TF-IDF by default (no external model download or API call needed). Swap in `sentence-transformers` or an embeddings API for stronger semantic retrieval on larger, messier document sets.
- **Failure classification over pass/fail:** the Critic doesn't just approve/reject — it labels *why*, which is what makes the Optimizer's targeted fixes possible instead of blind retries.
- **A real bug, found and fixed during development:** the TF-IDF vectorizer was originally fit only on the base document set, before PDF content was ingested — meaning PDF-specific terms were never in the vocabulary and could never be retrieved, no matter how relevant. Fixed by fitting the vectorizer over the full combined corpus before building the vector store.
- **Two providers, by design:** the Python backend uses Anthropic's API; the browser demo uses Google's free-tier Gemini API to keep it accessible with zero cost to try. The pipeline itself is provider-agnostic.

## Next steps

- Swap TF-IDF for a real embedding model for stronger semantic matching
- Add hallucination/incomplete-answer prompt-rewrite fixes to the Optimizer (implemented, needs a live API key to validate)
- Expand the eval set and add human-labeled ground truth for the Critic's judgments
- Support additional document formats (Word, PowerPoint) via the same text-extraction pattern used for PDFs

---

<div align="center">
<sub>Built by Sanjana Jirankali</sub>
</div>
