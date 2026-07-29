"""
Day 2 — Multimodal Input (PDF + Image)
---------------------------------------
Extends Day 1's RAG agent so the Researcher can take a PDF or an image as
part of the query, not just plain text.

Two paths:
1. PDF  -> extract text with pypdf, chunk it, add it into the SAME vector
           store as regular docs, so it becomes retrievable context.
2. Image -> base64-encode it and pass it directly to a vision-capable LLM
           call alongside the text question (no OCR needed — the model
           reads the image natively).

Run locally:
    pip install pypdf pillow chromadb anthropic scikit-learn numpy
    export ANTHROPIC_API_KEY=sk-...
    python day2_multimodal.py --pdf mydoc.pdf --query "..."
    python day2_multimodal.py --image screenshot.png --query "..."
"""

import os
import base64
import argparse
from pypdf import PdfReader

from day1_rag_agent import DOCS, TfidfEmbeddingFunction, retrieve, generate_answer  # reuse Day 1 pieces
import chromadb


def build_vector_store_with_pdf(pdf_path: str = None, persist_path: str = "./chroma_store"):
    """
    Same as Day 1's build_vector_store, but if a PDF is provided, its chunks
    are folded into the corpus BEFORE the TF-IDF vectorizer is fit. This
    fixes the retrieval-miss bug: fitting vocabulary only on the base docs
    meant PDF-specific words were never embeddable, so PDF content could
    never be retrieved no matter how relevant it was.
    """
    base_texts = [d["text"] for d in DOCS]
    base_ids = [d["id"] for d in DOCS]

    pdf_chunks, pdf_ids = [], []
    if pdf_path:
        pdf_chunks = extract_pdf_text(pdf_path)
        pdf_ids = [f"pdf_{os.path.basename(pdf_path)}_{i}" for i in range(len(pdf_chunks))]

    all_texts = base_texts + pdf_chunks
    all_ids = base_ids + pdf_ids

    client = chromadb.PersistentClient(path=persist_path)
    embed_fn = TfidfEmbeddingFunction(corpus=all_texts)  # vocabulary now includes PDF content

    try:
        client.delete_collection("research_docs")
    except Exception:
        pass
    collection = client.create_collection(name="research_docs", embedding_function=embed_fn)
    collection.add(ids=all_ids, documents=all_texts)

    return collection, len(pdf_chunks)


# ---------- PDF ingestion ----------
def extract_pdf_text(pdf_path: str, chunk_size: int = 500) -> list:
    """Extract text from a PDF and split into rough chunks for retrieval."""
    reader = PdfReader(pdf_path)
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)

    # naive fixed-size chunking — good enough for a first pass
    words = full_text.split()
    chunks = [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]
    return chunks


def add_pdf_to_store(collection, pdf_path: str):
    """Kept for reference — superseded by build_vector_store_with_pdf, which fixes
    the vocabulary-freezing bug described above. Left here so the diff/history
    of the bug fix is visible."""
    chunks = extract_pdf_text(pdf_path)
    ids = [f"pdf_{os.path.basename(pdf_path)}_{i}" for i in range(len(chunks))]
    if chunks:
        collection.add(ids=ids, documents=chunks)
    return len(chunks)


# ---------- Image handling ----------
def encode_image(image_path: str) -> tuple:
    """Read an image file and return (base64_data, media_type)."""
    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    media_type = f"image/{'jpeg' if ext == 'jpg' else ext}"
    with open(image_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return data, media_type


def generate_answer_with_image(query: str, retrieved_chunks: list, image_path: str):
    """
    Call a vision-capable LLM with retrieved text context AND an image.
    The model reads the image directly — no separate OCR/captioning step.
    """
    context = "\n\n".join(f"[{cid}] {text}" for cid, text in retrieved_chunks)
    text_prompt = (
        f"Use the retrieved context AND the attached image to answer the question. "
        f"If neither contains the answer, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}"
    )

    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        img_data, media_type = encode_image(image_path)
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": media_type, "data": img_data
                    }},
                    {"type": "text", "text": text_prompt},
                ],
            }],
        )
        return resp.content[0].text

    return (
        "[NO API KEY FOUND] Set ANTHROPIC_API_KEY to run the vision call. "
        "Retrieval-only preview of what would be sent:\n\n" + text_prompt
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", help="Path to a PDF to ingest into the knowledge base")
    parser.add_argument("--image", help="Path to an image to include in the query")
    parser.add_argument("--query", required=True, help="Your question")
    args = parser.parse_args()

    collection, n_chunks = build_vector_store_with_pdf(pdf_path=args.pdf)
    if args.pdf:
        print(f"Ingested {n_chunks} chunks from {args.pdf}")

    retrieved = retrieve(collection, args.query, k=2)
    print("\n=== Retrieved chunks ===")
    for cid, text in retrieved:
        print(f"- ({cid}) {text[:100]}...")

    print("\n=== Answer ===")
    if args.image:
        print(generate_answer_with_image(args.query, retrieved, args.image))
    else:
        print(generate_answer(args.query, retrieved))
