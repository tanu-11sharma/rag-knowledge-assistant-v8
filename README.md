# RAG Knowledge Assistant (v8: verified / grounded citations)

A small retrieval-augmented generation (RAG) Q&A agent that answers questions
over a bundled set of company policy documents, cites the exact source
sentence for every answer, and — v8's twist — runs each citation through a
lightweight faithfulness/groundedness check before presenting it as fact.

## What it does

1. **Index** — On startup, `app/data/*.txt` (sample HR/IT policy documents)
   are split into sentence-level chunks and indexed with a from-scratch
   TF-IDF vectorizer (`app/retriever.py`).
2. **Retrieve** — A question is vectorized the same way and compared to
   every chunk with cosine similarity to find the most relevant sentences.
3. **Verify** — Each retrieved chunk is scored against the *question* for
   content-word overlap (`app/verifier.py`). Chunks that don't clearly
   address what was asked are flagged `low` confidence instead of being
   silently included as if they were solid evidence.
4. **Ground + cite** — Only citations that pass the support check are used
   to compose the answer by default; if nothing clears the bar, the
   response is explicitly hedged ("weak match, not a definitive answer")
   rather than stated as fact. Every answer sentence is tagged with a
   `[doc_id#chunk_index]` citation and a per-citation `grounding_confidence`.

## Why it's relevant

RAG is one of the most common patterns in production AI systems today, and
one of its most common failure modes is a retriever returning the "closest
available" chunk even when it doesn't actually answer the question — the
system then states it with full confidence anyway. This project adds a
small, dependency-free verification layer on top of retrieval to catch
exactly that: every citation gets an explicit `supported` flag and
`grounding_confidence`, and low-confidence answers are hedged instead of
asserted. This is orthogonal to *which* retrieval strategy is used
upstream (this repo's earlier versions explored BM25, MMR-diversified
retrieval, multi-hop decomposition, recency weighting, and hybrid
TF-IDF+exact-match retrieval) — v8 focuses on verifying the evidence once
it's retrieved, a guardrail pattern that composes with any of them.

**Note:** This is a demo over synthetic sample HR policy text, not a real
company's actual policies, and is not legal, HR, or compliance advice.

## Project structure

```
app/
  data/            sample policy documents (.txt)
  retriever.py     from-scratch TF-IDF + cosine similarity retriever
  verifier.py      content-overlap faithfulness/groundedness checker
  rag.py           retrieve -> verify -> grounded, cited answer synthesis
  main.py          FastAPI app (/health, /documents, /ask)
tests/             pytest suite covering retriever, verifier, RAG pipeline, and API
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --reload
```

## Test

```bash
pytest -v
```

## Example usage

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many days do I have to submit an expense report?", "top_k": 2}'
```

Example response:

```json
{
  "question": "How many days do I have to submit an expense report?",
  "answer": "Employees must submit expense reports within 14 days of incurring a cost. [expense_policy#0]",
  "grounding_confidence": "medium",
  "citations": [
    {
      "doc_id": "expense_policy",
      "title": "Travel and Expense Policy",
      "chunk_index": 0,
      "text": "Employees must submit expense reports within 14 days of incurring a cost.",
      "score": 0.5893,
      "supported": true,
      "grounding_confidence": "medium"
    }
  ]
}
```

## Docker

```bash
docker build -t rag-knowledge-assistant-v8 .
docker run -p 8000:8000 rag-knowledge-assistant-v8
```
