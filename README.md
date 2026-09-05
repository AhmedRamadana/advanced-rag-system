# Advanced RAG System — Housing Bank Internal Procedures

An Advanced RAG system (Router + Basic RAG + Advanced RAG techniques) built on top of 3 internal procedure manuals from Housing Bank:

1. Central Mail and Files Unit Procedures Manual (29 pages)
2. Central Alarm Unit Procedures Manual (20 pages)
3. Assets and Warehouse Operations Procedures Manual (29 pages)

All source documents are in Arabic; the system is designed to answer questions in either Arabic or English.

---

## Goal

Build a RAG system that doesn't treat every question the same way. A **Router** classifies each incoming question into one of three paths (`simple` / `basic_rag` / `advanced_rag`), and Advanced RAG techniques (Query Understanding + Retrieval Improvement) are only applied when a question actually needs them — with per-question cost tracking and evaluation.

---

## Project Structure

```
advanced-rag-system/
├── data/
│   ├── raw/                          → the 3 source PDFs
│   └── processed/                    → preview files for manual inspection
├── results/                          → evaluation outputs and reports
├── src/
│   ├── ingestion/
│   │   ├── loader.py                 → Document Loading (pdfplumber + Arabic RTL text repair)
│   │   ├── chunker.py                → Text Chunking (recursive, structure-aware splitting)
│   │   ├── vectorstore_builder.py    → Embedding (gemini-embedding-001) + ChromaDB storage
│   │   └── backfill_dates.py         → Utility for populating/fixing document-level date metadata
│   ├── retrieval/
│   │   ├── retriever.py              → Retrieval Strategy (semantic similarity search)
│   │   ├── reranker.py               → Retrieval Improvement: reordering retrieved chunks by relevance
│   │   ├── compression.py            → Retrieval Improvement: contextual compression of retrieved chunks
│   │   └── crag.py                   → Retrieval Improvement: Corrective RAG (self-correction on weak retrieval)
│   ├── query_transform/
│   │   ├── rewriter.py               → Query Understanding: query rewriting
│   │   ├── multi_query.py            → Query Understanding: multi-query generation
│   │   ├── decomposition.py          → Query Understanding: sub-question decomposition
│   │   ├── hyde.py                   → Query Understanding: Hypothetical Document Embeddings
│   │   └── self_query.py             → Query Understanding: metadata-filtered querying
│   ├── generation/
│   │   └── generator.py              → LLM Integration + Response Generation (anti-hallucination prompt)
│   ├── routing/
│   │   └── router.py                 → Classifies each question before any retrieval happens
│   ├── evaluation/
│   │   ├── judge.py                  → LLM-as-judge scoring (Context Relevance, Faithfulness, Answer Relevance, Correctness)
│   │   └── metrics.py                → Metric definitions/aggregation for the evaluation results
│   ├── llm_client.py                 → Shared wrapper for every Gemini call (tracks tokens/cost/latency)
│   └── test_connection.py            → Quick sanity check for the API key and model names
├── vectorstore/                      → ChromaDB persistent collection (built automatically, not edited by hand)
├── config.py                         → Central settings (models, paths, pricing, chunk sizes)
├── requirements.txt
└── .env                               → GEMINI_API_KEY (not committed to any repository)
```

---

## Known Limitations

- **Occasional dropped "م" (meem) character inside the Arabic word "من" ("from")** in text extracted from the source PDFs. This was verified using two different extraction libraries (`pdfplumber` and `PyMuPDF`), and both show the same issue — indicating the problem originates in the PDF's own text layer (likely a font/ligature encoding issue in the source files), not in the choice of extraction library. The impact on semantic retrieval quality is expected to be limited, since the affected word is a common preposition.
- The ChromaDB collection currently uses the default distance metric (L2/Euclidean) rather than an explicitly configured cosine metric.

---

## Environment Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Create a `.env` file in the project root containing:
```
GEMINI_API_KEY=your_key_here
```

### Running the pipeline

```bash
# Build the vector store (safe to re-run; resumable if interrupted)
python -m src.ingestion.vectorstore_builder

# Test retrieval only
python -m src.retrieval.retriever

# Test the router
python -m src.routing.router

# Test the full pipeline (retrieval + generation)
python -m src.generation.generator
```

---

## Key Engineering Decisions

- **pdfplumber over PyPDF2**: the source manuals are dense with actor/action tables, and pdfplumber preserves visual layout (line breaks, spacing) more faithfully, which matters when each table row is one logical unit of meaning.
- **ChromaDB over FAISS**: an embedded (no separate server) vector database well suited to this project's scale (a few hundred chunks), with automatic on-disk persistence.
- **gemini-embedding-001**: Google's current stable embedding model, with explicit support for Arabic among 100+ languages (the older text-embedding-004 has been retired).
- **Two-tier model strategy**: `LIGHT_MODEL` (cheap, fast) is used for routing, query-transformation techniques, and the evaluation judge; `GENERATION_MODEL` (stronger, more expensive) is reserved for final answer generation only — keeping cost proportional to task complexity.
- **Centralized `llm_client.py`**: every LLM call in the project (router, rewriter, generator, judge) goes through this single wrapper, so token/cost/latency accounting stays accurate and no call is ever silently left out of the cost total.
- **Router-first design**: the router runs before any retrieval, deciding whether a question needs the corpus at all, a single retrieval pass, or one/more Advanced RAG techniques — avoiding forcing every question through every technique.
