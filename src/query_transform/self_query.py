"""
Query Transformation Technique: Self-Query

Why we need it:
Plain semantic (embedding) search only understands MEANING similarity -
it has no concept of "before a certain date" or "only from this specific
document". If a question implies a constraint like "which procedures
were issued before 2025?", a normal vector search would ignore that
constraint entirely and just return whatever is semantically closest,
potentially mixing in content from documents that don't satisfy the
date condition at all.

How Self-Query solves this:
We ask the LLM to read the question and split it into two things:
1. semantic_query - the actual content being asked about, with any
   metadata constraint removed (this is what we embed and search with).
2. filter - a structured condition (field, operator, value) extracted
   from the metadata-related part of the question, if any. This is
   applied as a database-level filter BEFORE similarity search even
   runs, guaranteeing that only chunks satisfying the condition are
   considered at all.

Why this needs the numeric yearmonth fields we backfilled earlier:
ChromaDB's comparison operators ($lt, $gt, $gte, $lte) require numeric
metadata, which is exactly why we added issue_yearmonth and
last_review_yearmonth (see backfill_dates.py) instead of relying on the
original string dates.

If we skipped this step:
Date- or document-scoped questions would be answered using only
semantic similarity, silently ignoring the explicit constraint in the
question and potentially retrieving accurate-sounding but wrongly-
sourced content (e.g. citing a 2026 procedure when the question
specifically asked about pre-2025 ones).
"""

import json
import re
import sys
from pathlib import Path
from dataclasses import dataclass

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src import llm_client
from src.retrieval.retriever import get_collection, embed_query, RetrievedChunk
from src.ingestion.chunker import DOCUMENT_METADATA

AVAILABLE_SOURCE_FILES = list(DOCUMENT_METADATA.keys())

SELF_QUERY_SYSTEM_INSTRUCTION = f"""You are a self-query assistant for a document retrieval system. Each
stored chunk has this metadata available for filtering:

- source_file: the exact filename of the document it came from. Must be
  exactly one of: {AVAILABLE_SOURCE_FILES}
- issue_yearmonth: an integer in YYYYMM format representing when the
  document was issued (e.g. 202408 means August 2024).
- last_review_yearmonth: an integer in YYYYMM format representing when
  the document was last reviewed.

Given a question, extract:
1. "semantic_query": the core content question, with any metadata
   constraint (dates, specific document references) REMOVED, so it can
   be used for pure semantic search.
2. "filter": a structured filter object if the question implies one, of
   the shape {{"field": "<source_file|issue_yearmonth|last_review_yearmonth>",
   "operator": "<$eq|$ne|$lt|$lte|$gt|$gte>", "value": <matching value>}},
   or null if the question implies no metadata constraint at all.

Examples of constraints and how to convert them:
- "issued before 2025" -> {{"field": "issue_yearmonth", "operator": "$lt", "value": 202501}}
- "reviewed in or after 2026" -> {{"field": "last_review_yearmonth", "operator": "$gte", "value": 202601}}
- "issued in August 2024" -> {{"field": "issue_yearmonth", "operator": "$eq", "value": 202408}}

If the question has no date or document-scope constraint, "filter" must be null.

Return ONLY a JSON object, no markdown formatting, no extra text, in
exactly this shape:
{{"semantic_query": "<string>", "filter": <object or null>}}"""


@dataclass
class SelfQueryResult:
    original_query: str
    semantic_query: str
    extracted_filter: dict | None
    retrieved_chunks: list[RetrievedChunk]
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_seconds: float


def _extract_json(raw_text: str) -> dict:
    """Strip any markdown code-fence wrapping before parsing the JSON object."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE)
    return json.loads(cleaned)


def extract_self_query(question: str) -> tuple[str, dict | None, "llm_client.LLMResponse"]:
    """Split a question into its semantic part and an optional metadata filter."""
    llm_result = llm_client.generate(
        prompt=f"Question: {question}",
        system_instruction=SELF_QUERY_SYSTEM_INSTRUCTION,
        temperature=0.0,
    )

    try:
        parsed = _extract_json(llm_result.text)
        semantic_query = parsed.get("semantic_query", question)
        extracted_filter = parsed.get("filter")
    except (json.JSONDecodeError, AttributeError):
        # Safe fallback: no filter, search with the original question as-is.
        semantic_query, extracted_filter = question, None

    return semantic_query, extracted_filter, llm_result


def self_query_retrieve(question: str, top_k: int = 5) -> SelfQueryResult:
    """
    Extract a semantic query + optional metadata filter from the
    question, then run a filtered similarity search: only chunks whose
    metadata satisfies the filter are even considered, before ranking
    by semantic similarity.
    """
    semantic_query, extracted_filter, llm_result = extract_self_query(question)

    collection = get_collection()
    query_vector = embed_query(semantic_query)

    where_clause = None
    if extracted_filter:
        where_clause = {extracted_filter["field"]: {extracted_filter["operator"]: extracted_filter["value"]}}

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        where=where_clause,
    )

    retrieved_chunks = [
        RetrievedChunk(chunk_id=chunk_id, text=text, metadata=metadata, distance=distance)
        for chunk_id, text, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]

    return SelfQueryResult(
        original_query=question,
        semantic_query=semantic_query,
        extracted_filter=extracted_filter,
        retrieved_chunks=retrieved_chunks,
        input_tokens=llm_result.input_tokens,
        output_tokens=llm_result.output_tokens,
        cost_usd=llm_result.cost_usd,
        latency_seconds=llm_result.latency_seconds,
    )


if __name__ == "__main__":
    test_question = "What alarm-system related procedures were issued before 2025?"
    print(f"Original question: {test_question}\n")

    result = self_query_retrieve(test_question, top_k=5)

    print(f"Semantic query extracted: {result.semantic_query}")
    print(f"Filter extracted: {result.extracted_filter}\n")

    print(f"Retrieved chunks: {len(result.retrieved_chunks)}")
    for i, chunk in enumerate(result.retrieved_chunks, start=1):
        print(f"  {i}. (distance={chunk.distance:.4f}) {chunk.metadata['source_file']}, "
              f"page {chunk.metadata['page_number']}, issued {chunk.metadata['issue_date']}")

    print(f"\nCost (self-query extraction call only): ${result.cost_usd}")
    print(f"Latency (self-query extraction call only): {result.latency_seconds}s")