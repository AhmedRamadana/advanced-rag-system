"""
Query Transformation Technique: Multi-Query

Why we need it:
A single phrasing of a question, even a well-rewritten one, only looks
at the corpus from one angle. If the actual answer in the manual uses
slightly different terminology or structure than the question implies,
that one search might miss it. Multi-Query addresses this by asking the
LLM to generate several DIFFERENT reformulations of the same underlying
question - each emphasizing different keywords or framing - then
searching with all of them and merging the results.

Why this improves recall:
Each reformulation is like casting the search net from a different
angle. A chunk that one phrasing's embedding sits far from might be much
closer to another phrasing's embedding. Combining results across several
searches increases the chance of finding the truly relevant chunk,
compared to relying on any single phrasing.

Why we deduplicate by chunk_id:
The same chunk is very likely to be retrieved by more than one of the
query variations (that's actually a good sign - it means multiple
angles agree it's relevant). Without deduplication, that chunk would be
sent to the LLM multiple times, wasting tokens/cost without adding new
information.

If we skipped this step:
We would rely on a single search angle only, potentially missing
relevant chunks that a differently-phrased version of the same question
would have found - directly hurting Context Relevance and, downstream,
answer completeness.
"""

import json
import re
import sys
from pathlib import Path
from dataclasses import dataclass

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src import llm_client
from src.retrieval.retriever import retrieve, RetrievedChunk

MULTI_QUERY_SYSTEM_INSTRUCTION = """You are a search-query generation assistant for a document retrieval
system. Given a user's question, generate 3 DIFFERENT search queries
that each approach the same underlying information need from a
different angle (different keywords, different phrasing, different
level of specificity), so that searching with all of them together
covers more ground than a single query would.

Rules:
- All 3 queries must aim to find the same underlying answer, just via
  different wording/angles.
- Keep the same language as the original question.
- Return ONLY a JSON array of exactly 3 strings, no markdown formatting,
  no extra text. Example shape: ["query one", "query two", "query three"]"""


@dataclass
class MultiQueryResult:
    original_query: str
    query_variations: list[str]
    merged_chunks: list[RetrievedChunk]
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_seconds: float


def _extract_json_array(raw_text: str) -> list[str]:
    """Strip any markdown code-fence wrapping before parsing the JSON array."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE)
    return json.loads(cleaned)


def generate_query_variations(question: str) -> tuple[list[str], "llm_client.LLMResponse"]:
    """Ask the LLM for 3 differently-phrased versions of the question."""
    llm_result = llm_client.generate(
        prompt=f"Original question: {question}",
        system_instruction=MULTI_QUERY_SYSTEM_INSTRUCTION,
        temperature=0.5,  # some variety is desirable here, unlike routing/rewriting
    )

    try:
        variations = _extract_json_array(llm_result.text)
        if not isinstance(variations, list) or not variations:
            raise ValueError("Empty or invalid variations list")
    except (json.JSONDecodeError, ValueError):
        # Safe fallback: if parsing fails, just use the original question alone.
        variations = [question]

    return variations, llm_result


def multi_query_retrieve(question: str, top_k_per_query: int = 3) -> MultiQueryResult:
    """
    Generate multiple phrasings of the question, retrieve chunks for
    each, and merge the results into one deduplicated list (keeping the
    best/lowest distance seen for any chunk retrieved more than once).
    """
    variations, llm_result = generate_query_variations(question)
    all_queries = [question] + variations

    best_by_id: dict[str, RetrievedChunk] = {}
    for query in all_queries:
        results = retrieve(query, top_k=top_k_per_query)
        for chunk in results:
            existing = best_by_id.get(chunk.chunk_id)
            if existing is None or chunk.distance < existing.distance:
                best_by_id[chunk.chunk_id] = chunk

    merged_chunks = sorted(best_by_id.values(), key=lambda c: c.distance)

    return MultiQueryResult(
        original_query=question,
        query_variations=variations,
        merged_chunks=merged_chunks,
        input_tokens=llm_result.input_tokens,
        output_tokens=llm_result.output_tokens,
        cost_usd=llm_result.cost_usd,
        latency_seconds=llm_result.latency_seconds,
    )


if __name__ == "__main__":
    test_question = "What is the process for handling lost or damaged registered mail?"
    print(f"Original question: {test_question}\n")

    result = multi_query_retrieve(test_question, top_k_per_query=3)

    print("Generated query variations:")
    for v in result.query_variations:
        print(f"  - {v}")

    print(f"\nMerged unique chunks found: {len(result.merged_chunks)}")
    for i, chunk in enumerate(result.merged_chunks, start=1):
        print(f"  {i}. (distance={chunk.distance:.4f}) {chunk.metadata['source_file']}, "
              f"page {chunk.metadata['page_number']}")

    print(f"\nCost (query generation only): ${result.cost_usd}")
    print(f"Latency (query generation only): {result.latency_seconds}s")