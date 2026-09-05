"""
Retrieval Improvement Technique: Reranking

Why we need it:
The vector database ranks chunks by mathematical distance between
embeddings - a fast, useful, but approximate signal for "relevance".
Two chunks can sit at similar distances from a query vector while one
is actually much more useful for answering the specific question than
the other, because distance captures general topical similarity, not
fine-grained relevance to exactly what's being asked. Reranking adds a
second, more precise judgment pass: we retrieve a LARGER pool of
candidates first (casting a wider net), then ask the LLM to directly
read and score each candidate's actual relevance to the question, and
keep only the best ones.

Why we ask the LLM to score ALL candidates in one prompt (not one call
per chunk):
Sending one request per candidate would multiply our cost and latency
by the pool size. Batching all candidates into a single prompt with a
numbered list lets the model compare them against each other and against
the question in one pass, which is both cheaper and lets the model make
relative judgments (this one is clearly better than that one).

If we skipped this step:
We would trust raw vector distance as the final word on relevance, even
in cases where it's a poor proxy for what actually helps answer the
specific question - potentially keeping a topically-related but
unhelpful chunk over a more genuinely useful one that happened to sit
slightly farther away in embedding space.
"""

import json
import re
import sys
from pathlib import Path
from dataclasses import dataclass

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src import llm_client
from src.retrieval.retriever import RetrievedChunk

RERANK_SYSTEM_INSTRUCTION = """You are a relevance-scoring assistant for a document retrieval system.
You will be given a question and a numbered list of candidate text
passages retrieved for it. Score how genuinely relevant and useful each
passage is for answering the question, on a scale from 0 (completely
irrelevant) to 10 (directly and fully answers the question).

Judge each passage on its actual content and usefulness for answering
the specific question - not on how long it is or how many keywords it
shares with the question.

Return ONLY a JSON array with one object per passage, in the SAME order
as the input list, of exactly this shape:
[{"index": <passage number>, "relevance_score": <0-10 integer>}, ...]"""


@dataclass
class RerankResult:
    original_query: str
    ranked_chunks: list[RetrievedChunk]
    scores: list[int]
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_seconds: float


def _extract_json_array(raw_text: str) -> list[dict]:
    """Strip any markdown code-fence wrapping before parsing the JSON array."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE)
    return json.loads(cleaned)


def _format_candidates(chunks: list[RetrievedChunk]) -> str:
    """Turn candidate chunks into a numbered list for the reranking prompt."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(f"[Passage {i}]\n{chunk.text}")
    return "\n\n".join(parts)


def rerank_chunks(question: str, candidate_chunks: list[RetrievedChunk], top_k: int = 5) -> RerankResult:
    """
    Re-score a pool of candidate chunks by true relevance to the
    question (using the LLM as a judge) and return only the top_k best,
    re-ordered by that judgment instead of raw vector distance.
    """
    prompt = f"""Question: {question}

Candidate passages:
{_format_candidates(candidate_chunks)}"""

    llm_result = llm_client.generate(
        prompt=prompt,
        system_instruction=RERANK_SYSTEM_INSTRUCTION,
        temperature=0.0,
    )

    try:
        scored = _extract_json_array(llm_result.text)
        score_by_index = {item["index"]: item["relevance_score"] for item in scored}
    except (json.JSONDecodeError, AttributeError, KeyError):
        # Safe fallback: if scoring fails, keep the original vector-distance order.
        score_by_index = {i + 1: len(candidate_chunks) - i for i in range(len(candidate_chunks))}

    indexed_chunks = list(enumerate(candidate_chunks, start=1))
    indexed_chunks.sort(key=lambda pair: score_by_index.get(pair[0], 0), reverse=True)

    top_chunks = [chunk for _, chunk in indexed_chunks[:top_k]]
    top_scores = [score_by_index.get(idx, 0) for idx, _ in indexed_chunks[:top_k]]

    return RerankResult(
        original_query=question,
        ranked_chunks=top_chunks,
        scores=top_scores,
        input_tokens=llm_result.input_tokens,
        output_tokens=llm_result.output_tokens,
        cost_usd=llm_result.cost_usd,
        latency_seconds=llm_result.latency_seconds,
    )


if __name__ == "__main__":
    from src.retrieval.retriever import retrieve

    test_question = "Who is responsible for approving the disposal of unused fixed assets?"
    print(f"Question: {test_question}\n")

    # Cast a wider net first (larger pool than we actually want to keep)
    candidates = retrieve(test_question, top_k=8)

    print("--- Before reranking (vector-distance order) ---")
    for i, c in enumerate(candidates, start=1):
        print(f"  {i}. (distance={c.distance:.4f}) {c.metadata['source_file']}, page {c.metadata['page_number']}")

    result = rerank_chunks(test_question, candidates, top_k=3)

    print("\n--- After reranking (LLM relevance-score order) ---")
    for i, (chunk, score) in enumerate(zip(result.ranked_chunks, result.scores), start=1):
        print(f"  {i}. (score={score}/10) {chunk.metadata['source_file']}, page {chunk.metadata['page_number']}")

    print(f"\nCost (reranking call only): ${result.cost_usd}")
    print(f"Latency (reranking call only): {result.latency_seconds}s")
    