"""
Query Transformation Technique: Decomposition

Why we need it:
Some questions genuinely bundle multiple distinct information needs
into one sentence - for example, a comparison question ("compare X's
approval process with Y's approval process") actually requires looking
up TWO separate things in the corpus, not one. If we search with the
combined question as-is, its embedding becomes a blurry average of both
topics, and retrieval is likely to strongly favor whichever topic is
more prominent in the wording - silently dropping the other one.

How this differs from Multi-Query:
Multi-Query generates several DIFFERENT PHRASINGS of the SAME
underlying question (same information need, different wording).
Decomposition generates DIFFERENT SUB-QUESTIONS, each covering a
genuinely different part of a compound question. They solve different
problems: Multi-Query improves recall for one topic; Decomposition
ensures ALL topics in a multi-part question actually get searched for.

Why we retrieve separately per sub-question (not just concatenate them):
Searching with each sub-question independently lets each one find its
own most relevant chunks, uncontaminated by the other sub-question's
different vocabulary. We then merge the results so the final generation
step has evidence for every part of the original compound question.

If we skipped this step:
A multi-part or comparison question would be searched as a single blurry
query, likely retrieving strong evidence for only one side of the
comparison and leaving the model to answer the other side from general
knowledge (or not at all) - directly causing incomplete or unfaithful
answers.
"""

import json
import re
import sys
from pathlib import Path
from dataclasses import dataclass

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src import llm_client
from src.retrieval.retriever import retrieve, RetrievedChunk

DECOMPOSITION_SYSTEM_INSTRUCTION = """You are a question decomposition assistant for a document retrieval
system. Given a complex or multi-part question, break it down into 2 to
4 simpler, self-contained sub-questions such that:
- Each sub-question can be answered independently by searching a
  document corpus.
- Together, answering all the sub-questions provides enough information
  to fully answer the original question.
- Each sub-question should be understandable on its own, without
  needing the original question for context (spell out any entities
  that were only referenced implicitly in the original).

If the question is already simple and single-part, just return it
unchanged as the only item in the list.

Keep the same language as the original question.

Return ONLY a JSON array of strings, no markdown formatting, no extra
text. Example shape: ["sub-question one", "sub-question two"]"""


@dataclass
class DecompositionResult:
    original_query: str
    sub_questions: list[str]
    merged_chunks: list[RetrievedChunk]
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_seconds: float


def _extract_json_array(raw_text: str) -> list[str]:
    """Strip any markdown code-fence wrapping before parsing the JSON array."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE)
    return json.loads(cleaned)


def decompose_question(question: str) -> tuple[list[str], "llm_client.LLMResponse"]:
    """Ask the LLM to break the question into independent sub-questions."""
    llm_result = llm_client.generate(
        prompt=f"Original question: {question}",
        system_instruction=DECOMPOSITION_SYSTEM_INSTRUCTION,
        temperature=0.2,
    )

    try:
        sub_questions = _extract_json_array(llm_result.text)
        if not isinstance(sub_questions, list) or not sub_questions:
            raise ValueError("Empty or invalid sub-questions list")
    except (json.JSONDecodeError, ValueError):
        # Safe fallback: treat the original question as its own single sub-question.
        sub_questions = [question]

    return sub_questions, llm_result


def decompose_and_retrieve(question: str, top_k_per_subquestion: int = 3) -> DecompositionResult:
    """
    Break the question into sub-questions, retrieve chunks for each
    sub-question independently, and merge the results (deduplicated,
    keeping each chunk's best/lowest distance across sub-questions).
    """
    sub_questions, llm_result = decompose_question(question)

    best_by_id: dict[str, RetrievedChunk] = {}
    for sub_q in sub_questions:
        results = retrieve(sub_q, top_k=top_k_per_subquestion)
        for chunk in results:
            existing = best_by_id.get(chunk.chunk_id)
            if existing is None or chunk.distance < existing.distance:
                best_by_id[chunk.chunk_id] = chunk

    merged_chunks = sorted(best_by_id.values(), key=lambda c: c.distance)

    return DecompositionResult(
        original_query=question,
        sub_questions=sub_questions,
        merged_chunks=merged_chunks,
        input_tokens=llm_result.input_tokens,
        output_tokens=llm_result.output_tokens,
        cost_usd=llm_result.cost_usd,
        latency_seconds=llm_result.latency_seconds,
    )


if __name__ == "__main__":
    test_question = (
        "Compare the alarm system's card-access procedure with the "
        "fingerprint-access procedure, and explain which committee "
        "approvals differ between them."
    )
    print(f"Original question: {test_question}\n")

    result = decompose_and_retrieve(test_question, top_k_per_subquestion=3)

    print("Sub-questions generated:")
    for sq in result.sub_questions:
        print(f"  - {sq}")

    print(f"\nMerged unique chunks found: {len(result.merged_chunks)}")
    for i, chunk in enumerate(result.merged_chunks, start=1):
        print(f"  {i}. (distance={chunk.distance:.4f}) {chunk.metadata['source_file']}, "
              f"page {chunk.metadata['page_number']}")

    print(f"\nCost (decomposition call only): ${result.cost_usd}")
    print(f"Latency (decomposition call only): {result.latency_seconds}s")