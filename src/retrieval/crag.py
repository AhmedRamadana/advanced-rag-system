"""
Retrieval Improvement Technique: CRAG (Corrective RAG)

Why we need it:
Every retrieval improvement technique so far (rewriting, multi-query,
decomposition, HyDE, self-query, reranking, compression) tries to
IMPROVE what gets retrieved. CRAG handles the case where, despite all
that effort, the retrieved content is still genuinely irrelevant to the
question - which happens whenever a question has no real answer in our
corpus at all (this is a deliberate scenario in several of our test
questions). In that situation, no amount of reranking or compression
can fix bad content; the correct move is to recognize it's bad and
discard it, rather than silently handing the LLM irrelevant "evidence"
and hoping it notices on its own.

How CRAG works here:
After retrieval, we ask the LLM to grade the OVERALL usefulness of the
retrieved set against the question, as one of three grades:
- "correct": the retrieved context is sufficient and relevant.
- "ambiguous": partially relevant/incomplete - proceed, but the final
  answer should be treated as possibly incomplete.
- "incorrect": the retrieved context does not meaningfully help answer
  the question at all.
If the grade is "incorrect", we discard the retrieved chunks entirely
(return an empty context) instead of passing them to generation. This
forces our generation prompt's explicit rule ("if the context doesn't
answer the question, say so") to trigger cleanly, instead of the model
having to notice on its own that irrelevant context was smuggled in.

Why this matters specifically for our test questions:
Several of our ten test questions are expected to trigger weak/
irrelevant retrieval (Q10 in the task is explicitly designed around
this). CRAG is the mechanism that turns "we retrieved something, but it
was useless" into "we correctly recognized we have no good evidence" -
which is the more honest and defensible outcome for those cases.

If we skipped this step:
Retrieved chunks - however irrelevant - would always be forwarded to
generation. The final answer would then depend entirely on the
generation prompt's instructions to notice the mismatch on its own,
with no explicit safeguard confirming that actually happened.
"""

import json
import re
import sys
from pathlib import Path
from dataclasses import dataclass

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src import llm_client
from src.retrieval.retriever import RetrievedChunk

CRAG_SYSTEM_INSTRUCTION = """You are a retrieval-quality grading assistant for a document retrieval
system. Given a question and the set of passages that were retrieved
for it, grade the OVERALL usefulness of these passages for answering
the question, as exactly one of:

- "correct": the passages, together, contain enough relevant
  information to answer the question well.
- "ambiguous": the passages are somewhat related but incomplete or only
  partially useful for answering the question.
- "incorrect": the passages do not meaningfully help answer the
  question at all - they are about a different topic or too tangential
  to be useful evidence.

Return ONLY a JSON object, no markdown formatting, no extra text, in
exactly this shape:
{"grade": "<correct|ambiguous|incorrect>", "reason": "<one concise sentence>"}"""


@dataclass
class CRAGResult:
    original_query: str
    grade: str
    reason: str
    usable_chunks: list[RetrievedChunk]  # empty list if grade == "incorrect"
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_seconds: float


def _extract_json(raw_text: str) -> dict:
    """Strip any markdown code-fence wrapping before parsing the JSON object."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE)
    return json.loads(cleaned)


def _format_context(chunks: list[RetrievedChunk]) -> str:
    parts = [f"[Passage {i}]\n{c.text}" for i, c in enumerate(chunks, start=1)]
    return "\n\n".join(parts)


def crag_evaluate(question: str, chunks: list[RetrievedChunk]) -> CRAGResult:
    """
    Grade the overall relevance of retrieved chunks to the question.
    If graded "incorrect", the retrieved chunks are discarded
    (usable_chunks will be empty) so generation correctly falls back to
    "insufficient information" instead of being fed irrelevant context.
    """
    prompt = f"""Question: {question}

Retrieved passages:
{_format_context(chunks)}"""

    llm_result = llm_client.generate(
        prompt=prompt,
        system_instruction=CRAG_SYSTEM_INSTRUCTION,
        temperature=0.0,
    )

    try:
        parsed = _extract_json(llm_result.text)
        grade = parsed.get("grade", "ambiguous")
        reason = parsed.get("reason", "")
        if grade not in {"correct", "ambiguous", "incorrect"}:
            grade = "ambiguous"
    except (json.JSONDecodeError, AttributeError):
        # Safe fallback: if grading fails, assume "ambiguous" (keep chunks
        # but this signals downstream that quality wasn't confirmed).
        grade, reason = "ambiguous", "Fallback: CRAG grading output could not be parsed."

    usable_chunks = [] if grade == "incorrect" else chunks

    return CRAGResult(
        original_query=question,
        grade=grade,
        reason=reason,
        usable_chunks=usable_chunks,
        input_tokens=llm_result.input_tokens,
        output_tokens=llm_result.output_tokens,
        cost_usd=llm_result.cost_usd,
        latency_seconds=llm_result.latency_seconds,
    )


if __name__ == "__main__":
    from src.retrieval.retriever import retrieve

    test_cases = [
        # Expected to be genuinely well-matched to our corpus -> "correct"
        "What are the steps for the annual inventory count of stationery warehouses?",
        # Expected to have no real answer in our corpus -> "incorrect"
        "What is the bank's process for setting its overall foreign currency interest rate policy?",
    ]

    for question in test_cases:
        print(f"Question: {question}")
        chunks = retrieve(question, top_k=5)
        result = crag_evaluate(question, chunks)

        print(f"  Grade: {result.grade}")
        print(f"  Reason: {result.reason}")
        print(f"  Usable chunks kept: {len(result.usable_chunks)} (out of {len(chunks)} retrieved)")
        print(f"  Cost: ${result.cost_usd} | Latency: {result.latency_seconds}s")
        print()
        