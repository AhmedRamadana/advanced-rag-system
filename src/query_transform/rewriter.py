"""
Query Transformation Technique: Rewriting

Why we need it:
A question retrieval sees is compared, as a vector, against our stored
chunks. If the question is phrased conversationally, with filler words,
vague pronouns, or unclear structure, its embedding drifts away from
the clean, formal wording used in the actual procedure manuals - even
though a human would easily understand what's being asked. Rewriting
fixes this by asking the LLM to restate the question in a clear,
keyword-focused way BEFORE we embed it and search with it.

Why an LLM does this (not a simple text-cleaning script):
Resolving vague references into explicit, searchable terms requires
understanding intent, not just removing stopwords.

If we skipped this step:
A vaguely or colloquially phrased question would be embedded and
searched exactly as typed, potentially missing the right chunk even
though the underlying information intent was actually answerable.
"""

import sys
from pathlib import Path
from dataclasses import dataclass

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src import llm_client

REWRITER_SYSTEM_INSTRUCTION = """You are a query rewriting assistant for a document retrieval system.
Rewrite the user's question into a single, clear, keyword-focused
search query optimized for semantic retrieval, while preserving its
original meaning and language.

Rules:
- Resolve vague pronouns or implicit references into explicit terms.
- Remove conversational filler ("so like", "can you tell me", etc.).
- Keep it as a single, well-formed question or query phrase.
- Do not answer the question - only rewrite it.
- Return ONLY the rewritten query text, with no extra explanation, labels, or quotation marks."""


@dataclass
class RewriteResult:
    original_query: str
    rewritten_query: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_seconds: float


def rewrite_query(question: str) -> RewriteResult:
    """Rewrite a question into a clearer, retrieval-friendly form."""
    llm_result = llm_client.generate(
        prompt=f"Original question: {question}",
        system_instruction=REWRITER_SYSTEM_INSTRUCTION,
        temperature=0.2,
    )

    return RewriteResult(
        original_query=question,
        rewritten_query=llm_result.text.strip().strip('"'),
        input_tokens=llm_result.input_tokens,
        output_tokens=llm_result.output_tokens,
        cost_usd=llm_result.cost_usd,
        latency_seconds=llm_result.latency_seconds,
    )


if __name__ == "__main__":
    test_questions = [
        "so like, what do you even do when that alarm thing goes off, all the steps and stuff?",
        "who has to sign off on it when they want to get rid of old stuff nobody uses anymore in the warehouse?",
    ]

    for q in test_questions:
        result = rewrite_query(q)
        print(f"Original:  {result.original_query}")
        print(f"Rewritten: {result.rewritten_query}")
        print(f"Cost:  | Latency: {result.latency_seconds}s")
        print()
