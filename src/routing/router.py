"""
The Router (Advanced Mode, Stage 1)

Why we need it:
Not every question needs the same amount of machinery. Some questions
(e.g. general RAG-theory questions with no answer in our banking-
procedures corpus) don't need document retrieval at all. Others need a
single straightforward retrieval. Others are genuinely hard (multi-part,
ambiguous, comparison-based, or date/metadata-dependent) and benefit
from one or more advanced techniques (rewriting, decomposition, HyDE,
self-query, reranking, compression, CRAG).

The router's job is to look at a question BEFORE any retrieval happens
and decide which of these three paths it should take, so the rest of
the pipeline only does as much work as the question actually needs.

Why this matters specifically for this project:
Several of our ten test questions (Q1, Q6, Q8 per the task) are about
RAG concepts in general and have no answer inside our Arabic banking-
procedures corpus. A good router should recognize these as "simple"
(answerable directly, no retrieval needed) rather than forcing them
through retrieval against an unrelated corpus - which would return weak,
irrelevant chunks and could mislead the final answer.

Why we ask for structured JSON output instead of free text:
The rest of the pipeline (in pipeline.py, built later) needs to
programmatically read "route" and "techniques" to decide what to run
next. Free-text explanations can't be reliably parsed by code; a fixed
JSON shape can.

If we skipped this component:
Every question would be forced through the same fixed pipeline
regardless of whether it actually needs retrieval or advanced
techniques - wasting cost and latency on simple questions, and
potentially degrading answer quality by feeding irrelevant retrieved
context into questions that didn't need it at all.
"""

import json
import re
import sys
from pathlib import Path
from dataclasses import dataclass

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src import llm_client

VALID_ROUTES = {"simple", "basic_rag", "advanced_rag"}
VALID_TECHNIQUES = {
    "rewriting", "multi_query", "decomposition", "hyde", "self_query",
    "reranking", "compression", "crag",
}

ROUTER_SYSTEM_INSTRUCTION = """You are a routing classifier for a Retrieval-Augmented Generation (RAG)
system. The system's document corpus contains Arabic-language internal
bank procedure manuals (asset/warehouse management, mail/records
handling, and central alarm system operations) - NOT general knowledge
about AI or RAG itself.

For each incoming question, classify it into exactly one route:

- "simple": The question can be answered directly from general
  knowledge and does NOT require looking anything up in the document
  corpus. This includes general conceptual/theoretical questions about
  RAG, AI, or unrelated general-knowledge topics that have nothing to
  do with bank procedures.

- "basic_rag": The question requires looking something up in the
  document corpus, and a single straightforward retrieval (using the
  question mostly as-is) is enough to find the answer. Use this for
  clear, single-part, well-specified factual lookups.

- "advanced_rag": The question requires looking something up in the
  document corpus, but plain retrieval alone is unlikely to work well.
  Use this when the question is ambiguous, multi-part, requires
  comparing/combining information from multiple places, depends on
  document metadata (like dates), or is phrased very differently from
  how the corpus would phrase it (benefiting from rewriting or
  hypothetical-answer techniques).

If the route is "advanced_rag", also suggest which techniques (one or
more) would help, chosen ONLY from this list:
["rewriting", "multi_query", "decomposition", "hyde", "self_query",
"reranking", "compression", "crag"]
- rewriting: question is poorly phrased or too colloquial for good retrieval
- multi_query: question would benefit from being searched multiple ways
- decomposition: question has multiple sub-questions bundled together
- hyde: question is abstract/conceptual, and imagining a hypothetical
  answer first would help retrieval
- self_query: question depends on a date or other document metadata filter
- reranking: many candidate chunks may be retrieved and need re-ordering
  by true relevance
- compression: retrieved chunks are likely to be long and need
  compressing to the relevant part before generation
- crag: initial retrieval is likely to be weak/irrelevant and needs a
  corrective fallback strategy

Respond with ONLY a JSON object, no markdown formatting, no extra text,
in exactly this shape:
{"route": "<simple|basic_rag|advanced_rag>", "reason": "<one concise sentence>", "techniques": [<list of technique strings, empty list if route is not advanced_rag>]}"""


@dataclass
class RouterResult:
    route: str
    reason: str
    techniques: list[str]
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_seconds: float


def _extract_json(raw_text: str) -> dict:
    """
    Models sometimes wrap JSON in markdown code fences (```json ... ```)
    even when told not to. This strips any such wrapping before parsing,
    so a small formatting quirk doesn't crash the whole pipeline.
    """
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE)
    return json.loads(cleaned)


def route_question(question: str) -> RouterResult:
    """
    Classify a question into simple / basic_rag / advanced_rag and,
    for advanced_rag, suggest which techniques are relevant.
    Falls back to a safe default (basic_rag, no techniques) if the
    model's output can't be parsed, so a formatting glitch never
    crashes the whole pipeline.
    """
    llm_result = llm_client.generate(
        prompt=f"Question to classify: {question}",
        system_instruction=ROUTER_SYSTEM_INSTRUCTION,
        temperature=0.0,  # routing should be as deterministic as possible
    )

    try:
        parsed = _extract_json(llm_result.text)
        route = parsed.get("route", "basic_rag")
        reason = parsed.get("reason", "")
        techniques = parsed.get("techniques", [])

        if route not in VALID_ROUTES:
            route = "basic_rag"
        techniques = [t for t in techniques if t in VALID_TECHNIQUES]

    except (json.JSONDecodeError, AttributeError):
        # Safe fallback: if the model didn't return valid JSON, default
        # to basic_rag so the question still gets answered, rather than
        # crashing the pipeline.
        route, reason, techniques = "basic_rag", "Fallback: router output could not be parsed.", []

    return RouterResult(
        route=route,
        reason=reason,
        techniques=techniques,
        input_tokens=llm_result.input_tokens,
        output_tokens=llm_result.output_tokens,
        cost_usd=llm_result.cost_usd,
        latency_seconds=llm_result.latency_seconds,
    )


if __name__ == "__main__":
    # Quick manual test with a few different kinds of questions
    test_questions = [
        "What is Retrieval-Augmented Generation and why was it introduced?",
        "What are the steps for the annual inventory count of stationery warehouses?",
        "Compare the alarm system's card-access procedure with the fingerprint-access procedure, and explain which committee approvals differ between them.",
    ]

    for q in test_questions:
        result = route_question(q)
        print(f"Question: {q}")
        print(f"  Route: {result.route}")
        print(f"  Reason: {result.reason}")
        print(f"  Techniques: {result.techniques}")
        print(f"  Cost: ${result.cost_usd} | Latency: {result.latency_seconds}s")
        print()