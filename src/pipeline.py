"""
The Pipeline Orchestrator

Why we need it:
Every component so far (router, 5 query-transform techniques, 3
retrieval-improvement techniques, generation) works correctly on its
own, but nothing yet decides WHICH ones to actually run for a given
question, or in what order. This file is the "conductor": it asks the
router for a decision, then dynamically wires together only the
components that decision calls for - so a simple question doesn't pay
for retrieval it doesn't need, and a complex question gets exactly the
techniques the router judged relevant, not all eight every time.

Design decisions worth calling out:
1. "simple" route uses generate_simple_answer() (unrestricted, general
   knowledge) instead of generate_answer() (context-only) - see
   generator.py's docstring for why this distinction matters.
2. For "advanced_rag", exactly ONE query-transform technique actually
   produces the initial retrieval (rewriting, multi_query,
   decomposition, hyde, or self_query) - it doesn't make sense to run
   several of these simultaneously since they each replace the basic
   retrieval step in a different way. We pick whichever one the router
   suggested, in a fixed priority order chosen for specificity
   (self_query and decomposition change WHAT is searched most
   fundamentally, so they take priority over hyde/multi_query/rewriting
   which change HOW the same underlying question is searched).
3. reranking / compression / crag are NOT mutually exclusive - the
   router can suggest any combination of these three, and we apply
   whichever ones it suggested, in this fixed order: reranking (narrow
   a larger candidate pool down) -> compression (shrink surviving
   chunks) -> crag (final overall relevance gate, potentially discarding
   everything). This order matters: reranking needs the full pool before
   narrowing, compression should happen on the already-narrowed set to
   avoid wasted compression calls on chunks we're about to discard
   anyway, and CRAG's overall judgment should be the LAST gate.
4. Every LLM call made along the way (router, whichever query-transform
   technique ran, reranking, compression, crag, final generation) is
   individually recorded in call_breakdown, so the total pipeline cost
   for a question is the sum of ACTUAL calls made for THAT question -
   not a flat estimate.

If we skipped this orchestration layer:
We would have to manually decide, for every one of our ten test
questions, which functions to call and in what order - slow, inconsistent,
and exactly the kind of manual process an "agentic"/adaptive RAG system
is supposed to replace.
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
from src.routing.router import route_question, RouterResult
from src.retrieval.retriever import retrieve, RetrievedChunk
from src.query_transform.rewriter import rewrite_query
from src.query_transform.multi_query import multi_query_retrieve
from src.query_transform.decomposition import decompose_and_retrieve
from src.query_transform.hyde import hyde_retrieve
from src.query_transform.self_query import self_query_retrieve
from src.retrieval.reranker import rerank_chunks
from src.retrieval.compression import compress_chunks
from src.retrieval.crag import crag_evaluate
from src.generation.generator import generate_answer, generate_simple_answer, format_context

# Larger candidate pool size used when reranking will narrow it down afterward.
RERANK_POOL_SIZE = 8


@dataclass
class CallRecord:
    """One individual LLM call made anywhere in the pipeline for this question."""
    step: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_seconds: float


@dataclass
class PipelineResult:
    question: str
    route: str
    route_reason: str
    techniques_used: list[str]
    final_context: str
    answer: str
    sources: list[dict]
    call_breakdown: list[CallRecord] = field(default_factory=list)

    @property
    def total_cost_usd(self) -> float:
        return round(sum(c.cost_usd for c in self.call_breakdown), 8)

    @property
    def total_latency_seconds(self) -> float:
        return round(sum(c.latency_seconds for c in self.call_breakdown), 3)


def _record(breakdown: list[CallRecord], step: str, result) -> None:
    """Append one LLM call's cost/latency/tokens to the running breakdown list."""
    breakdown.append(CallRecord(
        step=step,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=result.cost_usd,
        latency_seconds=result.latency_seconds,
    ))


def _run_query_transform(question: str, techniques: list[str], breakdown: list[CallRecord]) -> list[RetrievedChunk]:
    """
    Run whichever ONE query-transform technique the router suggested
    (priority order: self_query > decomposition > hyde > multi_query >
    rewriting), or plain retrieval if none of these were suggested.
    """
    pool_size = RERANK_POOL_SIZE if "reranking" in techniques else config.TOP_K

    if "self_query" in techniques:
        result = self_query_retrieve(question, top_k=pool_size)
        _record(breakdown, "self_query", result)
        return result.retrieved_chunks

    if "decomposition" in techniques:
        per_sub = max(2, pool_size // 2)
        result = decompose_and_retrieve(question, top_k_per_subquestion=per_sub)
        _record(breakdown, "decomposition", result)
        return result.merged_chunks[:pool_size]

    if "hyde" in techniques:
        result = hyde_retrieve(question, top_k=pool_size)
        _record(breakdown, "hyde", result)
        return result.retrieved_chunks

    if "multi_query" in techniques:
        per_query = max(2, pool_size // 2)
        result = multi_query_retrieve(question, top_k_per_query=per_query)
        _record(breakdown, "multi_query", result)
        return result.merged_chunks[:pool_size]

    if "rewriting" in techniques:
        rw_result = rewrite_query(question)
        _record(breakdown, "rewriting", rw_result)
        return retrieve(rw_result.rewritten_query, top_k=pool_size)

    # No query-transform technique suggested: plain retrieval, no extra LLM call.
    return retrieve(question, top_k=pool_size)


def _run_retrieval_improvements(
    question: str, chunks: list[RetrievedChunk], techniques: list[str], breakdown: list[CallRecord]
) -> list[RetrievedChunk]:
    """Apply reranking, then compression, then CRAG - whichever were suggested."""
    if "reranking" in techniques:
        result = rerank_chunks(question, chunks, top_k=config.TOP_K)
        _record(breakdown, "reranking", result)
        chunks = result.ranked_chunks
    else:
        chunks = chunks[:config.TOP_K]

    if "compression" in techniques:
        result = compress_chunks(question, chunks)
        _record(breakdown, "compression", result)
        chunks = result.compressed_chunks

    if "crag" in techniques:
        result = crag_evaluate(question, chunks)
        _record(breakdown, "crag", result)
        chunks = result.usable_chunks

    return chunks


def run_pipeline(question: str) -> PipelineResult:
    """
    Full end-to-end pipeline for one question: route -> (retrieve with
    the right technique(s)) -> generate -> return everything needed for
    evaluation and cost reporting.
    """
    breakdown: list[CallRecord] = []

    router_result: RouterResult = route_question(question)
    _record(breakdown, "router", router_result)

    if router_result.route == "simple":
        gen_result = generate_simple_answer(question)
        _record(breakdown, "generation", gen_result)

        return PipelineResult(
            question=question,
            route=router_result.route,
            route_reason=router_result.reason,
            techniques_used=[],
            final_context="",
            answer=gen_result.answer,
            sources=[],
            call_breakdown=breakdown,
        )

    if router_result.route == "basic_rag":
        chunks = retrieve(question, top_k=config.TOP_K)
    else:  # advanced_rag
        pool = _run_query_transform(question, router_result.techniques, breakdown)
        chunks = _run_retrieval_improvements(question, pool, router_result.techniques, breakdown)

    gen_result = generate_answer(question, chunks)
    _record(breakdown, "generation", gen_result)

    return PipelineResult(
        question=question,
        route=router_result.route,
        route_reason=router_result.reason,
        techniques_used=router_result.techniques,
        final_context=format_context(chunks),
        answer=gen_result.answer,
        sources=gen_result.sources,
        call_breakdown=breakdown,
    )


if __name__ == "__main__":
    test_questions = [
        "What is Retrieval-Augmented Generation and why was it introduced?",
        "What are the steps for the annual inventory count of stationery warehouses?",
        "Compare the alarm system's card-access procedure with the fingerprint-access procedure, and explain which committee approvals differ between them.",
    ]

    for q in test_questions:
        result = run_pipeline(q)
        print(f"Question: {q}")
        print(f"  Route: {result.route} | Reason: {result.route_reason}")
        print(f"  Techniques used: {result.techniques_used}")
        print(f"  Answer: {result.answer[:200]}...")
        print(f"  Sources: {result.sources}")
        print("  Call breakdown:")
        for call in result.call_breakdown:
            print(f"    - {call.step}: ${call.cost_usd} | {call.latency_seconds}s")
        print(f"  TOTAL cost: ${result.total_cost_usd} | TOTAL latency: {result.total_latency_seconds}s")
        print()