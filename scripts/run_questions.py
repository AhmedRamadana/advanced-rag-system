"""
Final Orchestrator: runs the full conditional pipeline (Router -> Query
Understanding -> Retrieval Improvement -> Generation -> Evaluation) for
each of the project's official test questions, and produces the final
results table + per-question cost breakdown required by the task.

Why this file exists as a separate script (not folded into any single
component):
Every component built so far (router, 5 query-transform techniques, 3
retrieval-improvement techniques, generator, evaluator) was designed to
do ONE job well and be independently testable. This script is the
"conductor" that decides, PER QUESTION, which of those already-built
pieces actually need to run - mirroring the task's explicit requirement
that not every question should be forced through every technique.

How the conditional pipeline is decided:
1. Router classifies the question first, before any retrieval happens.
2. "simple" -> answered directly from general knowledge, no retrieval,
   no query/retrieval-improvement techniques at all.
3. "basic_rag" -> a single plain retrieval pass, no extra techniques.
4. "advanced_rag" -> we look at exactly which techniques the router
   named:
   - Any of {rewriting, multi_query, decomposition, hyde, self_query}
     that were named are ALL actually run (not just one arbitrarily
     picked), and their resulting chunks are merged (deduplicated by
     chunk_id, keeping the best/lowest distance seen for any chunk
     found by more than one technique). If the router named
     advanced_rag but no query-understanding technique, we fall back to
     a plain retrieval pass as the starting pool.
   - Any of {crag, reranking, compression} that were named are then
     applied, IN THIS FIXED ORDER: crag -> reranking -> compression.
     CRAG runs first because if it grades the pool as "incorrect", we
     discard everything immediately and skip reranking/compression
     entirely - there's no point spending extra LLM calls polishing
     content we're about to throw away anyway.

Why every question gets a full breakdown of ALL possible steps (not
just the ones that ran):
The task requires knowing, per question, which steps were executed and
which were not (e.g. "Router: executed, $0.0001; HyDE: not executed,
$0"). Recording every possible step name with an executed=True/False
flag - rather than only listing steps that happened to run - is what
makes the final cost table complete and auditable.

Design decision worth flagging explicitly: for "simple"-route questions
(no retrieval performed by design), Context Relevance and Faithfulness
are NOT computed, since both metrics require a real retrieved context
to judge against - computing them against an empty context would
produce a low score that misleadingly looks like a system failure,
rather than the correct, intentional "no retrieval needed" behavior.
Only Answer Relevance and Correctness are computed for these questions.

IMPORTANT - Corpus substitutions (per the task's own instruction:
"If the supplied corpus does not support one of these questions,
replace it with a corpus-supported question while preserving the
intended evaluation challenge"):

Most of the task's original ten questions (Q1, Q2, Q3, Q4, Q5, Q6, Q8,
Q9) are general-knowledge questions ABOUT RAG as a concept - they are
not about our corpus (three Arabic-language Housing Bank internal
procedure manuals) at all, and are expected to route to "simple" with
no retrieval. That is intentional and matches "Expected Focus" in the
task for each of them.

Two questions, however, explicitly require corpus-grounded retrieval
and our corpus contains nothing about "RAG" as a topic:
- Q7 (original: "Find documents about RAG published after 2024") was
  replaced with a corpus-supported question that preserves the exact
  same evaluation challenge (a metadata/date constraint requiring
  Self-Query): "What alarm-system related procedures were issued
  before 2025?" - our corpus does contain real issue-date metadata
  (the alarm manual: issued 2024-08; the other two: issued 2026-02),
  so this question exercises Self-Query exactly as intended.
- Q10 (original: "choose an appropriate question from the corpus"
  designed to trigger weak/irrelevant initial retrieval) was set to a
  question that is clearly WITHIN a covered domain (the alarm/access
  control system) so the router correctly sends it to retrieval, but
  whose specific detail (handling a breach during a public holiday) is
  not actually covered anywhere in the manual. This is deliberately
  different from an earlier draft of this substitution ("bank's policy
  for setting foreign currency interest rates"), which was found during
  testing to be mis-routed to "simple" by the router BEFORE any
  retrieval was attempted at all - since the router correctly
  recognized that question as entirely outside the corpus's domain,
  CRAG never got a chance to run. A question that is topically relevant
  but detail-unsupported is what actually reaches retrieval and lets
  CRAG grade the (expected to be weak) retrieved context as
  "incorrect", exercising the intended correction path.
"""

import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
from src import llm_client
from src.routing.router import route_question
from src.retrieval.retriever import retrieve, RetrievedChunk
from src.retrieval.reranker import rerank_chunks
from src.retrieval.compression import compress_chunks
from src.retrieval.crag import crag_evaluate
from src.query_transform.rewriter import rewrite_query
from src.query_transform.multi_query import multi_query_retrieve
from src.query_transform.decomposition import decompose_and_retrieve
from src.query_transform.hyde import hyde_retrieve
from src.query_transform.self_query import self_query_retrieve
from src.generation.generator import generate_answer, format_context
from src.evaluation.metrics import (
    evaluate_context_relevance, evaluate_faithfulness,
    evaluate_answer_relevance, evaluate_correctness,
)

# =============================================================================
# The task's official ten test questions.
# Q7 and Q10 are DELIBERATE, DOCUMENTED SUBSTITUTIONS - see the module
# docstring above for exactly why and what evaluation challenge each
# substitution preserves. Q3 is answered with the explicit preceding
# context spelled out, per the task's own instruction for handling its
# intentionally ambiguous wording.
# =============================================================================
QUESTIONS = {
    "Q1": "What is Retrieval-Augmented Generation (RAG)?",
    "Q2": "What are the main limitations of RAG?",
    "Q3": "Given the limitations of a standard RAG system just discussed (e.g. retrieval failures, stale or irrelevant context, added latency and cost) - why is it bad for a production system to be affected by these limitations?",
    "Q4": "What are the challenges and failure modes of RAG?",
    "Q5": "Compare RAG and fine-tuning, explain their advantages and disadvantages, and state when each should be used.",
    "Q6": "How does RAG reduce hallucination?",
    "Q7": "What alarm-system related procedures were issued before 2025?",  # SUBSTITUTED - see docstring
    "Q8": "What is reranking and why is it useful in RAG?",
    "Q9": "Which approach should be used for a simple question versus a complex multi-part question?",
    "Q10": "What is the procedure for handling a security breach detected by the fingerprint access system during a public holiday?",  # SUBSTITUTED - see docstring
}

# Documented so the substitution is traceable in the saved results file,
# not just in this script's comments.
QUESTION_NOTES = {
    "Q3": "Ambiguous per the task by design; preceding context (Q2's RAG limitations) made explicit in the wording sent to the system.",
    "Q7": "SUBSTITUTED from the original 'Find documents about RAG published after 2024' - our corpus has no RAG-related documents. Replaced with a corpus-supported date-metadata question that exercises the same Self-Query challenge.",
    "Q10": "SUBSTITUTED - a corpus-relevant question (alarm/access system domain) whose specific detail (public-holiday handling) is not covered in the manual, intended to trigger weak/irrelevant initial retrieval and exercise the CRAG correction path.",
}

QUERY_TRANSFORM_TECHNIQUES = {"rewriting", "multi_query", "decomposition", "hyde", "self_query"}
RETRIEVAL_IMPROVEMENT_TECHNIQUES = {"crag", "reranking", "compression"}

ALL_STEP_NAMES = [
    "router", "rewriting", "multi_query", "decomposition", "hyde", "self_query",
    "crag", "reranking", "compression", "generation",
    "judge_context_relevance", "judge_faithfulness", "judge_answer_relevance", "judge_correctness",
]

SIMPLE_ROUTE_SYSTEM_INSTRUCTION = """You are a helpful, knowledgeable assistant answering a general-knowledge
question. This question does NOT relate to the bank's internal
procedure documents, so answer it directly from your own general
knowledge, clearly and concisely, in the same language the question
was asked in."""


@dataclass
class StepRecord:
    """One line in the per-question cost/execution breakdown."""
    step: str
    executed: bool
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_seconds: float = 0.0


@dataclass
class QuestionResult:
    question_id: str
    question: str
    note: str
    route: str
    route_reason: str
    techniques_named_by_router: list[str]
    answer: str
    sources: list[dict]
    context_relevance_score: int | None
    faithfulness_score: int | None
    answer_relevance_score: int
    correctness_score: int
    steps: list[StepRecord]
    total_cost_usd: float
    total_latency_seconds: float


def _empty_step_records() -> dict[str, StepRecord]:
    """Start every question with every possible step marked as NOT executed."""
    return {name: StepRecord(step=name, executed=False) for name in ALL_STEP_NAMES}


def _merge_chunks(*chunk_lists: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Combine chunks from multiple techniques, deduplicated by chunk_id,
    keeping the lowest (best) distance seen for any chunk found more than once."""
    best_by_id: dict[str, RetrievedChunk] = {}
    for chunks in chunk_lists:
        for chunk in chunks:
            existing = best_by_id.get(chunk.chunk_id)
            if existing is None or chunk.distance < existing.distance:
                best_by_id[chunk.chunk_id] = chunk
    return sorted(best_by_id.values(), key=lambda c: c.distance)


def _generate_simple_answer(question: str):
    """Answer a 'simple' route question directly, without any retrieval -
    this deliberately does NOT use generator.generate_answer(), because
    that function's system prompt forbids using general knowledge, which
    is exactly what a 'simple' question needs."""
    return llm_client.generate(
        prompt=question,
        system_instruction=SIMPLE_ROUTE_SYSTEM_INSTRUCTION,
        temperature=0.3,
        model_name=config.GENERATION_MODEL,
    )


def run_query_understanding(question: str, techniques: list[str], steps: dict[str, StepRecord]) -> list[RetrievedChunk]:
    """Run every query-understanding technique the router named, and merge
    their resulting chunks. Falls back to plain retrieval if the router
    named advanced_rag but no query-understanding technique specifically."""
    named = [t for t in techniques if t in QUERY_TRANSFORM_TECHNIQUES]
    chunk_lists = []

    if "rewriting" in named:
        rw = rewrite_query(question)
        chunks = retrieve(rw.rewritten_query, top_k=config.TOP_K)
        chunk_lists.append(chunks)
        steps["rewriting"] = StepRecord("rewriting", True, rw.input_tokens, rw.output_tokens, rw.cost_usd, rw.latency_seconds)

    if "multi_query" in named:
        mq = multi_query_retrieve(question, top_k_per_query=3)
        chunk_lists.append(mq.merged_chunks)
        steps["multi_query"] = StepRecord("multi_query", True, mq.input_tokens, mq.output_tokens, mq.cost_usd, mq.latency_seconds)

    if "decomposition" in named:
        dc = decompose_and_retrieve(question, top_k_per_subquestion=3)
        chunk_lists.append(dc.merged_chunks)
        steps["decomposition"] = StepRecord("decomposition", True, dc.input_tokens, dc.output_tokens, dc.cost_usd, dc.latency_seconds)

    if "hyde" in named:
        hd = hyde_retrieve(question, top_k=config.TOP_K)
        chunk_lists.append(hd.retrieved_chunks)
        steps["hyde"] = StepRecord("hyde", True, hd.input_tokens, hd.output_tokens, hd.cost_usd, hd.latency_seconds)

    if "self_query" in named:
        sq = self_query_retrieve(question, top_k=config.TOP_K)
        chunk_lists.append(sq.retrieved_chunks)
        steps["self_query"] = StepRecord("self_query", True, sq.input_tokens, sq.output_tokens, sq.cost_usd, sq.latency_seconds)

    if not chunk_lists:
        # advanced_rag was chosen but no query-understanding technique was
        # named (only retrieval-improvement techniques, e.g. just
        # "reranking") -> start from a plain retrieval pass.
        return retrieve(question, top_k=config.TOP_K)

    return _merge_chunks(*chunk_lists)


def run_retrieval_improvement(question: str, chunks: list[RetrievedChunk], techniques: list[str], steps: dict[str, StepRecord]) -> list[RetrievedChunk]:
    """Apply crag -> reranking -> compression, in that fixed order, for
    whichever of these the router actually named. See module docstring
    for why this specific order (CRAG first, to avoid wasting cost
    polishing content that gets discarded anyway)."""
    named = [t for t in techniques if t in RETRIEVAL_IMPROVEMENT_TECHNIQUES]

    if "crag" in named:
        cr = crag_evaluate(question, chunks)
        steps["crag"] = StepRecord("crag", True, cr.input_tokens, cr.output_tokens, cr.cost_usd, cr.latency_seconds)
        chunks = cr.usable_chunks
        if cr.grade == "incorrect":
            return chunks  # empty - skip reranking/compression, nothing left to improve

    if "reranking" in named and chunks:
        rr = rerank_chunks(question, chunks, top_k=min(config.TOP_K, len(chunks)))
        steps["reranking"] = StepRecord("reranking", True, rr.input_tokens, rr.output_tokens, rr.cost_usd, rr.latency_seconds)
        chunks = rr.ranked_chunks

    if "compression" in named and chunks:
        cp = compress_chunks(question, chunks)
        steps["compression"] = StepRecord("compression", True, cp.input_tokens, cp.output_tokens, cp.cost_usd, cp.latency_seconds)
        chunks = cp.compressed_chunks

    return chunks


def run_pipeline_for_question(question_id: str, question: str) -> QuestionResult:
    """Run the full conditional pipeline for one question, end to end."""
    steps = _empty_step_records()
    note = QUESTION_NOTES.get(question_id, "")

    # --- Step 1: Route ---
    route_result = route_question(question)
    steps["router"] = StepRecord(
        "router", True, route_result.input_tokens, route_result.output_tokens,
        route_result.cost_usd, route_result.latency_seconds,
    )

    context_text = ""

    # --- Step 2: Retrieval (route-dependent) ---
    if route_result.route == "simple":
        chunks: list[RetrievedChunk] = []

    elif route_result.route == "basic_rag":
        chunks = retrieve(question, top_k=config.TOP_K)

    else:  # advanced_rag
        chunks = run_query_understanding(question, route_result.techniques, steps)
        chunks = run_retrieval_improvement(question, chunks, route_result.techniques, steps)

    # --- Step 3: Generation ---
    if route_result.route == "simple":
        llm_result = _generate_simple_answer(question)
        answer = llm_result.text
        sources = []
        steps["generation"] = StepRecord(
            "generation", True, llm_result.input_tokens, llm_result.output_tokens,
            llm_result.cost_usd, llm_result.latency_seconds,
        )
    else:
        gen_result = generate_answer(question, chunks)
        answer = gen_result.answer
        sources = gen_result.sources
        context_text = gen_result.context_used
        steps["generation"] = StepRecord(
            "generation", True, gen_result.input_tokens, gen_result.output_tokens,
            gen_result.cost_usd, gen_result.latency_seconds,
        )

    # --- Step 4: Evaluation ---
    # Context Relevance and Faithfulness are skipped for "simple" route
    # questions - see module docstring for why (no retrieval was
    # performed by design, so there is no context to judge).
    context_relevance_score = None
    faithfulness_score = None

    if route_result.route != "simple":
        cr = evaluate_context_relevance(question, context_text)
        steps["judge_context_relevance"] = StepRecord("judge_context_relevance", True, cr.input_tokens, cr.output_tokens, cr.cost_usd, cr.latency_seconds)
        context_relevance_score = cr.score

        fa = evaluate_faithfulness(context_text, answer)
        steps["judge_faithfulness"] = StepRecord("judge_faithfulness", True, fa.input_tokens, fa.output_tokens, fa.cost_usd, fa.latency_seconds)
        faithfulness_score = fa.score

    ar = evaluate_answer_relevance(question, answer)
    steps["judge_answer_relevance"] = StepRecord("judge_answer_relevance", True, ar.input_tokens, ar.output_tokens, ar.cost_usd, ar.latency_seconds)

    co = evaluate_correctness(question, answer, context_text)
    steps["judge_correctness"] = StepRecord("judge_correctness", True, co.input_tokens, co.output_tokens, co.cost_usd, co.latency_seconds)

    step_list = list(steps.values())
    total_cost = round(sum(s.cost_usd for s in step_list), 8)
    total_latency = round(sum(s.latency_seconds for s in step_list), 3)

    return QuestionResult(
        question_id=question_id,
        question=question,
        note=note,
        route=route_result.route,
        route_reason=route_result.reason,
        techniques_named_by_router=route_result.techniques,
        answer=answer,
        sources=sources,
        context_relevance_score=context_relevance_score,
        faithfulness_score=faithfulness_score,
        answer_relevance_score=ar.score,
        correctness_score=co.score,
        steps=step_list,
        total_cost_usd=total_cost,
        total_latency_seconds=total_latency,
    )


def print_summary(result: QuestionResult):
    print(f"\n{'=' * 70}")
    print(f"{result.question_id}: {result.question}")
    if result.note:
        print(f"NOTE: {result.note}")
    print(f"{'=' * 70}")
    print(f"Route: {result.route}  |  Reason: {result.route_reason}")
    print(f"Techniques named by router: {result.techniques_named_by_router or 'none'}")
    print(f"\nAnswer:\n{result.answer}")
    if result.sources:
        print("\nSources:")
        for s in result.sources:
            print(f"  - {s['source_file']}, page {s['page_number']}")
    print(f"\nScores: Context Relevance={result.context_relevance_score}  "
          f"Faithfulness={result.faithfulness_score}  "
          f"Answer Relevance={result.answer_relevance_score}  "
          f"Correctness={result.correctness_score}")
    print("\nStep breakdown (executed steps only):")
    for s in result.steps:
        if s.executed:
            print(f"  - {s.step}: {s.input_tokens} in / {s.output_tokens} out tokens, "
                  f"${s.cost_usd}, {s.latency_seconds}s")
    print(f"\nTOTAL: ${result.total_cost_usd} | {result.total_latency_seconds}s")


def print_final_table(results: list[QuestionResult]):
    """Print the exact 'Required Results Table' shape from the task
    (section 14), so it can be copy-pasted straight into the report."""
    print(f"\n\n{'#' * 100}")
    print("REQUIRED RESULTS TABLE")
    print(f"{'#' * 100}")
    header = f"{'ID':<4} {'Route':<13} {'Techniques Used':<40} {'CtxRel':<7} {'Faith':<7} {'AnsRel':<7} {'Correct':<8} {'Cost':<10} {'Latency':<8}"
    print(header)
    print("-" * len(header))
    for r in results:
        techniques_str = ", ".join(r.techniques_named_by_router) if r.techniques_named_by_router else "-"
        ctx = r.context_relevance_score if r.context_relevance_score is not None else "N/A"
        faith = r.faithfulness_score if r.faithfulness_score is not None else "N/A"
        print(f"{r.question_id:<4} {r.route:<13} {techniques_str:<40} {str(ctx):<7} {str(faith):<7} "
              f"{r.answer_relevance_score:<7} {r.correctness_score:<8} ${r.total_cost_usd:<9} {r.total_latency_seconds:<8}")


def main():
    results = []
    start = time.time()

    for qid, question in QUESTIONS.items():
        print(f"\n>>> Running {qid}...")
        result = run_pipeline_for_question(qid, question)
        print_summary(result)
        results.append(result)

    total_time = round(time.time() - start, 1)
    total_cost = round(sum(r.total_cost_usd for r in results), 6)

    print_final_table(results)

    print(f"\n\n{'#' * 70}")
    print(f"ALL {len(results)} QUESTIONS COMPLETE")
    print(f"Total cost across all questions: ${total_cost}")
    print(f"Total wall-clock time: {total_time}s")
    print(f"{'#' * 70}")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.RESULTS_DIR / "results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)

    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    main()