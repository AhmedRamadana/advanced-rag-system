"""
Components 6 & 7 of 8: LLM Integration + Response Generation

Component 6 - LLM Integration (how we pass context to the LLM):
We don't just paste the raw question to Gemini. We build a structured
prompt that clearly separates three things: (a) behavioral instructions
(system_instruction), (b) the retrieved context chunks with their
sources, and (c) the actual question. Keeping these separate and
explicit is what lets the model reliably distinguish "background
material I was given" from "the question I must answer."

Component 7 - Response Generation (how we phrase the prompt correctly):
The exact wording of the prompt is what prevents hallucination. Three
rules are enforced here:
1. Answer ONLY using the provided context - if the answer is not in the
   context, say so explicitly instead of guessing from general
   knowledge. This matters a lot for this project specifically, because
   some of our test questions (Q1, Q6, Q8) are general RAG-theory
   questions with NO answer inside our banking-procedures corpus - the
   model must be willing to say "the provided documents do not address
   this" rather than inventing something that sounds plausible.
2. Cite the source (file name + page number) for every claim, so a
   human reviewer can verify the answer against the original manual.
3. Answer in the same language as the question (our test questions are
   in English, but the corpus is in Arabic - the model needs to bridge
   that gap explicitly rather than defaulting to Arabic-only answers).

If we skipped a careful prompt like this:
The model would likely blend genuine document content with its own
general knowledge without any warning, produce answers that "sound"
grounded but aren't, and never cite where information came from -
making it impossible to later distinguish a well-grounded answer from
a hallucinated one during evaluation.
"""

import sys
from pathlib import Path
from dataclasses import dataclass

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src import llm_client
from src.retrieval.retriever import retrieve, RetrievedChunk

SYSTEM_INSTRUCTION = """You are a careful, precise assistant that answers questions using ONLY
the reference material provided to you in the prompt.

Strict rules you must always follow:
1. Base your answer strictly on the provided context below. Do not use
   outside/general knowledge to fill gaps.
2. If the provided context does not contain enough information to
   answer the question, say so explicitly (e.g. "The provided documents
   do not contain information to answer this question") instead of
   guessing or inventing an answer.
3. Whenever you state a fact drawn from the context, mention which
   source it came from, using the [source_file, page X] format given
   with each context chunk.
4. Answer in the same language the question was asked in.
5. Be concise and directly address the question - do not repeat the
   context verbatim."""

SIMPLE_SYSTEM_INSTRUCTION = """You are a knowledgeable, helpful assistant. Answer the question directly
and accurately using your own general knowledge. Be concise and clear.
Answer in the same language the question was asked in."""


@dataclass
class GenerationResult:
    """The final answer plus everything needed for cost tracking and evaluation."""
    answer: str
    sources: list[dict]
    context_used: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_seconds: float


def format_context(chunks: list[RetrievedChunk]) -> str:
    """
    Turn retrieved chunks into a numbered, source-labeled block of text
    to embed in the prompt, so the model (and later, us, when reading
    the answer) can trace every piece of context back to its origin.
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        source_label = f"[{chunk.metadata['source_file']}, page {chunk.metadata['page_number']}]"
        parts.append(f"--- Context {i} {source_label} ---\n{chunk.text}")
    return "\n\n".join(parts)


def build_prompt(question: str, context: str) -> str:
    """Combine the context and the question into the final prompt text."""
    return f"""CONTEXT:
{context}

QUESTION:
{question}

Answer the question following all the rules given in your instructions."""


def generate_answer(question: str, retrieved_chunks: list[RetrievedChunk]) -> GenerationResult:
    """
    Component 6+7 entry point: given a question and its retrieved
    chunks, build the anti-hallucination prompt and get the final answer.
    Use this for questions routed to "basic_rag" or "advanced_rag" -
    i.e. questions that DID go through retrieval.
    """
    context = format_context(retrieved_chunks)
    prompt = build_prompt(question, context)

    result = llm_client.generate(
        prompt=prompt,
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.2,  # low temperature: we want faithful, consistent answers, not creativity
    )

    sources = [
        {"source_file": c.metadata["source_file"], "page_number": c.metadata["page_number"]}
        for c in retrieved_chunks
    ]

    return GenerationResult(
        answer=result.text,
        sources=sources,
        context_used=context,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=result.cost_usd,
        latency_seconds=result.latency_seconds,
    )


def generate_simple_answer(question: str) -> GenerationResult:
    """
    For questions routed to "simple" by the router - i.e. questions that
    do NOT need retrieval at all (e.g. general RAG-theory questions with
    no answer in our corpus). Uses an unrestricted system instruction so
    the model answers from its own general knowledge, instead of the
    context-only rule in generate_answer() which would incorrectly force
    it to say "insufficient information" for every such question (since
    there would be no context at all to work with).
    """
    result = llm_client.generate(
        prompt=question,
        system_instruction=SIMPLE_SYSTEM_INSTRUCTION,
        temperature=0.3,
    )

    return GenerationResult(
        answer=result.text,
        sources=[],
        context_used="",
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=result.cost_usd,
        latency_seconds=result.latency_seconds,
    )


if __name__ == "__main__":
    # End-to-end manual test: retrieval + generation together (Basic RAG in action)
    test_question = "What are the steps for the annual inventory count of stationery warehouses?"
    print(f"Question: {test_question}\n")

    chunks = retrieve(test_question, top_k=5)
    result = generate_answer(test_question, chunks)

    print("--- Answer ---")
    print(result.answer)
    print("\n--- Sources ---")
    for s in result.sources:
        print(f"  {s['source_file']}, page {s['page_number']}")
    print(f"\nTokens: {result.input_tokens} in / {result.output_tokens} out")
    print(f"Cost: ${result.cost_usd}")
    print(f"Latency: {result.latency_seconds}s")