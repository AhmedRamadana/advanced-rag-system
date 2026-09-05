"""
Query Transformation Technique: HyDE (Hypothetical Document Embeddings)

Why we need it:
Our stored chunks are written in a documentary, procedural style ("The
following steps must be taken... The responsible actor is..."), not in
a question style. A question's embedding and a matching document
chunk's embedding, while related, don't always sit as close together in
vector space as two similarly-styled documents would. HyDE works around
this style mismatch: instead of embedding the raw QUESTION, we first ask
the LLM to imagine a plausible ANSWER, written in the same documentary
style as our corpus, and embed THAT instead. Even if the imagined answer
gets some details wrong, its writing style and vocabulary are much
closer to the real matching chunk than the original question's style
was - which is what actually drives a better nearest-neighbor match.

Why we embed the hypothetical answer with task_type="retrieval_document"
(not "retrieval_query", unlike our other retrieval calls):
The hypothetical text is deliberately written to resemble a document
passage, not a search query. Embedding it in "document" mode - the same
mode used for the real stored chunks - keeps it consistent with how the
target chunks were embedded, which is the whole theoretical basis for
why HyDE works.

When this helps most:
Abstract or conceptual questions, where the gap between how a person
phrases a question and how the manual states the answer is largest.

If we skipped this step:
We would rely purely on direct question-to-document similarity, which
works fine for straightforward factual lookups but can under-perform on
more abstract or awkwardly-phrased questions where a question-shaped
vector doesn't naturally land near an answer-shaped vector in embedding
space.
"""

import sys
from pathlib import Path
from dataclasses import dataclass

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config
from src import llm_client
from src.retrieval.retriever import get_collection, RetrievedChunk

genai.configure(api_key=config.GEMINI_API_KEY)

HYDE_SYSTEM_INSTRUCTION = """You are simulating a passage from an internal bank procedures manual.
Given a question, write a short (2-4 sentence) HYPOTHETICAL passage,
written in the same formal, procedural, documentary style as an
internal operations manual, that would plausibly answer the question -
as if it were an excerpt taken directly from such a manual.

Rules:
- Write it as a declarative, documentary passage - NOT as a
  conversational answer, and NOT addressed to the reader.
- It is fine if the specific details you invent are not accurate - this
  passage is only used to search for real matching content, it will
  never be shown to the end user or treated as a real fact.
- Match the language of the original question.
- Return ONLY the hypothetical passage text, nothing else."""


@dataclass
class HydeResult:
    original_query: str
    hypothetical_document: str
    retrieved_chunks: list[RetrievedChunk]
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_seconds: float


@retry(
    retry=retry_if_exception_type(ResourceExhausted),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    stop=stop_after_attempt(6),
)
def _embed_as_document(text: str) -> list[float]:
    """Embed text in 'document' mode - see module docstring for why this differs from a normal query embedding."""
    result = genai.embed_content(
        model=config.EMBEDDING_MODEL,
        content=text,
        task_type="retrieval_document",
    )
    return result["embedding"]


def generate_hypothetical_document(question: str) -> tuple[str, "llm_client.LLMResponse"]:
    """Ask the LLM to imagine a documentary-style passage that would answer the question."""
    llm_result = llm_client.generate(
        prompt=f"Question: {question}",
        system_instruction=HYDE_SYSTEM_INSTRUCTION,
        temperature=0.4,
    )
    return llm_result.text.strip(), llm_result


def hyde_retrieve(question: str, top_k: int = config.TOP_K) -> HydeResult:
    """
    Generate a hypothetical answer passage, embed it in document mode,
    and use it (instead of the raw question) to search the vector store.
    """
    hypothetical_text, llm_result = generate_hypothetical_document(question)

    collection = get_collection()
    hyde_vector = _embed_as_document(hypothetical_text)

    results = collection.query(query_embeddings=[hyde_vector], n_results=top_k)

    retrieved_chunks = [
        RetrievedChunk(chunk_id=chunk_id, text=text, metadata=metadata, distance=distance)
        for chunk_id, text, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]

    return HydeResult(
        original_query=question,
        hypothetical_document=hypothetical_text,
        retrieved_chunks=retrieved_chunks,
        input_tokens=llm_result.input_tokens,
        output_tokens=llm_result.output_tokens,
        cost_usd=llm_result.cost_usd,
        latency_seconds=llm_result.latency_seconds,
    )


if __name__ == "__main__":
    test_question = "How does the bank keep the alarm system's operations resilient and up to date over time?"
    print(f"Original question: {test_question}\n")

    result = hyde_retrieve(test_question, top_k=5)

    print(f"Hypothetical document generated:\n{result.hypothetical_document}\n")

    print(f"Retrieved chunks: {len(result.retrieved_chunks)}")
    for i, chunk in enumerate(result.retrieved_chunks, start=1):
        print(f"  {i}. (distance={chunk.distance:.4f}) {chunk.metadata['source_file']}, "
              f"page {chunk.metadata['page_number']}")

    print(f"\nCost (HyDE generation call only): ${result.cost_usd}")
    print(f"Latency (HyDE generation call only): {result.latency_seconds}s")