"""
Retrieval Improvement Technique: Contextual Compression

Why we need it:
Our chunks (~800 characters each) often contain a mix of genuinely
relevant content and "noise" - repeated page headers (version number,
issue date, document title printed on every page), or parts of a
procedure table unrelated to the specific question being asked. Sending
the full raw chunk to the final generation step means paying for and
exposing the model to that noise on every single call, even though only
a fraction of the chunk actually answers the question.

How Contextual Compression fixes this:
Instead of trusting each retrieved chunk as-is, we ask the LLM to read
it against the specific question and extract ONLY the portion that is
actually relevant - discarding boilerplate headers and unrelated
sentences. If a chunk turns out to contain nothing relevant at all
(a false positive from retrieval), we return an empty result for it, so
downstream code can drop it entirely rather than sending noise to
generation.

Why we batch all candidate chunks into ONE prompt:
Same reasoning as reranking - one call per chunk would multiply cost
and latency by the number of chunks. Batching keeps this efficient.

If we skipped this step:
Every retrieved chunk, including its repeated headers and any
tangential content, would be forwarded in full to the final generation
prompt - increasing token cost on every single question, and giving the
model more irrelevant text to potentially get distracted by.
"""

import json
import re
import sys
from pathlib import Path
from dataclasses import dataclass, replace

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src import llm_client
from src.retrieval.retriever import RetrievedChunk

COMPRESSION_SYSTEM_INSTRUCTION = """You are a context-compression assistant for a document retrieval
system. You will be given a question and a numbered list of retrieved
passages, each of which may contain a mix of relevant content, repeated
document headers/boilerplate, and unrelated text.

For EACH passage, extract ONLY the sentence(s) that are directly
relevant to answering the question, in their original wording (do not
paraphrase or translate). Discard headers, boilerplate, and unrelated
content. If a passage contains NOTHING relevant to the question at all,
return an empty string for it.

Return ONLY a JSON array of strings, one per passage, in the SAME order
and SAME count as the input list. Example shape for 3 passages:
["relevant excerpt from passage 1", "", "relevant excerpt from passage 3"]"""


@dataclass
class CompressionResult:
    original_query: str
    compressed_chunks: list[RetrievedChunk]  # only chunks that had relevant content kept
    dropped_count: int                        # how many chunks had nothing relevant
    original_char_count: int
    compressed_char_count: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_seconds: float


def _extract_json_array(raw_text: str) -> list[str]:
    """Strip any markdown code-fence wrapping before parsing the JSON array."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE)
    return json.loads(cleaned)


def _format_candidates(chunks: list[RetrievedChunk]) -> str:
    """Turn candidate chunks into a numbered list for the compression prompt."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(f"[Passage {i}]\n{chunk.text}")
    return "\n\n".join(parts)


def compress_chunks(question: str, chunks: list[RetrievedChunk]) -> CompressionResult:
    """
    Extract only the question-relevant excerpt from each chunk, and
    drop any chunk that turns out to contain nothing relevant at all.
    """
    original_char_count = sum(len(c.text) for c in chunks)

    prompt = f"""Question: {question}

Candidate passages:
{_format_candidates(chunks)}"""

    llm_result = llm_client.generate(
        prompt=prompt,
        system_instruction=COMPRESSION_SYSTEM_INSTRUCTION,
        temperature=0.0,
    )

    try:
        excerpts = _extract_json_array(llm_result.text)
        if len(excerpts) != len(chunks):
            raise ValueError("Excerpt count doesn't match input chunk count")
    except (json.JSONDecodeError, ValueError, AttributeError):
        # Safe fallback: if compression fails, keep all chunks uncompressed.
        excerpts = [c.text for c in chunks]

    compressed_chunks = []
    dropped_count = 0
    for chunk, excerpt in zip(chunks, excerpts):
        if excerpt.strip():
            compressed_chunks.append(replace(chunk, text=excerpt.strip()))
        else:
            dropped_count += 1

    compressed_char_count = sum(len(c.text) for c in compressed_chunks)

    return CompressionResult(
        original_query=question,
        compressed_chunks=compressed_chunks,
        dropped_count=dropped_count,
        original_char_count=original_char_count,
        compressed_char_count=compressed_char_count,
        input_tokens=llm_result.input_tokens,
        output_tokens=llm_result.output_tokens,
        cost_usd=llm_result.cost_usd,
        latency_seconds=llm_result.latency_seconds,
    )


if __name__ == "__main__":
    from src.retrieval.retriever import retrieve

    test_question = "What are the steps for the annual inventory count of stationery warehouses?"
    print(f"Question: {test_question}\n")

    candidates = retrieve(test_question, top_k=5)
    result = compress_chunks(test_question, candidates)

    print(f"Chunks kept: {len(result.compressed_chunks)} | Chunks dropped as irrelevant: {result.dropped_count}\n")

    for i, chunk in enumerate(result.compressed_chunks, start=1):
        print(f"--- Compressed chunk {i} ({chunk.metadata['source_file']}, page {chunk.metadata['page_number']}) ---")
        print(chunk.text)
        print()

    reduction_pct = 100 * (1 - result.compressed_char_count / result.original_char_count)
    print(f"Original total characters: {result.original_char_count}")
    print(f"Compressed total characters: {result.compressed_char_count}")
    print(f"Reduction: {reduction_pct:.1f}%")
    print(f"\nCost (compression call only): ${result.cost_usd}")
    print(f"Latency (compression call only): {result.latency_seconds}s")