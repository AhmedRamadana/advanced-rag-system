"""
Component 5/8: Retrieval Strategy

Why we need it:
This is where a user's question actually gets connected to our stored
content for the first time. Everything before this (loading, chunking,
embedding, storing) was preparation; this is the step that turns a raw
question into "here are the N most semantically relevant chunks from
the corpus."

How similarity search works (in plain terms):
We convert the question into a vector using the SAME embedding model
we used to store the chunks. ChromaDB then compares that query vector
against all 284 stored vectors and returns the ones that are
mathematically "closest" (most similar in meaning), not the ones that
share the most literal words. This is what lets a question phrased
differently from the manual's exact wording still find the right chunk.

Why task_type="retrieval_query" here (different from storage):
When we stored chunks, we used task_type="retrieval_document". Google's
embedding model produces slightly different (optimized) vectors
depending on whether the text is something to be searched (a query) or
something to be found (a document). Using the wrong task_type for either
side measurably reduces retrieval quality, even though the code would
still run without errors - this is a common, easy-to-miss mistake.

If we skipped this step / got it wrong:
Without it, there is no way to find relevant content for a question at
all. If we get task_type wrong, the code runs and returns *something*,
but retrieval accuracy silently degrades - the kind of subtle bug that
would only show up later as unexplained low "Context Relevance" scores
in evaluation, with no obvious error message pointing to the cause.
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass

os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config
from src.ingestion.vectorstore_builder import COLLECTION_NAME

genai.configure(api_key=config.GEMINI_API_KEY)


@dataclass
class RetrievedChunk:
    """One search result: the chunk text plus its metadata and how close it was to the query."""
    chunk_id: str      # unique identifier, used to de-duplicate results across multiple searches
    text: str
    metadata: dict
    distance: float   # lower = more similar (ChromaDB default is a distance, not a similarity score)


@retry(
    retry=retry_if_exception_type(ResourceExhausted),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    stop=stop_after_attempt(6),
)
def embed_query(query: str) -> list[float]:
    """Embed a user question using the query-optimized mode (see module docstring)."""
    result = genai.embed_content(
        model=config.EMBEDDING_MODEL,
        content=query,
        task_type="retrieval_query",
    )
    return result["embedding"]


def get_collection():
    """Connect to the persistent ChromaDB collection we built earlier."""
    client = chromadb.PersistentClient(path=str(config.VECTORSTORE_DIR))
    return client.get_collection(name=COLLECTION_NAME)


def retrieve(query: str, top_k: int = config.TOP_K) -> list[RetrievedChunk]:
    """
    Core retrieval function: given a question, return the top_k most
    semantically similar chunks stored in the vector database.
    """
    collection = get_collection()
    query_vector = embed_query(query)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
    )

    retrieved = []
    for chunk_id, text, metadata, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        retrieved.append(
            RetrievedChunk(chunk_id=chunk_id, text=text, metadata=metadata, distance=distance)
        )

    return retrieved


if __name__ == "__main__":
    # Quick manual test with a sample question from the corpus's domain.
    test_question = "ما هي إجراءات الجرد السنوي لمستودعات القرطاسية؟"
    print(f"Question: {test_question}\n")

    results = retrieve(test_question, top_k=3)
    for i, r in enumerate(results, start=1):
        print(f"--- Result {i} (distance={r.distance:.4f}) ---")
        print(f"Source: {r.metadata['source_file']}, page {r.metadata['page_number']}")
        print(r.text[:300])
        print()