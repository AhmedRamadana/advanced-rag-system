"""
Components 3 & 4 of 8: Embedding Model + Vector Database

Why we need Embeddings (Component 3):
Computers cannot compare the "meaning" of two pieces of text directly.
An embedding model converts each chunk of text into a list of numbers
(a vector) such that chunks with similar meaning end up close together
in that numeric space. This is what lets us later ask "which stored
chunk is closest in meaning to the user's question?" instead of relying
on exact keyword matches.

Why gemini-embedding-001 specifically:
It is Google's current stable embedding model (text-embedding-004, the
older one, was shut down in January 2026). It explicitly supports
Arabic among 100+ languages, which matters because our entire corpus
is in Arabic.

Why a Vector Database / ChromaDB (Component 4):
Once we have hundreds of vectors, we need a place to store them that can
answer "give me the top-K closest vectors to this query vector"
efficiently, without us writing our own similarity-search math by hand.
ChromaDB is an embedded (no separate server needed) vector database,
well suited to a project of this size, and persists data to disk so we
don't have to recompute embeddings every time we run the project.

If we skipped this step:
We would have text chunks but no way to numerically compare a user's
question against them - there would be no semantic search at all, only
literal keyword matching (like Ctrl+F), which fails whenever the user
phrases their question differently from the exact wording in the manual.

Resilience additions in this version:
Free-tier API keys have request-rate limits. Embedding 284 chunks in a
row can occasionally hit that limit mid-run. Two additions handle this:
1. Retry with exponential backoff (via `tenacity`) - if a request is
   rate-limited, we wait and try again instead of crashing.
2. Resumability - before embedding, we check which chunk_ids are
   already stored in ChromaDB and skip them, so re-running this script
   after an interruption continues from where it stopped instead of
   re-spending quota on chunks we already processed.
"""

import os
import sys
import time
from pathlib import Path

# Silence ChromaDB's anonymous telemetry - it was printing harmless but
# noisy "Failed to send telemetry event" warnings in the terminal.
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config
from src.ingestion.loader import load_all_pdfs
from src.ingestion.chunker import chunk_all_documents

genai.configure(api_key=config.GEMINI_API_KEY)

COLLECTION_NAME = "hbtf_procedures"


@retry(
    retry=retry_if_exception_type(ResourceExhausted),
    wait=wait_exponential(multiplier=2, min=5, max=60),  # wait 5s, 10s, 20s... up to 60s
    stop=stop_after_attempt(6),
)
def embed_text(text: str, task_type: str = "retrieval_document") -> list[float]:
    """
    Convert a single piece of text into an embedding vector.
    task_type differs between the text we STORE ("retrieval_document")
    and the text we SEARCH WITH later ("retrieval_query") - the model
    uses this hint to produce vectors optimized for each role.

    If Google returns a rate-limit error (ResourceExhausted), the
    @retry decorator automatically waits and tries again, up to 6
    attempts, instead of letting the whole script crash.
    """
    result = genai.embed_content(
        model=config.EMBEDDING_MODEL,
        content=text,
        task_type=task_type,
    )
    return result["embedding"]


def build_vectorstore():
    """
    Full pipeline: load PDFs -> chunk them -> embed each NEW chunk ->
    store everything in a persistent ChromaDB collection on disk.
    Chunks already present in the collection (from a previous run) are
    skipped, so this function is safe to re-run after an interruption.
    """
    print("Step 1/3: Loading and chunking documents...")
    all_pages = load_all_pdfs()
    all_chunks = chunk_all_documents(all_pages)
    print(f"Total chunks in corpus: {len(all_chunks)}")

    print("\nStep 2/3: Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=str(config.VECTORSTORE_DIR))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    already_done = set(collection.get()["ids"])
    remaining_chunks = [c for c in all_chunks if c.chunk_id not in already_done]
    print(f"Already embedded previously: {len(already_done)}")
    print(f"Remaining to embed now: {len(remaining_chunks)}")

    if not remaining_chunks:
        print("\nNothing left to do - collection is already complete.")
        return

    print("\nStep 3/3: Embedding and storing each remaining chunk...")
    for i, chunk in enumerate(remaining_chunks, start=1):
        vector = embed_text(chunk.text, task_type="retrieval_document")

        collection.add(
            ids=[chunk.chunk_id],
            embeddings=[vector],
            documents=[chunk.text],
            metadatas=[chunk.metadata],
        )

        if i % 20 == 0 or i == len(remaining_chunks):
            print(f"  Embedded {i}/{len(remaining_chunks)} remaining chunks "
                  f"(collection total: {collection.count()})")

        time.sleep(0.5)  # slightly more conservative pacing to avoid rate limits

    print(f"\nDone. Collection '{COLLECTION_NAME}' now has {collection.count()} items.")
    print(f"Stored persistently at: {config.VECTORSTORE_DIR}")


if __name__ == "__main__":
    build_vectorstore()