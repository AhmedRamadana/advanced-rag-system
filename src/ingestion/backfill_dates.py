"""
One-time backfill / repair script for chunk metadata.

Why we need this (two issues fixed at once):
1. BUG FIX: DOCUMENT_METADATA in chunker.py originally used underscore-
   separated filenames (e.g. "Central_Mail_and_Files_Unit_Procedures_
   Manual.pdf") that didn't match the ACTUAL filenames on disk (which
   use spaces). This meant 181 of our 284 chunks (everything except the
   Assets/Warehouse manual) were silently stored WITHOUT document_title,
   issue_date, or last_review_date metadata - a bug that went unnoticed
   because our earlier tests happened to only inspect chunks from the
   one manual whose filename matched correctly.
2. NEW CAPABILITY: ChromaDB's comparison filters ($lt, $gt, etc.) only
   work reliably on numeric fields, not string dates. We add
   issue_yearmonth and last_review_yearmonth as integers (e.g. 202602)
   so Self-Query can filter on date RANGES, not just exact matches.

Why we UPDATE metadata instead of re-embedding everything:
ChromaDB lets us update the metadata attached to existing items without
touching their embeddings. Since we already spent real, rate-limited
API quota embedding all 284 chunks once, re-running the full embedding
pipeline just to fix metadata would be wasteful and risky. This script
only patches metadata using the source_file field (which WAS stored
correctly all along) as the lookup key - no embedding calls at all.

If we skipped this step:
Self-Query would silently fail to find any date-based filter for most
of the corpus (since the metadata it depends on would be missing for
181 of 284 chunks), and any "before/after a given year" style question
would be impossible to answer correctly via metadata filtering.
"""

import os
import sys
from pathlib import Path

os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config
from src.ingestion.vectorstore_builder import COLLECTION_NAME
from src.ingestion.chunker import DOCUMENT_METADATA


def yearmonth_to_int(date_str: str) -> int:
    """Convert 'YYYY-MM' into an integer like 202602, so numeric comparisons work."""
    year, month = date_str.split("-")
    return int(year) * 100 + int(month)


def repair_and_backfill_metadata():
    client = chromadb.PersistentClient(path=str(config.VECTORSTORE_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)

    all_items = collection.get()
    ids = all_items["ids"]
    metadatas = all_items["metadatas"]

    updated_metadatas = []
    fixed_count = 0
    for metadata in metadatas:
        new_metadata = dict(metadata)
        source_file = metadata["source_file"]
        doc_meta = DOCUMENT_METADATA.get(source_file)

        if doc_meta is None:
            print(f"  WARNING: no metadata mapping found for '{source_file}' - leaving as-is.")
            updated_metadatas.append(new_metadata)
            continue

        if "issue_date" not in metadata:
            fixed_count += 1

        new_metadata.update(doc_meta)  # (re)sets document_title, issue_date, last_review_date
        new_metadata["issue_yearmonth"] = yearmonth_to_int(doc_meta["issue_date"])
        new_metadata["last_review_yearmonth"] = yearmonth_to_int(doc_meta["last_review_date"])
        updated_metadatas.append(new_metadata)

    collection.update(ids=ids, metadatas=updated_metadatas)
    print(f"\nUpdated metadata for all {len(ids)} chunks.")
    print(f"Chunks that were missing date metadata and got repaired: {fixed_count}")

    # Verify with a spot-check on one chunk from each source file
    seen_files = set()
    print("\n--- Verification: one sample chunk per document ---")
    fresh = collection.get()
    for chunk_id, metadata in zip(fresh["ids"], fresh["metadatas"]):
        if metadata["source_file"] not in seen_files:
            seen_files.add(metadata["source_file"])
            print(f"{chunk_id}: {metadata}")


if __name__ == "__main__":
    repair_and_backfill_metadata()