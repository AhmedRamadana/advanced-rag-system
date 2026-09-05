"""
Component 2/8: Text Chunking

Why we need it:
A whole PDF page (or a whole procedure that spans several pages) is too
large and too "mixed" a unit to search against directly. If we embedded
one giant block of text, the embedding would represent an *average* of
many different steps and actors, and a specific question about one step
would retrieve a bloated chunk full of irrelevant extra text - hurting
both retrieval precision and LLM cost (more tokens to read per call).

Why recursive, structure-aware splitting (not a blind fixed-size cut):
These manuals don't use a single consistent numbering scheme across all
three PDFs, so writing a strict "split by procedure number" regex would
be fragile and could silently produce wrong splits on the documents that
don't match the pattern.
Instead, we try to cut at the most natural boundary available, in order
of preference: a blank line (paragraph break), then a single line break
(each table row / step is usually its own line in the extracted text),
then a space. We only fall back to a hard character cut if none of those
boundaries exist within a reasonable distance - this preserves the
"step" and "actor -> action" structure of the tables automatically,
without needing brittle rules tied to this exact document's layout.

If we skipped this step and just embedded whole pages:
Retrieval would return oversized, noisy chunks; the LLM would receive
irrelevant context alongside the relevant part, increasing both cost
(more tokens) and the risk of the model getting confused about which
part of the context actually answers the question.
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config
from src.ingestion.loader import PageDocument, load_all_pdfs

# Try splitting at these boundaries, in this order of preference.
SEPARATORS = ["\n\n", "\n", ". ", " "]

# Document-level metadata (issue date / last review date).
# These dates come straight from the header table printed on every page
# of each manual. We read them manually instead of parsing them with a
# regex because the PDF extraction of Arabic ligatures (e.g. "الإصدار")
# is inconsistent enough that automated date-field extraction would be
# unreliable - using the real, human-verified dates is safer than a
# fragile automated guess, and still gives us genuine (not synthetic)
# metadata to use later for date-based filtering (Self-Query).
DOCUMENT_METADATA = {
    "Assets and wearhouse operation Tasks and Procedures Manual.pdf": {
        "document_title": "دليل إجراءات وحدة الموجودات وعمليات المستودعات",
        "issue_date": "2026-02",
        "last_review_date": "2026-02",
    },
    "Central Mail and Files Unit Procedures Manual.pdf": {
        "document_title": "دليل اجراءات عمل وحدة البريد المركزي والملفات",
        "issue_date": "2026-02",
        "last_review_date": "2026-02",
    },
    "Central Alarm Tasks And Procedures Manual.pdf": {
        "document_title": "دليل إجراءات وحدة الانذار المركزي",
        "issue_date": "2024-08",
        "last_review_date": "2026-02",
    },
}


@dataclass
class Chunk:
    """One retrievable unit of text, ready to be embedded."""
    chunk_id: str
    text: str
    source_file: str
    page_number: int
    metadata: dict = field(default_factory=dict)


def _split_text(text: str, chunk_size: int, overlap: int, separators: list[str]) -> list[str]:
    """
    Recursively split `text` into pieces of at most `chunk_size` characters,
    preferring to cut at the earliest separator in `separators` (tried in
    order) so we don't break in the middle of a line or word unnecessarily.
    Consecutive chunks share `overlap` characters so context isn't lost
    right at a cut point.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    separator = separators[0] if separators else ""
    remaining_separators = separators[1:] if len(separators) > 1 else []

    pieces = text.split(separator) if separator else list(text)

    chunks = []
    current = ""
    for piece in pieces:
        candidate = current + (separator if current else "") + piece
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
                # start next chunk with overlap from the end of the previous one
                current = current[-overlap:] + separator + piece
            else:
                # a single piece is already too big -> split it further
                if remaining_separators:
                    chunks.extend(_split_text(piece, chunk_size, overlap, remaining_separators))
                else:
                    # last resort: hard cut by character count
                    for i in range(0, len(piece), chunk_size - overlap):
                        chunks.append(piece[i:i + chunk_size])
                current = ""
    if current.strip():
        chunks.append(current)

    return [c for c in chunks if c.strip()]


def chunk_document(pages: list[PageDocument]) -> list[Chunk]:
    """
    Take all pages belonging to ONE source file (in page order) and turn
    them into a list of Chunks, keeping track of which page each chunk
    started on.
    """
    source_file = pages[0].source_file
    doc_meta = DOCUMENT_METADATA.get(source_file, {})

    chunks = []
    chunk_counter = 0

    for page in pages:
        pieces = _split_text(page.text, config.CHUNK_SIZE, config.CHUNK_OVERLAP, SEPARATORS)
        for piece in pieces:
            chunk_counter += 1
            chunks.append(
                Chunk(
                    chunk_id=f"{source_file}::chunk_{chunk_counter}",
                    text=piece,
                    source_file=source_file,
                    page_number=page.page_number,
                    metadata={
                        "source_file": source_file,
                        "page_number": page.page_number,
                        **doc_meta,
                    },
                )
            )
    return chunks


def chunk_all_documents(all_pages: list[PageDocument]) -> list[Chunk]:
    """Group pages by source file, then chunk each document separately."""
    pages_by_file: dict[str, list[PageDocument]] = {}
    for page in all_pages:
        pages_by_file.setdefault(page.source_file, []).append(page)

    all_chunks = []
    for source_file, pages in pages_by_file.items():
        pages_sorted = sorted(pages, key=lambda p: p.page_number)
        doc_chunks = chunk_document(pages_sorted)
        print(f"{source_file}: {len(pages)} pages -> {len(doc_chunks)} chunks")
        all_chunks.extend(doc_chunks)

    return all_chunks


if __name__ == "__main__":
    all_pages = load_all_pdfs()
    all_chunks = chunk_all_documents(all_pages)

    print(f"\nTotal chunks created: {len(all_chunks)}")
    print("\n--- Preview of chunk #1 ---")
    print(f"chunk_id: {all_chunks[0].chunk_id}")
    print(f"metadata: {all_chunks[0].metadata}")
    print(f"text ({len(all_chunks[0].text)} chars):\n{all_chunks[0].text}")

    with open("data/processed/chunk_preview.txt", "w", encoding="utf-8") as f:
        for c in all_chunks[:5]:
            f.write(f"=== {c.chunk_id} (page {c.page_number}, {len(c.text)} chars) ===\n")
            f.write(c.text + "\n\n")
    print("\nSaved first 5 chunks to data/processed/chunk_preview.txt for inspection.")