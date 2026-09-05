"""
Component 1/8: Document Loading

Why we need it:
Our raw data lives inside PDF files. An LLM / vector database cannot work
with a PDF file directly - we first need to pull the raw text out of it,
page by page, together with basic metadata (which file, which page,
which document title) so later steps (chunking, retrieval, citation)
know where every piece of text came from.

Why pdfplumber specifically:
These manuals are full of tables (step / actor / action). pdfplumber
extracts text while keeping line breaks and spacing closer to the visual
layout than some alternatives, which matters when a "row" in a table is
really one logical unit of meaning.

Known issue we handle here (Arabic RTL reversal):
pdfplumber (like several PDF text extractors) returns Arabic text in
reversed visual order - both word order and character order within each
Arabic word come out backwards. If left unfixed, every downstream step
(chunking, embedding, retrieval) would be working with unreadable
garbled text, which would silently ruin semantic search quality.
The fix_arabic_line() function below repairs this per line by reversing
word order and reversing characters only within Arabic words, while
leaving embedded numbers/Latin text (dates, codes like "BPM") untouched.
"""

import re
import sys
from pathlib import Path
from dataclasses import dataclass, field

import pdfplumber

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config

ARABIC_CHAR_PATTERN = re.compile(r"[\u0600-\u06FF]")


def fix_arabic_line(line: str) -> str:
    """
    Repair a single line of pdfplumber-extracted text that contains
    reversed Arabic. Words are reversed in order; within each word,
    characters are reversed only if the word actually contains Arabic
    script (so numbers, dates, and Latin acronyms stay untouched).
    """
    words = line.split(" ")
    fixed_words = []
    for word in reversed(words):
        if ARABIC_CHAR_PATTERN.search(word):
            fixed_words.append(word[::-1])
        else:
            fixed_words.append(word)
    return " ".join(fixed_words)


def fix_arabic_text(text: str) -> str:
    """Apply fix_arabic_line() to every line in a multi-line text block."""
    lines = text.split("\n")
    return "\n".join(fix_arabic_line(line) for line in lines)


@dataclass
class PageDocument:
    """One page of text plus the metadata we need to keep alongside it."""
    source_file: str        # e.g. "Central_Alarm_Tasks_And_Procedures_Manual.pdf"
    page_number: int        # 1-indexed page number inside that file
    text: str                # cleaned, correctly-ordered extracted text
    metadata: dict = field(default_factory=dict)


def load_pdf(file_path: Path) -> list[PageDocument]:
    """Extract and clean text from every page of a single PDF file."""
    pages = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            raw_text = page.extract_text() or ""
            clean_text = fix_arabic_text(raw_text)
            pages.append(
                PageDocument(
                    source_file=file_path.name,
                    page_number=i,
                    text=clean_text,
                    metadata={"source_file": file_path.name, "page_number": i},
                )
            )
    return pages


def load_all_pdfs(raw_dir: Path = config.DATA_RAW_DIR) -> list[PageDocument]:
    """Load every PDF found inside the raw data folder."""
    all_pages = []
    pdf_files = sorted(raw_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {raw_dir}")

    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file.name}")
        pages = load_pdf(pdf_file)
        print(f"  -> extracted {len(pages)} pages")
        all_pages.extend(pages)

    return all_pages


if __name__ == "__main__":
    # Quick manual test: run this file directly to sanity-check extraction
    docs = load_all_pdfs()
    print(f"\nTotal pages loaded across all PDFs: {len(docs)}")
    print("\n--- Preview of first page (should now read correctly) ---")
    print(docs[0].text[:500])

    with open("data/processed/preview_check.txt", "w", encoding="utf-8") as f:
        f.write(docs[0].text)
    print("\nSaved a readable preview to data/processed/preview_check.txt - open it in VSCode to verify.")