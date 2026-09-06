"""
Central configuration for the Advanced RAG project.
Every other file imports settings from here instead of hardcoding them,
so if we need to change a model name or a path, we only change it once.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---- API ----
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ---- Models ----
# We use a single model for ALL text-generation tasks (router, query
# transforms, final answer, judge). We initially tried splitting into a
# strong model + a lightweight model (gemini-3.5-flash-lite) to save
# cost on simple tasks like routing/judging, but the lightweight model
# proved unreliable on the free tier (repeated rate-limit errors causing
# 2-3 minute delays per call, confirmed via retry logging). Given the
# project deadline, we prioritized reliability over marginal cost
# savings and standardized on gemini-3.5-flash-lite everywhere.
GENERATION_MODEL = "gemini-3.5-flash-lite"
EMBEDDING_MODEL = "models/gemini-embedding-001"

# ---- Paths ----
BASE_DIR = Path(__file__).resolve().parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
RESULTS_DIR = BASE_DIR / "results"

# ---- Chunking ----
CHUNK_SIZE = 800        # characters per chunk (tuned for Arabic table-based procedure text)
CHUNK_OVERLAP = 150     # overlap between consecutive chunks to preserve context

# ---- Retrieval ----
TOP_K = 5               # number of chunks retrieved per query

# ---- Pricing (USD per 1M tokens) - gemini-3.5-flash-lite ----
PRICE_INPUT_PER_1M = 0.30
PRICE_OUTPUT_PER_1M = 2.50
PRICE_EMBEDDING_PER_1M = 0.15
