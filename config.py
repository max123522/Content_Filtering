"""
config.py — Central configuration for the IAI Semantic DLP system.

Reads environment variables from a .env file and exposes them as typed
constants.  A single ENV flag toggles between development (OpenAI/Anthropic)
and production (IAI on-prem) LLM endpoints.
"""

import os
from pathlib import Path
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Load .env (silent if missing – production may inject vars directly)
# ---------------------------------------------------------------------------
_BASE_DIR = Path(__file__).resolve().parent
load_dotenv(_BASE_DIR / ".env", override=True)

# ---------------------------------------------------------------------------
# Environment toggle
# ---------------------------------------------------------------------------
ENV: str = os.getenv("ENV", "DEV").upper()  # DEV | PROD

# ---------------------------------------------------------------------------
# LLM settings
# ---------------------------------------------------------------------------
# To switch provider, set LLM_PROVIDER=openai or LLM_PROVIDER=anthropic in .env
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai").lower()  # openai | anthropic

if ENV == "PROD":
    LLM_BASE_URL: str = os.getenv("PROD_LLM_BASE_URL", "http://internal-iai-gpt-oss-12b/v1")
    LLM_API_KEY: str = os.getenv("PROD_LLM_API_KEY", "na")
    LLM_MODEL: str = os.getenv("PROD_LLM_MODEL", "iai-gpt-oss-12b")
elif LLM_PROVIDER == "anthropic":
    LLM_BASE_URL: str | None = None
    LLM_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    LLM_MODEL: str = os.getenv("DEV_LLM_MODEL", "claude-sonnet-4-5")
else:  # openai (default)
    LLM_BASE_URL: str | None = None
    LLM_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL: str = os.getenv("DEV_LLM_MODEL", "gpt-4o")

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
# DEV:  runs locally via SentenceTransformer (no external calls).
# PROD: calls the IAI on-prem embedding endpoint (OpenAI-compatible API).
# Force multilingual-e5-base — no xformers dependency, runs on CPU.
# Snowflake Arctic requires xformers (Linux/GPU only); do NOT use it here.
EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-base")

if ENV == "PROD":
    EMBED_BASE_URL: str = os.getenv("PROD_LLM_BASE_URL_EMBED", "")
    EMBED_API_KEY: str = os.getenv("PROD_LLM_API_KEY", "na")
    EMBED_MODEL: str = os.getenv("PROD_EMBED_MODEL", EMBEDDING_MODEL_NAME)
else:
    EMBED_BASE_URL: str | None = None
    EMBED_API_KEY: str = ""
    EMBED_MODEL: str = EMBEDDING_MODEL_NAME

# ---------------------------------------------------------------------------
# Reranker model (local only – disabled in PROD, no internet in closed env)
# ---------------------------------------------------------------------------
RERANKER_MODEL_NAME: str = os.getenv(
    "RERANKER_MODEL_NAME", "BAAI/bge-reranker-base"
)
ENABLE_RERANKER: bool = ENV != "PROD"

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
SQLITE_DB_PATH: Path = _BASE_DIR / "data" / "iai_dlp.db"
DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{SQLITE_DB_PATH}")

# ---------------------------------------------------------------------------
# Document ingestion
# ---------------------------------------------------------------------------
TERMS_DOCX_PATH: Path = _BASE_DIR / "words_content_filtering.docx"

# ---------------------------------------------------------------------------
# Flask
# ---------------------------------------------------------------------------
print(f"[CONFIG] provider={LLM_PROVIDER} model={LLM_MODEL} key={LLM_API_KEY[:16]}...")
FLASK_SECRET_KEY: str = os.getenv("FLASK_SECRET_KEY", "iai-dlp-super-secret-2024")
FLASK_DEBUG: bool = ENV == "DEV"
FLASK_PORT: int = int(os.getenv("FLASK_PORT", "5000"))

# ---------------------------------------------------------------------------
# Analysis settings
# ---------------------------------------------------------------------------
SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.65"))
TOP_K_TERMS: int = int(os.getenv("TOP_K_TERMS", "20"))
SEGMENT_WINDOW_SIZE: int = int(os.getenv("SEGMENT_WINDOW_SIZE", "150"))

# ---------------------------------------------------------------------------
# Concurrency settings
# ---------------------------------------------------------------------------
# MAX_PARALLEL_DOCS   — max files accepted in a single /scan request
# MAX_LLM_CONCURRENCY — max parallel LLM calls per document (protects IAI LLM server)
# MAX_TOTAL_ANALYSES  — global semaphore: max simultaneous analyses across ALL users
# TERMS_CACHE_TTL     — seconds to cache forbidden-terms list in memory
MAX_PARALLEL_DOCS: int   = int(os.getenv("MAX_PARALLEL_DOCS",   "5"))
MAX_LLM_CONCURRENCY: int = int(os.getenv("MAX_LLM_CONCURRENCY", "3"))
MAX_TOTAL_ANALYSES: int  = int(os.getenv("MAX_TOTAL_ANALYSES",  "10"))
TERMS_CACHE_TTL: int     = int(os.getenv("TERMS_CACHE_TTL",     "300"))
