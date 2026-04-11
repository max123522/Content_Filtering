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
# Embedding model (always local – Snowflake Arctic v2)
# ---------------------------------------------------------------------------
# Force multilingual-e5-base — no xformers dependency, runs on CPU.
# Snowflake Arctic requires xformers (Linux/GPU only); do NOT use it here.
EMBEDDING_MODEL_NAME: str = "intfloat/multilingual-e5-base"

# ---------------------------------------------------------------------------
# Reranker model (always local – BGE)
# ---------------------------------------------------------------------------
RERANKER_MODEL_NAME: str = os.getenv(
    "RERANKER_MODEL_NAME", "BAAI/bge-reranker-base"
)

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
SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.45"))
TOP_K_TERMS: int = int(os.getenv("TOP_K_TERMS", "5"))
SEGMENT_WINDOW_SIZE: int = int(os.getenv("SEGMENT_WINDOW_SIZE", "150"))
