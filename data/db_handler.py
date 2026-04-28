"""
db_handler.py — SQLAlchemy ORM models and database utilities.

Defines two tables:
    * forbidden_terms  — the classified keyword knowledge base
    * analysis_logs    — immutable audit trail of every scan

Usage:
    python data/db_handler.py            # initialize tables
    python data/db_handler.py --reset    # drop & recreate tables (DEV only)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    event as _sa_event,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# Ensure project root is importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


# ---------------------------------------------------------------------------
# ORM Base
# ---------------------------------------------------------------------------
class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


# ---------------------------------------------------------------------------
# Table A: Forbidden Terms (Knowledge Base)
# ---------------------------------------------------------------------------
class ForbiddenTerm(Base):
    """Classified keyword entry with optional semantic context."""

    __tablename__ = "forbidden_terms"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    term: str = Column(String(512), nullable=False, unique=True, index=True)
    context_description: Optional[str] = Column(
        Text,
        nullable=True,
        doc="Human-readable description of why this term is classified.",
    )
    embedding_json: Optional[str] = Column(
        Text,
        nullable=True,
        doc="JSON-serialised float list of the term's semantic embedding.",
    )
    created_at: datetime = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: datetime = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<ForbiddenTerm id={self.id} term={self.term!r}>"


# ---------------------------------------------------------------------------
# Table B: Analysis Logs (Immutable Audit Trail)
# ---------------------------------------------------------------------------
class AnalysisLog(Base):
    """One row per document scan attempt — never updated, only inserted."""

    __tablename__ = "analysis_logs"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    timestamp: datetime = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )
    filename: str = Column(String(512), nullable=False)
    decision: str = Column(String(16), nullable=False)  # "Approved" | "Blocked"
    reasoning: Optional[str] = Column(Text, nullable=True)
    confidence_score: Optional[float] = Column(Float, nullable=True)
    matched_terms: Optional[str] = Column(
        Text, nullable=True, doc="Comma-separated list of matched forbidden terms."
    )
    feedback: Optional[str] = Column(
        String(16), nullable=True, doc="User feedback: 'like' | 'dislike' | None"
    )

    def __repr__(self) -> str:
        return (
            f"<AnalysisLog id={self.id} file={self.filename!r} "
            f"decision={self.decision!r}>"
        )


# ---------------------------------------------------------------------------
# Engine & Session factory
# ---------------------------------------------------------------------------
def get_engine(database_url: str = config.DATABASE_URL):
    """
    Return a SQLAlchemy engine, creating parent directories for SQLite.

    For SQLite:
      - check_same_thread=False  — allows the same connection to be used
        across threads (required for multi-threaded Flask).
      - WAL journal mode          — multiple readers + one writer can proceed
        concurrently without "database is locked" errors.
      - synchronous=NORMAL        — safe durability without full fsync overhead.
    """
    if database_url.startswith("sqlite:///"):
        db_path = Path(database_url.replace("sqlite:///", ""))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(
            database_url,
            echo=False,
            future=True,
            connect_args={"check_same_thread": False},
        )

        @_sa_event.listens_for(engine, "connect")
        def _set_sqlite_pragmas(dbapi_conn, _record) -> None:
            dbapi_conn.execute("PRAGMA journal_mode=WAL")
            dbapi_conn.execute("PRAGMA synchronous=NORMAL")

        return engine
    return create_engine(database_url, echo=False, future=True)


_engine = get_engine()
SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)


def get_session() -> Session:
    """Return a new SQLAlchemy session (caller is responsible for closing)."""
    return SessionLocal()


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------
def init_db(drop_first: bool = False) -> None:
    """Create all tables (optionally dropping existing ones first)."""
    if drop_first:
        Base.metadata.drop_all(_engine)
        print("[db_handler] All tables dropped.")
    Base.metadata.create_all(_engine)
    print("[db_handler] Tables created / verified.")


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------
def upsert_term(
    term: str,
    context_description: Optional[str] = None,
    embedding_json: Optional[str] = None,
    *,
    session: Optional[Session] = None,
) -> ForbiddenTerm:
    """
    Insert a new term or update its embedding.

    CRITICAL: Existing ``context_description`` values that were manually
    enriched are NEVER overwritten — only the embedding is refreshed.
    """
    own_session = session is None
    if own_session:
        session = get_session()
    try:
        existing = (
            session.query(ForbiddenTerm).filter_by(term=term).first()
        )
        if existing:
            if embedding_json is not None:
                existing.embedding_json = embedding_json
            # Only set description if the term has none — never overwrite manual edits
            if context_description is not None and existing.context_description is None:
                existing.context_description = context_description
            existing.updated_at = datetime.now(timezone.utc)
            if own_session:
                session.commit()
            return existing
        else:
            new_term = ForbiddenTerm(
                term=term,
                context_description=context_description,
                embedding_json=embedding_json,
            )
            session.add(new_term)
            if own_session:
                session.commit()
            return new_term
    except Exception:
        if own_session:
            session.rollback()
        raise
    finally:
        if own_session:
            session.close()


def log_analysis(
    filename: str,
    decision: str,
    reasoning: str,
    confidence_score: float,
    matched_terms: str = "",
    *,
    session: Optional[Session] = None,
) -> AnalysisLog:
    """Persist a scan result to the audit log and return the saved record."""
    own_session = session is None
    if own_session:
        session = get_session()
    try:
        entry = AnalysisLog(
            filename=filename,
            decision=decision,
            reasoning=reasoning,
            confidence_score=confidence_score,
            matched_terms=matched_terms,
        )
        session.add(entry)
        if own_session:
            session.commit()
        return entry
    except Exception:
        if own_session:
            session.rollback()
        raise
    finally:
        if own_session:
            session.close()


def update_feedback(log_id: int, feedback: str) -> None:
    """Store user feedback ('like' | 'dislike') on a scan result."""
    with get_session() as session:
        entry = session.get(AnalysisLog, log_id)
        if entry:
            entry.feedback = feedback
            session.commit()


def get_all_terms(session: Optional[Session] = None) -> list[ForbiddenTerm]:
    """Return all forbidden terms from the DB."""
    own_session = session is None
    if own_session:
        session = get_session()
    try:
        return session.query(ForbiddenTerm).order_by(ForbiddenTerm.term).all()
    finally:
        if own_session:
            session.close()


def get_all_logs(limit: int = 200, session: Optional[Session] = None) -> list[AnalysisLog]:
    """Return recent analysis logs, newest first."""
    own_session = session is None
    if own_session:
        session = get_session()
    try:
        return (
            session.query(AnalysisLog)
            .order_by(AnalysisLog.timestamp.desc())
            .limit(limit)
            .all()
        )
    finally:
        if own_session:
            session.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IAI DLP — Database manager")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate all tables (DEV use only!)",
    )
    args = parser.parse_args()
    init_db(drop_first=args.reset)
    print("[db_handler] Done.")
