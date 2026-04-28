"""
graph_manager.py — LangGraph state-machine for the Semantic DLP pipeline.

Graph nodes (in order):
    1. parse_document      — Extract text from the uploaded file
    2. detect_candidates   — Vector search for semantically similar terms
    3. rerank_candidates   — BGE-Reranker cross-checks hits
    4. inject_context      — Fetch context_description from DB for each hit
    5. llm_reason          — PydanticAI agent analyses each segment
    6. aggregate_decision  — Merge segment decisions into document verdict
    7. persist_log         — Save audit record to analysis_logs table

State flows strictly forward; there are no cycles in the analysis path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

from langgraph.graph import StateGraph, END
from pydantic import BaseModel

import config
from core.document_parser import extract_and_segment, ExtractionError
from core.agents import DLPDecision, analyze_segment, analyze_full_document
from data.db_handler import (
    get_all_terms,
    log_analysis,
    ForbiddenTerm,
)


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class DLPState(BaseModel):
    """Mutable state object passed between LangGraph nodes."""

    # Input
    file_bytes: bytes = b""
    filename: str = ""

    # Intermediate
    full_text: str = ""
    segments: list[str] = []
    all_terms: list[dict] = []          # serialised ForbiddenTerm records
    term_embeddings: list[list[float]] = []  # parallel to all_terms
    candidate_hits: list[list[tuple[str, str, float]]] = []  # per segment
    reranked_hits: list[list[tuple[str, float]]] = []        # per segment

    # Output
    final_decision: Optional[DLPDecision] = None
    log_id: Optional[int] = None
    error: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

def node_parse_document(state: DLPState) -> DLPState:
    """Node 1 — Parse file bytes into text segments."""
    try:
        full_text, segments = extract_and_segment(
            state.file_bytes, filename=state.filename
        )
        state.full_text = full_text
        state.segments = segments
    except ExtractionError as exc:
        state.error = str(exc)
    return state


def node_detect_candidates(state: DLPState) -> DLPState:
    """Node 2 — Embed segments + terms; run vector similarity search."""
    if state.error or not state.segments:
        return state

    try:
        from models.embedding_service import EmbeddingService
        import numpy as np

        emb = EmbeddingService()
        db_terms: list[ForbiddenTerm] = get_all_terms()

        # Build parallel term list and embedding matrix
        term_records: list[dict] = []
        valid_vecs: list[list[float]] = []

        for t in db_terms:
            if t.embedding_json:
                try:
                    vec = json.loads(t.embedding_json)
                    term_records.append(
                        {"id": t.id, "term": t.term, "context": t.context_description or ""}
                    )
                    valid_vecs.append(vec)
                except Exception:
                    pass

        state.all_terms = term_records

        if not valid_vecs:
            # No embeddings available — fallback to substring match
            state.candidate_hits = _fallback_substring(state.segments, db_terms)
            return state

        term_matrix = np.array(valid_vecs, dtype=np.float32)
        candidate_hits: list[list[tuple[str, str, float]]] = []

        for segment in state.segments:
            q_vec = emb.encode(segment)
            hits = emb.top_k_similar(q_vec, term_matrix, k=config.TOP_K_TERMS)
            seg_hits = [
                (
                    term_records[idx]["term"],
                    term_records[idx]["context"],
                    score,
                )
                for idx, score in hits
            ]
            candidate_hits.append(seg_hits)

        state.candidate_hits = candidate_hits

    except Exception as exc:
        state.error = f"Embedding failed: {exc}"

    return state


def _fallback_substring(
    segments: list[str],
    db_terms: list[ForbiddenTerm],
) -> list[list[tuple[str, str, float]]]:
    """
    Fallback when embeddings are unavailable: simple case-insensitive
    substring search with a fixed score of 0.6.
    """
    result: list[list[tuple[str, str, float]]] = []
    for segment in segments:
        seg_lower = segment.lower()
        hits: list[tuple[str, str, float]] = []
        for t in db_terms:
            if t.term.lower() in seg_lower:
                hits.append((t.term, t.context_description or "", 0.6))
        result.append(hits)
    return result


def node_rerank_candidates(state: DLPState) -> DLPState:
    """Node 3 — Cross-encoder reranking to filter false positives."""
    if state.error or not state.candidate_hits:
        return state

    if not config.ENABLE_RERANKER:
        # PROD: reranker disabled (no internet in closed env) — pass through raw hits
        state.reranked_hits = [
            [(term, score) for term, _, score in hits]
            for hits in state.candidate_hits
        ]
        return state

    try:
        from models.reranker_service import RerankerService
        reranker = RerankerService()
        reranked: list[list[tuple[str, float]]] = []

        for segment, hits in zip(state.segments, state.candidate_hits):
            if not hits:
                reranked.append([])
                continue
            candidates = [(term, ctx) for term, ctx, _ in hits]
            ranked = reranker.rerank_candidates(segment, candidates)
            reranked.append(ranked)

        state.reranked_hits = reranked

    except Exception as exc:
        # Reranker unavailable — pass through raw hits
        state.reranked_hits = [
            [(term, score) for term, _, score in hits]
            for hits in state.candidate_hits
        ]

    return state


def node_inject_context(state: DLPState) -> DLPState:
    """
    Node 4 — Build (term, context_description) pairs for LLM prompt injection.

    This node simply selects the DB context descriptions for the top
    reranked hits per segment.  The actual injection happens in node_llm_reason.
    """
    # Nothing to transform here — reranked_hits already carry the data
    # needed for prompt injection. The node is a placeholder for future
    # enrichment (e.g., fetching related RAG passages).
    return state


def node_llm_reason(state: DLPState) -> DLPState:
    """Node 5 — Send each segment to the LLM for semantic analysis."""
    if state.error:
        return state

    segments_with_context: list[tuple[str, list[tuple[str, str]], list[str]]] = []

    # Build context for each segment from candidate_hits (contains full ctx)
    for i, segment in enumerate(state.segments):
        raw_hits = state.candidate_hits[i] if i < len(state.candidate_hits) else []
        reranked = state.reranked_hits[i] if i < len(state.reranked_hits) else []

        # Build a lookup from term → context from raw hits
        ctx_map = {term: ctx for term, ctx, _ in raw_hits}
        # Use reranked order for injection (top reranked terms)
        reranked_terms = [term for term, _ in reranked] if reranked else [t for t, _, _ in raw_hits[:3]]
        injected_contexts = [(t, ctx_map.get(t, "")) for t in reranked_terms if t in ctx_map]
        candidate_names = [t for t, _, _ in raw_hits]

        segments_with_context.append((segment, injected_contexts, candidate_names))

    # Only analyse segments with candidate hits (optimisation)
    flagged = [s for s in segments_with_context if s[2]]
    if not flagged:
        state.final_decision = DLPDecision(
            decision="Approved",
            reasoning="No sensitive terms detected in document.",
            confidence_score=1.0,
            matched_terms=[],
        )
        return state

    state.final_decision = analyze_full_document(flagged)
    return state


def node_aggregate_decision(state: DLPState) -> DLPState:
    """Node 6 — Ensure a final decision exists (handles error paths)."""
    if state.final_decision is None:
        if state.error:
            state.final_decision = DLPDecision(
                decision="Blocked",
                reasoning=f"Analysis error: {state.error}",
                confidence_score=0.0,
                matched_terms=[],
            )
        else:
            state.final_decision = DLPDecision(
                decision="Approved",
                reasoning="Document passed all checks.",
                confidence_score=1.0,
                matched_terms=[],
            )
    return state


def node_persist_log(state: DLPState) -> DLPState:
    """Node 7 — Write audit record to analysis_logs table."""
    decision = state.final_decision
    if decision is None:
        return state
    try:
        entry = log_analysis(
            filename=state.filename,
            decision=decision.decision,
            reasoning=decision.reasoning,
            confidence_score=decision.confidence_score,
            matched_terms=", ".join(decision.matched_terms),
        )
        state.log_id = entry.id
    except Exception as exc:
        print(f"[graph_manager] Failed to persist log: {exc}")
    return state


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Assemble and compile the LangGraph DLP workflow."""
    graph = StateGraph(DLPState)

    graph.add_node("parse_document", node_parse_document)
    graph.add_node("detect_candidates", node_detect_candidates)
    graph.add_node("rerank_candidates", node_rerank_candidates)
    graph.add_node("inject_context", node_inject_context)
    graph.add_node("llm_reason", node_llm_reason)
    graph.add_node("aggregate_decision", node_aggregate_decision)
    graph.add_node("persist_log", node_persist_log)

    graph.set_entry_point("parse_document")
    graph.add_edge("parse_document", "detect_candidates")
    graph.add_edge("detect_candidates", "rerank_candidates")
    graph.add_edge("rerank_candidates", "inject_context")
    graph.add_edge("inject_context", "llm_reason")
    graph.add_edge("llm_reason", "aggregate_decision")
    graph.add_edge("aggregate_decision", "persist_log")
    graph.add_edge("persist_log", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_compiled_graph = None


def get_graph():
    """Return the singleton compiled graph (lazy initialisation)."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_analysis(
    file_bytes: bytes,
    filename: str,
) -> DLPState:
    """
    Run the full DLP analysis pipeline on an uploaded document.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename:   Original filename (used for format detection + logging).

    Returns:
        Final DLPState with decision, reasoning, and log_id populated.
    """
    graph = get_graph()
    initial_state = DLPState(file_bytes=file_bytes, filename=filename)
    result = graph.invoke(initial_state)

    # LangGraph returns a plain dict — reconstruct the typed DLPState.
    if isinstance(result, dict):
        fd = result.get("final_decision")
        if isinstance(fd, dict):
            result["final_decision"] = DLPDecision(**fd)
        # file_bytes is not forwarded by LangGraph output — restore it.
        result.setdefault("file_bytes", file_bytes)
        result.setdefault("filename", filename)
        return DLPState(**result)

    return result
