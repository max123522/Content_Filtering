"""
agents.py — PydanticAI agents for semantic DLP reasoning.

Each agent wraps an LLM call and enforces a strict JSON output schema
via Pydantic models, preventing raw-string reasoning from leaking into
the pipeline.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from pydantic import BaseModel, Field

import config

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class DLPDecision(BaseModel):
    """Structured LLM output for a single document-segment analysis."""

    decision: str = Field(
        ...,
        description="'Approved' if the segment is safe to export, 'Blocked' if it contains classified information.",
        pattern="^(Approved|Blocked)$",
    )
    reasoning: str = Field(
        ...,
        description="A detailed semantic explanation (2-5 sentences) in English.",
        min_length=10,
    )
    confidence_score: float = Field(
        ...,
        description="Model certainty from 0.0 (uncertain) to 1.0 (certain).",
        ge=0.0,
        le=1.0,
    )
    matched_terms: list[str] = Field(
        default_factory=list,
        description="List of forbidden terms found in the segment.",
    )


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

def build_system_prompt(injected_contexts: list[tuple[str, str]]) -> str:
    """
    Build a context-enriched system prompt.

    Args:
        injected_contexts: List of (term, context_description) pairs retrieved
                           from the DB for this document segment.

    Returns:
        Full system prompt string.
    """
    base = (
        "You are an expert security analyst for Israel Aerospace Industries (IAI), "
        "specialising in Data Loss Prevention (DLP). Your role is to determine "
        "whether a given text segment contains classified or sensitive information "
        "that must NOT be exported outside of IAI.\n\n"
        "Key guidelines:\n"
        "1. Use SEMANTIC REASONING, not keyword matching. A term appearing in a "
        "   casual or public context does NOT make a document classified.\n"
        "2. Examples:\n"
        "   - 'The pilot adjusted the Arrow on the display.' → Casual → Approved\n"
        "   - 'Internal telemetry from the Arrow-3 interceptor test.' → Classified → Blocked\n"
        "3. Consider Hebrew and English equally. You may encounter Hebrew text.\n"
        "4. You MUST respond with valid JSON matching this exact schema:\n"
        '   {"decision": "Approved"|"Blocked", "reasoning": "...", '
        '   "confidence_score": 0.0-1.0, "matched_terms": ["term1", ...]}\n\n'
    )

    if injected_contexts:
        base += (
            "CLASSIFIED TERM CONTEXT (retrieved from the IAI knowledge base):\n"
            "The following terms were detected. Use this context to understand "
            "their classified nature:\n\n"
        )
        for term, ctx in injected_contexts:
            base += f"  • [{term}]: {ctx}\n"
        base += "\n"

    base += (
        "Analyze the segment below and return ONLY the JSON response. "
        "Do not include any text outside the JSON object."
    )
    return base


# ---------------------------------------------------------------------------
# LLM client — routes to OpenAI or Anthropic based on config.LLM_PROVIDER
# ---------------------------------------------------------------------------

def _call_openai(system_prompt: str, user_message: str) -> str:
    """Call an OpenAI-compatible endpoint."""
    from openai import OpenAI
    kwargs: dict = {"api_key": config.LLM_API_KEY}
    if config.LLM_BASE_URL:
        kwargs["base_url"] = config.LLM_BASE_URL
    client = OpenAI(**kwargs)
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
        max_tokens=512,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or "{}"


def _call_anthropic(system_prompt: str, user_message: str) -> str:
    """Call Anthropic Claude."""
    import anthropic as _anthropic
    client = _anthropic.Anthropic(api_key=config.LLM_API_KEY)
    msg = client.messages.create(
        model=config.LLM_MODEL,
        max_tokens=512,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
        temperature=0.1,
    )
    return msg.content[0].text if msg.content else "{}"


def _call_llm(system_prompt: str, user_message: str) -> str:
    """Route the LLM call based on the configured provider."""
    if config.LLM_PROVIDER == "anthropic":
        return _call_anthropic(system_prompt, user_message)
    return _call_openai(system_prompt, user_message)


def analyze_segment(
    segment: str,
    injected_contexts: list[tuple[str, str]],
    candidate_terms: list[str],
) -> DLPDecision:
    """
    Send a document segment to the LLM and return a validated DLPDecision.

    Automatically routes to Anthropic or OpenAI-compatible backend based on
    the model name in config.py.

    Args:
        segment:           The text to analyse.
        injected_contexts: (term, description) pairs from the DB.
        candidate_terms:   Terms flagged by vector search (for prompt context).

    Returns:
        DLPDecision with structured output.
    """
    system_prompt = build_system_prompt(injected_contexts)

    user_message = (
        f"Analyze the following document segment:\n\n"
        f"<segment>\n{segment}\n</segment>\n\n"
    )
    if candidate_terms:
        user_message += (
            f"Potentially sensitive terms detected: {', '.join(candidate_terms)}\n"
        )
    user_message += "Return only valid JSON matching the required schema."

    try:
        raw_json = _call_llm(system_prompt, user_message)

        # Strip any markdown fences Claude may add
        raw_json = raw_json.strip()
        if raw_json.startswith("```"):
            raw_json = raw_json.split("```")[1]
            if raw_json.startswith("json"):
                raw_json = raw_json[4:]
            raw_json = raw_json.strip()

        data = json.loads(raw_json)
        return DLPDecision(**data)
    except Exception as exc:
        # Safe-fail: default to Blocked to prevent accidental data leak
        return DLPDecision(
            decision="Blocked",
            reasoning=f"LLM analysis failed ({exc}). Defaulting to Blocked for safety.",
            confidence_score=0.0,
            matched_terms=candidate_terms,
        )


def analyze_full_document(
    segments_with_context: list[tuple[str, list[tuple[str, str]], list[str]]],
) -> DLPDecision:
    """
    Analyse all segments in parallel and aggregate into a single decision.

    Strategy: ANY 'Blocked' segment blocks the whole document.
    Final confidence = mean of all segment confidences.

    Parallelism is bounded by config.MAX_LLM_CONCURRENCY to avoid
    overloading the IAI on-prem LLM server.  Each segment's LLM call is
    independent (no shared state), so concurrent execution is safe.

    Args:
        segments_with_context: List of (segment, injected_contexts, candidate_terms).

    Returns:
        Aggregated DLPDecision for the full document.
    """
    if not segments_with_context:
        return DLPDecision(
            decision="Approved",
            reasoning="No text content found in document.",
            confidence_score=1.0,
        )

    n = len(segments_with_context)
    decisions: list[DLPDecision | None] = [None] * n

    with ThreadPoolExecutor(max_workers=min(n, config.MAX_LLM_CONCURRENCY)) as pool:
        future_to_idx = {
            pool.submit(analyze_segment, seg, ctx, cand): i
            for i, (seg, ctx, cand) in enumerate(segments_with_context)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                decisions[idx] = future.result()
            except Exception as exc:
                # Safe-fail: treat any unhandled exception as Blocked.
                decisions[idx] = DLPDecision(
                    decision="Blocked",
                    reasoning=(
                        f"Segment analysis raised an unexpected error ({exc}). "
                        "Defaulting to Blocked for safety."
                    ),
                    confidence_score=0.0,
                    matched_terms=[],
                )

    # All slots must be filled — guard against any future.result() gap.
    for i, d in enumerate(decisions):
        if d is None:
            decisions[i] = DLPDecision(
                decision="Blocked",
                reasoning="Segment result missing. Defaulting to Blocked for safety.",
                confidence_score=0.0,
                matched_terms=[],
            )

    blocked = [d for d in decisions if d.decision == "Blocked"]  # type: ignore[union-attr]
    all_matched = list({t for d in decisions for t in d.matched_terms})  # type: ignore[union-attr]
    avg_confidence = sum(d.confidence_score for d in decisions) / n  # type: ignore[union-attr]

    if blocked:
        combined_reasoning = " | ".join(
            f"[Seg {i + 1}]: {d.reasoning}"
            for i, d in enumerate(decisions)
            if d and d.decision == "Blocked"
        )
        return DLPDecision(
            decision="Blocked",
            reasoning=combined_reasoning,
            confidence_score=avg_confidence,
            matched_terms=all_matched,
        )

    first = next((d for d in decisions if d), None)
    return DLPDecision(
        decision="Approved",
        reasoning=first.reasoning if first else "Document is clean.",
        confidence_score=avg_confidence,
        matched_terms=[],
    )
