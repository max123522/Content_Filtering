"""
agents.py — PydanticAI agents for semantic DLP reasoning.

Each agent wraps an LLM call and enforces a strict JSON output schema
via Pydantic models, preventing raw-string reasoning from leaking into
the pipeline.
"""

from __future__ import annotations

import json
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
    Analyse all segments and aggregate into a single document-level decision.

    Strategy: ANY 'Blocked' segment blocks the whole document.
    Final confidence = mean of all segment confidences.

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

    decisions: list[DLPDecision] = []
    for segment, contexts, candidates in segments_with_context:
        d = analyze_segment(segment, contexts, candidates)
        decisions.append(d)

    blocked = [d for d in decisions if d.decision == "Blocked"]
    all_matched = list({t for d in decisions for t in d.matched_terms})
    avg_confidence = sum(d.confidence_score for d in decisions) / len(decisions)

    if blocked:
        # Merge all blocked reasoning
        combined_reasoning = " | ".join(
            f"[Seg {i+1}]: {d.reasoning}"
            for i, d in enumerate(decisions)
            if d.decision == "Blocked"
        )
        return DLPDecision(
            decision="Blocked",
            reasoning=combined_reasoning,
            confidence_score=avg_confidence,
            matched_terms=all_matched,
        )
    else:
        return DLPDecision(
            decision="Approved",
            reasoning=decisions[0].reasoning if decisions else "Document is clean.",
            confidence_score=avg_confidence,
            matched_terms=[],
        )
