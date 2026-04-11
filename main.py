"""
main.py — IAI Semantic DLP — CLI entry point.

Usage:
    python main.py --file path/to/document.pdf
    python main.py --file document.docx --json
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

# Force UTF-8 output so Hebrew and emoji print correctly on Windows.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def main() -> None:
    """Run the DLP analysis pipeline on a single document from the CLI."""
    parser = argparse.ArgumentParser(
        description="IAI Semantic DLP — Analyse a document for classified content."
    )
    parser.add_argument(
        "--file",
        required=True,
        metavar="PATH",
        help="Path to the document to analyse (.pdf, .docx).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of a human-readable summary.",
    )
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"[ERROR] File not found: {path}", file=sys.stderr)
        sys.exit(1)

    file_bytes = path.read_bytes()
    filename = path.name

    print(f"[IAI DLP] Analysing '{filename}' …")

    from core.graph_manager import run_analysis
    state = run_analysis(file_bytes=file_bytes, filename=filename)
    decision = state.final_decision

    if args.json:
        print(
            json.dumps(
                {
                    "decision": decision.decision,
                    "reasoning": decision.reasoning,
                    "confidence_score": decision.confidence_score,
                    "matched_terms": decision.matched_terms,
                    "log_id": state.log_id,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print()
        icon = "✅" if decision.decision == "Approved" else "🚨"
        print(f"  {icon}  DECISION:    {decision.decision}")
        print(f"  📊  CONFIDENCE:  {decision.confidence_score:.1%}")
        if decision.matched_terms:
            print(f"  ⚠️   TERMS:       {', '.join(decision.matched_terms)}")
        print(f"\n  💬  REASONING:")
        print(f"  {decision.reasoning}")
        print(f"\n  🗄️   LOG ID:      {state.log_id}")


if __name__ == "__main__":
    main()
