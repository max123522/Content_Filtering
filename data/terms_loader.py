"""
terms_loader.py — Parse words_content_filtering.docx and sync to DB.

Rules:
  * Terms are delimited by  /  within paragraphs.
  * Upsert logic: add new terms, refresh embeddings on existing ones.
  * NEVER overwrite manually-enriched context_description values.

Usage:
    python data/terms_loader.py
    python data/terms_loader.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional
import docx

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from data.db_handler import init_db, upsert_term, get_session

# Default context descriptions for known term categories
_DEFAULT_CONTEXTS: dict[str, str] = {
    # Weapon systems
    "Arrow": (
        "Arrow is a family of Israeli anti-ballistic missile systems developed by "
        "IAI and Rafael. References in technical documents may indicate classified "
        "interceptor design, trajectory data, or test results."
    ),
    "חץ": (
        "חץ הוא שם משפחה של מערכות נגד-טילים בליסטיים ישראליות שפותחו על ידי "
        "תע\"א ורפאל (Arrow-2, Arrow-3). אזכור במסמכים טכניים עשוי להצביע על "
        "מידע מסווג על עיצוב המיירט, נתוני מסלול או תוצאות ניסויים."
    ),
    "Lampad": (
        "Lampad (למפאד) is a classified IAI guided missile program. Any reference "
        "in external documents is a potential security breach."
    ),
    "Lrad": (
        "LRAD (Long Range Acoustic Device) or LRAD missile system — any mention "
        "in the context of IAI projects may indicate restricted development data."
    ),
    "Iron Dome": (
        "Iron Dome is the short-range air defence system. Internal telemetry, "
        "battery positions, or engagement logs are classified."
    ),
    "David's Sling": (
        "David's Sling (Magic Wand) is a medium-to-long range defence system. "
        "Technical parameters and operational data are classified."
    ),
    "Barak 8": (
        "Barak 8 is a long-range surface-to-air missile system developed jointly "
        "by IAI and DRDO. Specifications and test data are classified."
    ),
    "Barak MX": (
        "Barak MX is an advanced naval/land air-defence system by IAI. "
        "Configuration and engagement envelopes are classified."
    ),
    "Gabriel": (
        "Gabriel is a sea-skimming anti-ship missile by IAI. Seeker algorithms, "
        "propulsion parameters, and guidance data are classified."
    ),
    "Popeye": (
        "Popeye (Have Nap/Have Lite) is an air-to-surface standoff missile. "
        "Guidance and warhead details are classified."
    ),
    "Spice": (
        "SPICE (Smart Precise Impact Cost-Effective) is an IAI precision guidance "
        "kit. CEP data, seeker parameters, and integration specs are classified."
    ),
    "Python 5": (
        "Python 5 is an advanced WVRAAM developed by Rafael/IAI. Seeker and "
        "kinematic performance data are classified."
    ),
    "Derby": (
        "Derby (Python 4 BVR) is an active-radar BVRAAM. Range, radar seeker, "
        "and ECM resistance data are classified."
    ),
    "Heron TP": (
        "Heron TP (Eitan) is an MALE UAS by IAI. Sensor suites, mission systems, "
        "and communication frequencies are classified."
    ),
    "Eitan": (
        "Eitan is the Hebrew name for the Heron TP MALE drone. See Heron TP."
    ),
    "Hermes 900": (
        "Hermes 900 is an ISTAR UAS by Elbit. Payload and communication parameters "
        "are classified."
    ),
    "Harop": (
        "Harop is an IAI loitering munition (kamikaze drone). Seeker, navigation, "
        "and targeting data are classified."
    ),
    "Harpy": (
        "Harpy is an IAI anti-radiation autonomous weapon. RF homing and "
        "engagement parameters are classified."
    ),
    "LORA": (
        "LORA (Long Range Artillery) is a ballistic surface-to-surface missile "
        "by IAI. Range, CEP, and warhead data are classified."
    ),
    "Blue Spear": (
        "Blue Spear is a 5G sea-based cruise missile by IAI/ST Engineering. "
        "Seeker and navigation data are classified."
    ),
    "WindGuard": (
        "WindGuard is an active protection system component. Details about "
        "intercept logic and radar parameters are classified."
    ),
    # Technical components
    "Telemetry": (
        "Telemetry data from weapon systems tests contains classified flight "
        "performance and system health information."
    ),
    "Interceptor": (
        "Any reference to interceptor performance, kinematics, or test data "
        "for IAI systems is classified."
    ),
    "ELM-2084": (
        "ELM-2084 is an ELTA multi-mission radar. Waveform, frequency, and "
        "detection parameters are classified."
    ),
    "Encryption Key": (
        "Encryption keys for any defence communication system are classified "
        "and must never appear in external documents."
    ),
    "C4I": (
        "C4I (Command, Control, Communications, Computers, and Intelligence) "
        "architecture and protocols for IAI systems are classified."
    ),
    # Organisations / threat actors
    "IAI": (
        "Israel Aerospace Industries — internal project names, organisational "
        "charts, and unreleased product information are classified."
    ),
    "Hizballah": (
        "Hezbollah / Hizballah — any document linking IAI systems to this "
        "organisation's intelligence or targeting is classified."
    ),
    "Hamas": (
        "Hamas — references in the context of IAI operational data, targeting, "
        "or intelligence assessments are classified."
    ),
    "IRGC": (
        "Islamic Revolutionary Guard Corps — references in the context of IAI "
        "threat assessments or operations are classified."
    ),
    "Top Secret": (
        "Explicit classification marking. Document must be blocked."
    ),
    "Classified": (
        "Explicit classification marking. Document must be blocked."
    ),
    "Confidential": (
        "Explicit classification marking. Document must be blocked."
    ),
    "Restricted": (
        "Explicit classification marking. Document must be blocked."
    ),
}


def _extract_terms_from_docx(docx_path: Path) -> list[str]:
    """
    Parse a .docx file and return all terms delimited by '/'.

    Terms are stripped of whitespace and empty values are discarded.
    """
    doc = docx.Document(str(docx_path))
    raw_terms: list[str] = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        # Skip pure Hebrew heading lines (no slash separator)
        if "/" not in text:
            continue
        parts = [t.strip() for t in text.split("/")]
        raw_terms.extend(t for t in parts if t)
    return raw_terms


def _deduplicate(terms: list[str]) -> list[str]:
    """Return unique terms preserving first occurrence, case-insensitive dedup."""
    seen: set[str] = set()
    unique: list[str] = []
    for t in terms:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            unique.append(t)
    return unique


def run_ingestion(
    docx_path: Path = config.TERMS_DOCX_PATH,
    dry_run: bool = False,
    embed: bool = True,
) -> dict[str, int]:
    """
    Main ingestion routine.

    Returns:
        dict with keys 'inserted', 'updated', 'skipped'.
    """
    # Lazy import to avoid loading heavy models on every import
    embedding_service = None
    if embed:
        try:
            from models.embedding_service import EmbeddingService
            embedding_service = EmbeddingService()
        except Exception as exc:
            print(f"[terms_loader] Embedding service unavailable: {exc}")
            embedding_service = None

    terms = _deduplicate(_extract_terms_from_docx(docx_path))
    print(f"[terms_loader] Extracted {len(terms)} unique terms from {docx_path.name}")

    stats = {"inserted": 0, "updated": 0, "skipped": 0}
    session = get_session()
    try:
        for term in terms:
            # Compute embedding
            embedding_json: Optional[str] = None
            if embedding_service is not None:
                try:
                    vec = embedding_service.encode(term, is_query=False)
                    import json as _json
                    embedding_json = _json.dumps(vec.tolist())
                except Exception as exc:
                    print(f"[terms_loader] Could not embed '{term}': {exc}")

            # Determine if term already exists
            from data.db_handler import ForbiddenTerm
            existing = session.query(ForbiddenTerm).filter_by(term=term).first()
            context = _DEFAULT_CONTEXTS.get(term)

            if dry_run:
                action = "UPDATE" if existing else "INSERT"
                print(f"  [{action}] {term!r}")
                if not existing:
                    stats["inserted"] += 1
                else:
                    stats["updated"] += 1
                continue

            result = upsert_term(
                term=term,
                context_description=context,
                embedding_json=embedding_json,
                session=session,
            )
            session.flush()
            if existing:
                stats["updated"] += 1
            else:
                stats["inserted"] += 1

        if not dry_run:
            session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    print(
        f"[terms_loader] Done — "
        f"inserted={stats['inserted']}, updated={stats['updated']}, "
        f"skipped={stats['skipped']}"
    )
    return stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IAI DLP — Terms ingestion")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without writing to the DB.",
    )
    parser.add_argument(
        "--no-embed",
        action="store_true",
        help="Skip computing embeddings (faster, for testing).",
    )
    args = parser.parse_args()

    # Ensure DB tables exist
    init_db()
    run_ingestion(dry_run=args.dry_run, embed=not args.no_embed)
