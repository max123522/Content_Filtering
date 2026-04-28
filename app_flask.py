"""
app_flask.py — IAI Semantic DLP — High-End Flask Web Server.

Routes:
    GET  /               — Main upload & scan UI
    POST /scan           — Document upload endpoint → JSON response
    GET  /history        — Audit log viewer
    GET  /admin          — Admin panel (forbidden terms CRUD)
    POST /admin/term     — Add or update a forbidden term
    POST /admin/term/<id>/delete  — Delete a term
    POST /feedback/<log_id>       — Submit like/dislike on a scan result
    GET  /api/stats      — Dashboard statistics (JSON)
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)

import config
from data.db_handler import (
    init_db,
    get_all_terms,
    get_all_logs,
    upsert_term,
    update_feedback,
    get_session,
    ForbiddenTerm,
    AnalysisLog,
)
from core.graph_manager import invalidate_terms_cache

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = config.FLASK_SECRET_KEY

# Ensure DB exists on startup
init_db()

# ---------------------------------------------------------------------------
# Global concurrency guard
# ---------------------------------------------------------------------------
# Limits the total number of simultaneous document analyses across ALL users.
# Prevents overloading the IAI on-prem LLM / embedding servers.
# If all slots are taken, the request waits up to 5 minutes before returning
# a "Server busy" error.
_analysis_semaphore = threading.Semaphore(config.MAX_TOTAL_ANALYSES)


# ---------------------------------------------------------------------------
# Main scan page
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Render the main document upload & scan page."""
    return render_template("index.html")


@app.route("/scan", methods=["POST"])
def scan():
    """
    Accept 1–MAX_PARALLEL_DOCS documents and run the DLP pipeline.

    Single-file request  (field name: ``document``)
        → returns a flat JSON object (backwards compatible with existing UI).

    Multi-file request   (field name: ``documents``, up to MAX_PARALLEL_DOCS)
        → returns a JSON array, one result object per file, in upload order.

    Each analysis acquires a slot from _analysis_semaphore so that the total
    number of concurrent analyses across all users never exceeds
    MAX_TOTAL_ANALYSES.  Files in a multi-file request are analysed in
    parallel (bounded by the semaphore).

    File bytes are read from Werkzeug streams before spawning threads —
    Werkzeug request objects are not thread-safe.
    """
    from core.graph_manager import run_analysis

    # ── Collect uploaded files ────────────────────────────────────────
    raw_files = request.files.getlist("documents")
    if not raw_files:
        single = request.files.get("document")
        if single:
            raw_files = [single]

    files = [f for f in raw_files if f and f.filename]
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    if len(files) > config.MAX_PARALLEL_DOCS:
        return jsonify(
            {"error": f"Maximum {config.MAX_PARALLEL_DOCS} documents per request"}
        ), 400

    # Read all bytes NOW — Werkzeug streams must not be accessed from threads.
    file_data: list[tuple[str, bytes]] = [
        (f.filename, f.read()) for f in files
    ]

    # ── Worker function (runs inside a thread) ────────────────────────
    def _analyse(filename: str, file_bytes: bytes) -> dict:
        """Acquire a semaphore slot, run analysis, release slot."""
        acquired = _analysis_semaphore.acquire(timeout=300)  # 5-min timeout
        if not acquired:
            return {
                "filename": filename,
                "error": "Server busy — too many concurrent analyses. Please retry.",
            }
        try:
            state = run_analysis(file_bytes=file_bytes, filename=filename)
            decision = state.final_decision
            return {
                "filename": filename,
                "decision": decision.decision,
                "reasoning": decision.reasoning,
                "confidence_score": round(decision.confidence_score, 3),
                "matched_terms": decision.matched_terms,
                "log_id": state.log_id,
            }
        except Exception as exc:
            return {"filename": filename, "error": str(exc)}
        finally:
            _analysis_semaphore.release()

    # ── Single file — return flat dict (UI backwards compatible) ──────
    if len(file_data) == 1:
        return jsonify(_analyse(*file_data[0]))

    # ── Multiple files — process in parallel, preserve upload order ───
    results: list[dict | None] = [None] * len(file_data)
    with ThreadPoolExecutor(max_workers=len(file_data)) as pool:
        future_map = {
            pool.submit(_analyse, fname, fbytes): i
            for i, (fname, fbytes) in enumerate(file_data)
        }
        for future in as_completed(future_map):
            results[future_map[future]] = future.result()

    return jsonify(results)


# ---------------------------------------------------------------------------
# History / Audit Log
# ---------------------------------------------------------------------------

@app.route("/history")
def history():
    """Render the audit log page with all past scans."""
    logs = get_all_logs(limit=500)
    return render_template("history.html", logs=logs)


# ---------------------------------------------------------------------------
# Admin Panel
# ---------------------------------------------------------------------------

@app.route("/admin")
def admin():
    """Render the admin panel with all forbidden terms."""
    terms = get_all_terms()
    return render_template("admin.html", terms=terms)


@app.route("/admin/term", methods=["POST"])
def admin_add_term():
    """Add or update a forbidden term via the admin panel."""
    term_text = (request.form.get("term") or "").strip()
    context = (request.form.get("context_description") or "").strip()

    if not term_text:
        return jsonify({"error": "Term cannot be empty"}), 400

    embedding_json: str | None = None
    embedded = False
    try:
        from models.embedding_service import EmbeddingService
        import json as _json
        emb = EmbeddingService()
        vec = emb.encode(term_text, is_query=False)
        embedding_json = _json.dumps(vec.tolist())
        embedded = True
    except Exception as exc:
        print(f"[admin_add_term] Embedding failed: {exc}")

    upsert_term(
        term=term_text,
        context_description=context or None,
        embedding_json=embedding_json,
    )
    invalidate_terms_cache()

    return jsonify({"term": term_text, "embedded": embedded})


@app.route("/admin/term/<int:term_id>/delete", methods=["POST"])
def admin_delete_term(term_id: int):
    """Delete a forbidden term by ID."""
    session = get_session()
    try:
        term = session.get(ForbiddenTerm, term_id)
        if term:
            session.delete(term)
            session.commit()
    finally:
        session.close()
    invalidate_terms_cache()
    return redirect(url_for("admin"))


@app.route("/admin/term/<int:term_id>/edit", methods=["GET", "POST"])
def admin_edit_term(term_id: int):
    """GET: show edit form. POST: save and re-embed."""
    session = get_session()
    try:
        term = session.get(ForbiddenTerm, term_id)
        if not term:
            return redirect(url_for("admin"))

        if request.method == "GET":
            return render_template("edit_term.html", term=term)

        # POST — save
        context = (request.form.get("context_description") or "").strip()
        term.context_description = context or None
        try:
            from models.embedding_service import EmbeddingService
            import json as _json
            emb = EmbeddingService()
            vec = emb.encode(term.term, is_query=False)
            term.embedding_json = _json.dumps(vec.tolist())
        except Exception as exc:
            print(f"[admin_edit_term] Re-embedding failed: {exc}")
        session.commit()
    finally:
        session.close()
    invalidate_terms_cache()
    return redirect(url_for("admin"))


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

@app.route("/feedback/<int:log_id>", methods=["POST"])
def feedback(log_id: int):
    """Record user feedback ('like' or 'dislike') on a scan result."""
    value = request.json.get("feedback") if request.is_json else request.form.get("feedback")
    if value not in ("like", "dislike"):
        return jsonify({"error": "Invalid feedback value"}), 400
    update_feedback(log_id, value)
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Stats API
# ---------------------------------------------------------------------------

@app.route("/api/stats")
def api_stats():
    """Return dashboard statistics as JSON."""
    logs = get_all_logs(limit=1000)
    total = len(logs)
    blocked = sum(1 for l in logs if l.decision == "Blocked")
    approved = total - blocked
    terms_count = len(get_all_terms())
    return jsonify(
        {
            "total_scans": total,
            "blocked": blocked,
            "approved": approved,
            "terms_count": terms_count,
            "block_rate": round(blocked / total * 100, 1) if total else 0,
        }
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG,
        threaded=True,   # Handle each HTTP request in its own thread.
    )
