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

import os
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

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = config.FLASK_SECRET_KEY

# Ensure DB exists on startup
init_db()


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
    Receive an uploaded document, run the DLP pipeline, return JSON.

    Expected multipart field: `document` (file upload).
    """
    if "document" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    uploaded = request.files["document"]
    if not uploaded.filename:
        return jsonify({"error": "Empty filename"}), 400

    filename = uploaded.filename
    file_bytes = uploaded.read()

    # Lazy import — avoid loading heavy models at app startup
    from core.graph_manager import run_analysis

    try:
        state = run_analysis(file_bytes=file_bytes, filename=filename)
        decision = state.final_decision
        return jsonify(
            {
                "decision": decision.decision,
                "reasoning": decision.reasoning,
                "confidence_score": round(decision.confidence_score, 3),
                "matched_terms": decision.matched_terms,
                "log_id": state.log_id,
                "filename": filename,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


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
        vec = emb.encode(term_text)
        embedding_json = _json.dumps(vec.tolist())
        embedded = True
    except Exception as exc:
        print(f"[admin_add_term] Embedding failed: {exc}")

    upsert_term(
        term=term_text,
        context_description=context or None,
        embedding_json=embedding_json,
    )

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
            vec = emb.encode(term.term)
            term.embedding_json = _json.dumps(vec.tolist())
        except Exception as exc:
            print(f"[admin_edit_term] Re-embedding failed: {exc}")
        session.commit()
    finally:
        session.close()
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
    )
