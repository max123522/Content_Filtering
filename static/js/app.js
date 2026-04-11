/**
 * app.js — IAI Semantic DLP — Frontend Logic
 * Handles drag-and-drop upload, AJAX scan, result rendering, and feedback.
 */

'use strict';

/* ── Utility ────────────────────────────────────────────────────────────── */
const $ = id => document.getElementById(id);
const show = el => el && el.classList.add('show');
const hide = el => el && el.classList.remove('show');

function showToast(message, type = 'info') {
  const container = document.querySelector('.toast-container') || (() => {
    const c = document.createElement('div');
    c.className = 'toast-container';
    document.body.appendChild(c);
    return c;
  })();

  const icons = { success: '✅', error: '🚨', info: 'ℹ️', warning: '⚠️' };
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span>${icons[type] || icons.info}</span><span>${message}</span>`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.animation = 'none';
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(30px)';
    toast.style.transition = 'all 0.3s';
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

/* ── Scan step indicators ───────────────────────────────────────────────── */
const STEPS = ['Parsing', 'Embedding', 'Reranking', 'Reasoning', 'Complete'];

function animateScanSteps() {
  const container = $('scanSteps');
  if (!container) return;

  container.innerHTML = STEPS.map((s, i) =>
    `<div class="scan-step" id="step-${i}">
       <span class="step-dot">○</span> ${s}
     </div>`
  ).join('');

  let current = 0;
  return new Promise(resolve => {
    const interval = setInterval(() => {
      if (current > 0) {
        const prev = $(`step-${current - 1}`);
        if (prev) { prev.classList.remove('active'); prev.classList.add('done'); prev.querySelector('.step-dot').textContent = '✓'; }
      }
      if (current < STEPS.length) {
        const cur = $(`step-${current}`);
        if (cur) { cur.classList.add('active'); cur.querySelector('.step-dot').textContent = '⟳'; }
        current++;
      } else {
        clearInterval(interval);
        resolve();
      }
    }, 700);
  });
}

/* ── Confidence bar ─────────────────────────────────────────────────────── */
function renderConfidenceBar(score) {
  const pct = Math.round(score * 100);
  let cls = 'high';
  if (pct < 40) cls = 'low';
  else if (pct < 70) cls = 'medium';
  return `
    <div class="confidence-bar-container">
      <div class="confidence-bar">
        <div class="confidence-fill ${cls}" style="width:0%" data-target="${pct}"></div>
      </div>
      <span style="font-family:var(--font-display);font-size:.9rem;font-weight:600;color:var(--iai-silver-100);min-width:38px">${pct}%</span>
    </div>`;
}

function animateBars() {
  document.querySelectorAll('.confidence-fill[data-target]').forEach(el => {
    const target = el.getAttribute('data-target');
    requestAnimationFrame(() => setTimeout(() => { el.style.width = target + '%'; }, 50));
  });
}

/* ── Result renderer ────────────────────────────────────────────────────── */
function renderResult(data) {
  const panel   = $('resultPanel');
  const header  = $('resultHeader');
  const icon    = $('resultIcon');
  const title   = $('resultTitle');
  const body    = $('resultBody');

  if (!panel) return;

  const isBlocked = data.decision === 'Blocked';
  const statusCls = isBlocked ? 'blocked' : 'approved';

  header.className = `result-header glass-card ${statusCls}`;
  icon.textContent  = isBlocked ? '🚨' : '✅';
  title.textContent = data.decision.toUpperCase();
  title.className   = `result-title ${statusCls}`;

  const termsHtml = (data.matched_terms || []).length
    ? `<div style="margin-top:var(--space-md)">
         <div class="section-label">Matched Terms</div>
         <div class="terms-list">${data.matched_terms.map(t => `<span class="term-chip">${t}</span>`).join('')}</div>
       </div>`
    : '';

  body.innerHTML = `
    <div class="result-meta">
      <div class="meta-item">
        <span class="meta-label">File</span>
        <span class="meta-value">${data.filename || '—'}</span>
      </div>
      <div class="meta-item">
        <span class="meta-label">Confidence</span>
        <span class="meta-value">${Math.round((data.confidence_score || 0) * 100)}%</span>
      </div>
      <div class="meta-item">
        <span class="meta-label">Log ID</span>
        <span class="meta-value">#${data.log_id || '—'}</span>
      </div>
    </div>

    ${renderConfidenceBar(data.confidence_score || 0)}

    <div style="margin-top:var(--space-md)">
      <div class="section-label">Semantic Reasoning</div>
      <div class="reasoning-box">${data.reasoning || 'No reasoning provided.'}</div>
    </div>

    ${termsHtml}

    <div class="feedback-row" id="feedbackRow">
      <span class="feedback-label">Was this accurate?</span>
      <button class="btn-feedback" id="btnLike" onclick="submitFeedback(${data.log_id}, 'like')">
        👍 Yes
      </button>
      <button class="btn-feedback" id="btnDislike" onclick="submitFeedback(${data.log_id}, 'dislike')">
        👎 No
      </button>
    </div>
  `;

  show(panel);
  animateBars();
  panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/* ── Feedback ───────────────────────────────────────────────────────────── */
async function submitFeedback(logId, value) {
  if (!logId) return;
  try {
    await fetch(`/feedback/${logId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ feedback: value }),
    });
    const btnLike    = $('btnLike');
    const btnDislike = $('btnDislike');
    if (value === 'like') {
      btnLike?.classList.add('active-like');
      btnDislike?.classList.remove('active-dislike');
    } else {
      btnDislike?.classList.add('active-dislike');
      btnLike?.classList.remove('active-like');
    }
    showToast('Feedback recorded. Thank you!', 'success');
  } catch (e) {
    showToast('Failed to submit feedback.', 'error');
  }
}

/* ── File upload & scan ─────────────────────────────────────────────────── */
async function uploadAndScan(file) {
  const spinner = $('spinnerOverlay');
  const panel   = $('resultPanel');
  if (panel) hide(panel);

  show(spinner);

  // Animate steps in parallel
  animateScanSteps();

  const formData = new FormData();
  formData.append('document', file);

  try {
    const res  = await fetch('/scan', { method: 'POST', body: formData });
    const data = await res.json();

    hide(spinner);

    if (data.error) {
      showToast(`Error: ${data.error}`, 'error');
      return;
    }

    renderResult(data);
    loadStats();
  } catch (err) {
    hide(spinner);
    showToast(`Network error: ${err.message}`, 'error');
  }
}

/* ── Drag and Drop ──────────────────────────────────────────────────────── */
function initDropZone() {
  const zone  = $('uploadZone');
  const input = $('fileInput');
  if (!zone || !input) return;

  zone.addEventListener('click', () => input.click());

  input.addEventListener('change', e => {
    const file = e.target.files[0];
    if (file) uploadAndScan(file);
  });

  zone.addEventListener('dragover', e => {
    e.preventDefault();
    zone.classList.add('drag-over');
  });

  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));

  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) uploadAndScan(file);
  });
}

/* ── Stats dashboard ────────────────────────────────────────────────────── */
async function loadStats() {
  try {
    const res  = await fetch('/api/stats');
    const data = await res.json();
    const set  = (id, val) => { const el = $(id); if (el) el.textContent = val; };
    set('statTotal',     data.total_scans ?? '—');
    set('statBlocked',   data.blocked     ?? '—');
    set('statApproved',  data.approved    ?? '—');
    set('statTerms',     data.terms_count ?? '—');
    set('statBlockRate', data.block_rate != null ? data.block_rate + '%' : '—');
  } catch (_) { /* silent fail */ }
}

/* ── Admin: inline edit context (AJAX) ──────────────────────────────────── */
function editContext(btn) {
  const termId     = btn.dataset.termId;
  const currentCtx = btn.dataset.ctx || '';
  const cell       = $(`ctx-${termId}`);
  if (!cell) return;

  const existing = cell.querySelector('.edit-form');
  if (existing) { existing.remove(); return; }

  const wrapper = document.createElement('div');
  wrapper.className = 'edit-form';
  wrapper.style.cssText = 'margin-top:8px;display:flex;gap:8px;align-items:center;flex-wrap:wrap';

  const input = document.createElement('textarea');
  input.rows = 2;
  input.value = currentCtx;
  input.className = 'form-input';
  input.style.cssText = 'font-size:.8rem;padding:4px 8px;flex:1;min-width:200px;resize:vertical';
  input.placeholder = 'Enter context description…';

  const save = document.createElement('button');
  save.type = 'button';
  save.className = 'btn-iai';
  save.style.cssText = 'padding:6px 14px;font-size:.8rem';
  save.textContent = 'Save';

  const cancel = document.createElement('button');
  cancel.type = 'button';
  cancel.className = 'btn-secondary';
  cancel.style.cssText = 'padding:6px 10px;font-size:.8rem';
  cancel.textContent = 'Cancel';
  cancel.onclick = () => wrapper.remove();

  save.onclick = () => {
    save.textContent = 'Saving…';
    save.disabled = true;
    fetch('/admin/term/' + termId + '/edit', {
      method: 'POST',
      headers: {'Content-Type': 'application/x-www-form-urlencoded'},
      body: 'context_description=' + encodeURIComponent(input.value.trim())
    })
    .then(r => r.json())
    .then(data => {
      const textEl = cell.querySelector('.ctx-text');
      if (data.context_description) {
        textEl.textContent = data.context_description;
      } else {
        textEl.innerHTML = '<em style="color:var(--iai-silver-500)">No description</em>';
      }
      btn.dataset.ctx = data.context_description || '';
      wrapper.remove();
    })
    .catch(() => {
      save.textContent = 'Save';
      save.disabled = false;
      alert('Save failed, please try again.');
    });
  };

  wrapper.appendChild(input);
  wrapper.appendChild(save);
  wrapper.appendChild(cancel);
  cell.appendChild(wrapper);
  input.focus();
}

/* ── Init ───────────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  initDropZone();
  loadStats();

  // Highlight active nav link
  const path = window.location.pathname;
  document.querySelectorAll('.navbar-nav a').forEach(a => {
    if (a.getAttribute('href') === path) a.classList.add('active');
  });
});
