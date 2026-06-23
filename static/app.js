const form = document.getElementById("query-form");
const input = document.getElementById("query-input");
const submitBtn = document.getElementById("submit-btn");
const thread = document.getElementById("thread");
const emptyState = document.getElementById("empty-state");
const clearBtn = document.getElementById("clear-btn");
const webToggle = document.getElementById("web-toggle-input");

const sourceModal = document.getElementById("source-modal");
const sourceModalTag = document.getElementById("source-modal-tag");
const sourceModalBody = document.getElementById("source-modal-body");
const sourceModalClose = document.getElementById("source-modal-close");
const sourceModalBackdrop = document.querySelector(".source-modal-backdrop");

document.querySelectorAll(".example-card").forEach((card) => {
  card.addEventListener("click", () => {
    input.value = card.dataset.example;
    input.focus();
  });
});

clearBtn.addEventListener("click", () => {
  thread.innerHTML = "";
  emptyState.style.display = "";
});

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function renderAnswerWithCitations(text, sources) {
  const escaped = escapeHtml(text);
  return escaped.replace(/\[Source (\d+)\]/g, (match, n) => {
    const idx = parseInt(n, 10) - 1;
    if (!sources || !sources[idx]) return match;
    return `<button type="button" class="cite" data-cite="${idx}">${n}</button>`;
  });
}

function verdictInfo(verdict) {
  if (verdict === "fully_supported") return { cls: "supported", cardCls: "fully-supported", label: "✓ Fully supported" };
  if (verdict === "partially_supported") return { cls: "partial", cardCls: "partially-supported", label: "◐ Partially supported" };
  if (verdict === "not_supported") return { cls: "unsupported", cardCls: "not-supported", label: "✕ Not supported" };
  return { cls: "retry", cardCls: "", label: "Unverified" };
}

function scorePercent(score) {
  // rerank scores from a cross-encoder are roughly in [-10, 10]; clamp+normalize for a visual bar
  const clamped = Math.max(-2, Math.min(10, score));
  return Math.round(((clamped + 2) / 12) * 100);
}

function buildTurn(query, data) {
  const turn = document.createElement("div");
  turn.className = "turn";

  const qEl = document.createElement("div");
  qEl.className = "turn-question";
  qEl.textContent = query;
  turn.appendChild(qEl);

  const card = document.createElement("div");
  card.className = "answer-card";

  if (data.blocked_reason) {
    card.classList.add("blocked");
    card.innerHTML = `<div class="blocked-text">⚠ Blocked — ${escapeHtml(data.blocked_reason)}</div>`;
    turn.appendChild(card);
    return turn;
  }

  const v = verdictInfo(data.support && data.support.verdict);
  if (v.cardCls) card.classList.add(v.cardCls);

  const metaRow = document.createElement("div");
  metaRow.className = "answer-meta-row";
  metaRow.innerHTML = `
    <span class="verdict-badge ${v.cls}">${v.label}</span>
    ${data.retries > 0 ? `<span class="verdict-badge retry">${data.retries} retry${data.retries > 1 ? "ies" : ""}</span>` : ""}
    <span class="answer-time">${data.elapsed.toFixed(1)}s</span>
  `;
  card.appendChild(metaRow);

  const sources = data.sources || [];
  const textEl = document.createElement("div");
  textEl.className = "answer-text";
  textEl.innerHTML = renderAnswerWithCitations(data.answer || "", sources);
  card.appendChild(textEl);

  if (sources.length) {
    const row = document.createElement("div");
    row.className = "sources-row";
    sources.forEach((s, i) => {
      const pct = scorePercent(s.rerank_score);
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "source-card";
      btn.innerHTML = `
        <div class="source-card-doc">[${i + 1}] ${escapeHtml(s.source)}</div>
        <div class="source-card-page">page ${s.page}</div>
        <div class="source-card-bar-track"><div class="source-card-bar-fill" style="width:${pct}%"></div></div>
      `;
      btn.addEventListener("click", () => openSourceModal(s, i + 1));
      row.appendChild(btn);
    });
    card.appendChild(row);
  }

  textEl.querySelectorAll(".cite").forEach((btn) => {
    btn.addEventListener("click", () => {
      const idx = parseInt(btn.dataset.cite, 10);
      openSourceModal(sources[idx], idx + 1);
    });
  });

  turn.appendChild(card);
  return turn;
}

function openSourceModal(source, n) {
  if (!source) return;
  sourceModalTag.textContent = `[${n}] ${source.source} · page ${source.page}`;
  sourceModalBody.innerHTML = `
    <div class="source-modal-score">rerank score: ${source.rerank_score.toFixed(3)}</div>
    ${escapeHtml(source.text || "(full passage text not returned)")}
  `;
  sourceModal.classList.add("open");
  sourceModal.setAttribute("aria-hidden", "false");
}

function closeSourceModal() {
  sourceModal.classList.remove("open");
  sourceModal.setAttribute("aria-hidden", "true");
}

sourceModalClose.addEventListener("click", closeSourceModal);
sourceModalBackdrop.addEventListener("click", closeSourceModal);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeSourceModal();
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = input.value.trim();
  if (!query) return;

  emptyState.style.display = "none";
  input.value = "";
  input.disabled = true;
  submitBtn.disabled = true;

  const loadingTurn = document.createElement("div");
  loadingTurn.className = "turn";
  loadingTurn.innerHTML = `
    <div class="turn-question">${escapeHtml(query)}</div>
    <div class="answer-card">
      <div class="thinking-row">
        <div class="dot-flash"><span></span><span></span><span></span></div>
        Retrieving, grading relevance, generating…
      </div>
    </div>
  `;
  thread.appendChild(loadingTurn);
  loadingTurn.scrollIntoView({ behavior: "smooth", block: "end" });

  const start = performance.now();
  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    const data = await res.json();
    data.elapsed = (performance.now() - start) / 1000;

    const turnEl = buildTurn(query, data);
    thread.replaceChild(turnEl, loadingTurn);
    turnEl.scrollIntoView({ behavior: "smooth", block: "end" });
  } catch (err) {
    loadingTurn.querySelector(".thinking-row").innerHTML =
      `⚠ Request failed — ${escapeHtml(err.message)}`;
  } finally {
    input.disabled = false;
    submitBtn.disabled = false;
    input.focus();
  }
});

// After the existing sources row building code, add:
if (data.source_type === "web") {
  card.classList.add("web-sourced");

  const webBadge = document.createElement("span");
  webBadge.className = "web-badge";
  webBadge.textContent = "⚠ Web sourced — not from vetted guidelines";
  metaRow.insertBefore(webBadge, metaRow.firstChild);

  if (data.web_sources && data.web_sources.length) {
    const webRow = document.createElement("div");
    webRow.className = "web-sources-row";
    data.web_sources.forEach((s) => {
      const a = document.createElement("a");
      a.className = "web-source-link";
      a.href = s.url;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      a.innerHTML = `
        <span class="web-source-title">${escapeHtml(s.title)}</span>
        <span class="web-source-url">${escapeHtml(s.url)}</span>
        <span class="web-source-snippet">${escapeHtml(s.content)}</span>
      `;
      webRow.appendChild(a);
    });
    card.appendChild(webRow);
  }
}