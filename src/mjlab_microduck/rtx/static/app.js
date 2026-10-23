/* RTX Control Center — frontend (vanilla JS, không framework) */

const $ = (id) => document.getElementById(id);
let MATRIX = null;

/* ------------------------------------------------------------- tiện ích */

async function api(path, opts) {
  const r = await fetch(path, opts);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw Object.assign(new Error(data.error || r.statusText), { data });
  return data;
}

function toast(msg, isErr = false) {
  const el = document.createElement("div");
  el.className = "toast-item" + (isErr ? " err" : "");
  el.textContent = msg;
  $("toast").appendChild(el);
  setTimeout(() => el.remove(), 6000);
}

function fmtPct(a, b) { return b > 0 ? Math.min(100, Math.round((a / b) * 100)) : 0; }

const STATUS_VN = {
  ready: "SẴN SÀNG", weak: "YẾU", missing: "THIẾU", training: "ĐANG TRAIN",
};

/* ------------------------------------------------------- pipeline 3 tầng */

function renderTrainLive(live) {
  const box = $("train-live");
  const runs = live.training || [];
  if (!runs.length) {
    box.innerHTML = '<p class="muted">Không có tiến trình train nào đang chạy.</p>' +
      (live.queue && live.queue.status && live.queue.status.startsWith("training")
        ? `<p class="muted">Queue: đang chờ <b>${live.queue.current}</b>…</p>` : "");
  } else {
    box.innerHTML = runs.map((t) => {
      const pct = t.target ? fmtPct(t.iters || 0, t.target) : 0;
      const eta = t.eta_h != null ? ` · còn ~${t.eta_h}h` : "";
      const rate = t.iters_per_min ? ` · ${t.iters_per_min} it/phút` : "";
      return `
        <div style="margin-bottom:10px">
          <b>${t.family || t.task_id}</b> <span class="muted">(PID ${t.pid})</span>
          <div class="num muted" style="font-size:11px;margin:4px 0 3px">
            ${(t.iters ?? 0).toLocaleString("vi")} / ${(t.target ?? "?").toLocaleString("vi")} iters${rate}${eta}
          </div>
          <div class="progress"><div style="width:${pct}%"></div></div>
        </div>`;
    }).join("");
  }
  // Queue
  if (live.queue && live.queue.plan && live.queue.plan.length) {
    const q = live.queue;
    box.innerHTML += `
      <div style="border-top:1px solid var(--line);padding-top:8px;font-size:11.5px" class="muted">
        Queue: ${q.done.length ? "✓ " + q.done.join(", ") + " · " : ""}
        ${q.current ? "▶ <b>" + q.current + "</b> · " : ""}
        chờ: ${q.plan.filter((k) => k !== q.current && !q.done.includes(k)).join(", ") || "—"}
      </div>`;
  }
}

function renderTwinLive(matrix) {
  const box = $("twin-live");
  const withTwin = Object.values(matrix).filter((e) => e.twin);
  if (!withTwin.length) {
    box.innerHTML = '<p class="muted">Chạy [Demo] trên một model ở bảng dưới để xuất GIF.</p>';
    return;
  }
  withTwin.sort((a, b) => (b.twin.generated_at || "").localeCompare(a.twin.generated_at || ""));
  box.innerHTML = '<div class="twin-strip">' + withTwin.slice(0, 4).map((e) => {
    const t = e.twin;
    const cls = t.success ? "ok" : "bad";
    return `<div class="twin-card">
      <img class="twin-thumb" loading="lazy" src="/api/twin/${e.family.key}"
        onclick="showModal('/api/twin/${e.family.key}', '${e.family.label} — ${t.verdict || ""}')">
      <span><b>${e.family.label}</b><br><span class="${cls}">${t.success ? "✓" : "✗"} ${t.verdict || ""}</span></span>
    </div>`;
  }).join("") + "</div>";
}

function renderDeployLive(matrix) {
  const entries = Object.values(matrix).filter((e) => e.family.trainable);
  const onnxOk = entries.filter((e) => e.onnx).length;
  const ready = entries.filter((e) => e.status === "ready").length;
  $("deploy-live").innerHTML = `
    <ul class="checkpoint checklist">
      <li><span class="${ready === entries.length ? "ok-mark" : "no-mark"}">${ready === entries.length ? "✓" : "•"}</span>
          Model đạt target: <b>${ready}/${entries.length}</b></li>
      <li><span class="${onnxOk === entries.length ? "ok-mark" : "no-mark"}">${onnxOk === entries.length ? "✓" : "•"}</span>
          ONNX đã bake normalizer: <b>${onnxOk}/${entries.length}</b></li>
      <li><span class="no-mark">•</span> Manifest schema-2: qua <code>uv run publish</code></li>
    </ul>
    <p class="deploy-note">⚠ Chưa có robot thật — dừng ở bước rehearsal:
    nạp ONNX vào mô phỏng CPU (tầng 2) đúng luồng robot sẽ dùng.</p>`;
}

/* --------------------------------------------------------- matrix table */

function renderMatrix(matrix, live) {
  MATRIX = matrix;
  const groups = {};
  Object.values(matrix)
    .filter((e) => e.family.trainable || e.status !== "missing")
    .sort((a, b) => a.family.group.localeCompare(b.family.group) || a.family.key.localeCompare(b.family.key))
    .forEach((e) => { (groups[e.family.group] ||= []).push(e); });

  const busy = (live.training || []).length > 0;
  const rows = [];
  for (const [g, entries] of Object.entries(groups)) {
    for (const e of entries) {
      const f = e.family;
      const pct = fmtPct(e.iters, f.target_iters);
      const status = busy && live.training.some((t) => t.family === f.key) ? "training" : e.status;
      const twinCell = e.twin
        ? `<span class="${e.twin.success ? "ok-mark" : "no-mark"}">${e.twin.success ? "✓" : "✗"}</span>`
        : '<span class="no-mark">–</span>';
      const verdict = e.twin
        ? `<span class="${e.twin.success ? "ok" : "bad"}">${e.twin.success ? "✓" : "✗"}</span> ${e.twin.verdict || ""}`
        : '<span class="muted">chưa chạy</span>';
      const trainDisabled = busy && !status.startsWith("training") ? "disabled" : "";
      rows.push(`<tr>
        <td><span class="group-tag">${g}</span></td>
        <td><div class="model-name">${f.label}</div>${f.note ? `<div class="model-note">${f.note}</div>` : ""}</td>
        <td class="progress-cell">
          <div class="num">${e.iters.toLocaleString("vi")} / ${f.target_iters.toLocaleString("vi")} iters · ${pct}%</div>
          <div class="progress"><div style="width:${pct}%"></div></div>
        </td>
        <td>${e.onnx ? '<span class="ok-mark">✓</span>' : '<span class="no-mark">–</span>'}</td>
        <td>${twinCell}</td>
        <td><div class="verdict">${verdict}</div></td>
        <td><div class="actions">
          <button class="btn small ghost" onclick="act('demo','${f.key}')" ${e.onnx ? "" : "disabled"}>Demo</button>
          <button class="btn small ghost" onclick="act('export','${f.key}')" ${e.checkpoint ? "" : "disabled"}>Export</button>
          <button class="btn small ghost" onclick="act('smoke','${f.key}')" ${f.trainable ? "" : "disabled"} ${trainDisabled}>Smoke</button>
          <button class="btn small" onclick="act('train','${f.key}')" ${f.trainable ? "" : "disabled"} ${trainDisabled}>Train</button>
        </div></td>
      </tr>`);
    }
  }
  $("matrix-body").innerHTML = rows.join("");
  const trainable = Object.values(matrix).filter((e) => e.family.trainable);
  const ready = trainable.filter((e) => e.status === "ready").length;
  $("matrix-count").textContent = `${ready}/${trainable.length} sẵn sàng`;
}

/* ------------------------------------------------------------- refresh */

async function refreshLive() {
  try {
    const live = await api("/api/live");
    const g = live.gpu;
    $("gpu-text").textContent = g
      ? `${g.name} · ${g.util_pct}% · ${g.mem_used_mb}/${g.mem_total_mb} MB · ${g.temp_c}°C`
      : "GPU —";
    document.querySelector(".dot").classList.toggle("on", (live.training || []).length > 0);
    renderTrainLive(live);
    if (MATRIX) {
      // cập nhật nhanh trạng thái training trên bảng
      document.querySelectorAll("#matrix-body tr").forEach(() => {});
      refreshMatrix(live);
    }
  } catch (e) { /* bỏ qua — server có thể đang khởi động lại */ }
}

async function refreshMatrix(live) {
  try {
    const doc = await api("/api/matrix");
    renderMatrix(doc.matrix, live || { training: [] });
    renderTwinLive(doc.matrix);
    renderDeployLive(doc.matrix);
  } catch (e) { toast("Lỗi quét matrix: " + e.message, true); }
}

async function refreshLog() {
  try {
    const d = await api("/api/trainlog");
    $("log-file").textContent = d.file || "";
    const el = $("train-log");
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 30;
    el.textContent = d.lines.length ? d.lines.join("\n") : "—";
    if (atBottom) el.scrollTop = el.scrollHeight;
  } catch (e) { /* ignore */ }
}

/* ------------------------------------------------------------ actions */

async function act(action, key) {
  const label = action === "queue" ? "Queue train" : `${action} ${key}`;
  try {
    const r = await api("/api/action", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, key }),
    });
    toast(`▶ Đã khởi động: ${label} (nền)`);
  } catch (e) {
    toast(`${label}: ${e.message}`, true);
  }
  setTimeout(refreshLive, 1200);
  setTimeout(refreshMatrix, 4000);
}

function showModal(src, caption) {
  $("modal-img").src = src + "?t=" + Date.now();
  $("modal-cap").textContent = caption;
  $("modal").hidden = false;
}

/* --------------------------------------------------------------- main */

document.querySelectorAll("[data-action]").forEach((b) =>
  b.addEventListener("click", () => act(b.dataset.action)));

(async () => {
  await refreshMatrix();
  await refreshLive();
  refreshLog();
  setInterval(refreshLive, 5000);
  setInterval(() => refreshMatrix(), 15000);
  setInterval(refreshLog, 5000);
  $("last-update").textContent = "cập nhật " + new Date().toLocaleTimeString("vi-VN");
})();
