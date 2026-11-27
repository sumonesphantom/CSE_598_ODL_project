"use strict";

const $ = (s) => document.querySelector(s);
const state = {
  status: null, samples: [], current: null, cond: "", restorer: "auto",
  result: null, view: "raw", showDet: true, showGt: false, req: 0,
};
const COND = {
  rain: { title: "Heavy rain", icon: "i-rain" },
  dust: { title: "Dust storm", icon: "i-dust" },
  clean: { title: "View clear", icon: "i-clear" },
};
const NAMES = { rain: "Rain", dust: "Dust", clean: "Clear" };
const pct = (v) => `${Math.round(v * 100)}%`;
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/* ---------- boot ---------- */
async function boot() {
  try {
    state.samples = await (await fetch("/api/samples")).json();
  } catch { state.samples = []; }
  renderThumbs();
  pollStatus();
  wireControls();
}

async function pollStatus() {
  try {
    state.status = await (await fetch("/api/status")).json();
  } catch {
    setStatus(false, "Can't reach the demo server. Is app/server.py running?");
    return setTimeout(pollStatus, 3000);
  }
  const s = state.status;
  fillRestorers(s.restorers);
  if (s.load_error) return setStatus(false, `Models failed to load: ${s.load_error}`);
  if (!s.ready) {
    setStatus(false, "Loading the detector. The first run downloads DETR.");
    return setTimeout(pollStatus, 1500);
  }
  setStatus(true, s.weather_detector ? "Ready, condition detector loaded" : "Ready, but the condition detector isn't on this machine");
  if (!state.current && state.samples.length) {
    const h = new URLSearchParams(location.hash.slice(1));   // e.g. #sample=<id>&view=restored&gt=1
    if (h.get("restorer")) { state.restorer = h.get("restorer"); $("#restorer").value = state.restorer; }
    if (h.get("cond")) { state.cond = h.get("cond"); setSeg("#cond-seg", "cond", state.cond); }
    if (h.get("gt") === "1") { state.showGt = true; $("#t-gt").checked = true; }
    if (h.get("view")) state.pendingView = h.get("view");
    const pick = state.samples.find((x) => x.id === h.get("sample")) || state.samples[0];
    select({ sampleId: pick.id });
  }
}

function setStatus(ready, text) {
  $("#status").dataset.ready = ready;
  $("#status-text").textContent = text;
}

function fillRestorers(list) {
  const sel = $("#restorer");
  if (sel.options.length) return;
  for (const r of list) {
    const o = new Option(r.available ? r.label : `${r.label} (start with --diffusion)`, r.id);
    o.disabled = !r.available;
    sel.add(o);
  }
  sel.value = state.restorer;
}

function renderThumbs() {
  const ul = $("#thumbs");
  if (!state.samples.length) {
    ul.innerHTML = `<li class="step-note">No DAWN frames found in data/dawn. Upload a photo to try the pipeline.</li>`;
    return;
  }
  ul.innerHTML = state.samples.map((s) => `
    <li><button class="thumb" data-id="${esc(s.id)}" aria-label="${NAMES[s.condition]} frame ${esc(s.name)}, ${s.objects} labeled objects">
      <img src="/api/thumb/${encodeURIComponent(s.id)}" alt="" loading="lazy" width="150" height="100">
      <span><svg class="ic"><use href="#${COND[s.condition].icon}"/></svg>${NAMES[s.condition]}, ${s.objects} objects</span>
    </button></li>`).join("");
  ul.querySelectorAll(".thumb").forEach((b) => b.addEventListener("click", () => select({ sampleId: b.dataset.id })));
}

/* ---------- controls ---------- */
function wireControls() {
  segment("#view-seg", "view", (v) => { state.view = v; applyView(); });
  segment("#cond-seg", "cond", (v) => { state.cond = v; run(); });
  $("#restorer").addEventListener("change", (e) => { state.restorer = e.target.value; run(); });
  $("#t-det").addEventListener("change", (e) => { state.showDet = e.target.checked; drawBoxes(); });
  $("#t-gt").addEventListener("change", (e) => { state.showGt = e.target.checked; drawBoxes(); });
  $("#retry").addEventListener("click", run);
  $("#file").addEventListener("change", (e) => e.target.files[0] && upload(e.target.files[0]));
  new ResizeObserver(drawBoxes).observe($("#frame"));
  wireSplit();
  wireDrop();
}

function segment(sel, key, onPick) {
  const group = $(sel);
  group.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b || b.disabled) return;
    group.querySelectorAll("button").forEach((x) => x.setAttribute("aria-checked", x === b));
    onPick(b.dataset[key]);
  });
  group.addEventListener("keydown", (e) => {
    if (!["ArrowLeft", "ArrowRight"].includes(e.key)) return;
    const items = [...group.querySelectorAll("button:not(:disabled)")];
    const i = items.findIndex((x) => x.getAttribute("aria-checked") === "true");
    const next = items[(i + (e.key === "ArrowRight" ? 1 : items.length - 1)) % items.length];
    next.focus(); next.click();
  });
}

function setSeg(sel, key, value) {
  $(sel).querySelectorAll("button").forEach((b) => b.setAttribute("aria-checked", b.dataset[key] === value));
}

function wireSplit() {
  const viewer = $("#viewer");
  const move = (e) => {
    const r = viewer.getBoundingClientRect();
    const x = Math.min(Math.max((e.clientX - r.left) / r.width, 0.02), 0.98);
    viewer.style.setProperty("--split", `${x * 100}%`);
  };
  const handle = $("#handle");
  handle.addEventListener("pointerdown", (e) => {
    handle.setPointerCapture(e.pointerId);
    handle.addEventListener("pointermove", move);
    handle.addEventListener("pointerup", () => handle.removeEventListener("pointermove", move), { once: true });
  });
}

function wireDrop() {
  const viewer = $("#viewer"), drop = $("#drop");
  let depth = 0;
  viewer.addEventListener("dragenter", (e) => { e.preventDefault(); depth++; drop.hidden = false; });
  viewer.addEventListener("dragover", (e) => e.preventDefault());
  viewer.addEventListener("dragleave", () => { if (--depth <= 0) { depth = 0; drop.hidden = true; } });
  viewer.addEventListener("drop", (e) => {
    e.preventDefault(); depth = 0; drop.hidden = true;
    const f = e.dataTransfer.files[0];
    if (f) upload(f);
  });
}

async function upload(file) {
  if (!file.type.startsWith("image/")) return showError("That file isn't an image. Try a JPEG or PNG.");
  const url = await shrink(file, 1600).catch(() => null);
  if (!url) return showError("Couldn't read that image. Try a JPEG or PNG.");
  select({ image: url });
}

function shrink(file, max) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const k = Math.min(1, max / Math.max(img.naturalWidth, img.naturalHeight));
      const c = document.createElement("canvas");
      c.width = Math.round(img.naturalWidth * k); c.height = Math.round(img.naturalHeight * k);
      c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
      URL.revokeObjectURL(img.src);
      resolve(c.toDataURL("image/jpeg", 0.9));
    };
    img.onerror = reject;
    img.src = URL.createObjectURL(file);
  });
}

function select(item) {
  state.current = item;
  document.querySelectorAll(".thumb").forEach((t) => {
    const on = t.dataset.id === item.sampleId;
    t.setAttribute("aria-current", on);
    if (on) { const ul = $("#thumbs"), li = t.parentElement; ul.scrollTo({ left: li.offsetLeft - ul.offsetLeft - (ul.clientWidth - li.clientWidth) / 2 }); }
  });
  run();
}

/* ---------- run ---------- */
async function run() {
  if (!state.current || !state.status?.ready) return;
  const id = ++state.req;
  busy(true);
  hideError();
  const body = { restorer: state.restorer, condition: state.cond || null };
  if (state.current.sampleId) body.sample_id = state.current.sampleId; else body.image = state.current.image;
  let res, data;
  try {
    res = await fetch("/api/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    data = await res.json();
  } catch {
    if (id === state.req) { busy(false); showError("The demo server stopped responding. Check the terminal running app/server.py."); }
    return;
  }
  if (id !== state.req) return;
  busy(false);
  if (!res.ok) return showError(data.detail || "Something went wrong running this frame.");
  state.result = data;
  render();
}

function busy(on) {
  $("#viewer").dataset.busy = on;
  $("#steps").dataset.busy = on;
  if (on) {
    $("#status-text").textContent = "Running…";
    $("#empty-text").textContent = "Running the pipeline on this frame…";
  }
  else if (state.status?.ready) setStatus(true, state.status.weather_detector ? "Ready, condition detector loaded" : "Ready, but the condition detector isn't on this machine");
}

function showError(msg) { $("#error-text").textContent = msg; $("#error").hidden = false; }
function hideError() { $("#error").hidden = true; }

/* ---------- render ---------- */
function render() {
  const r = state.result;
  $("#viewer").dataset.loaded = "true";
  $("#img-raw").src = r.raw;
  $("#img-raw").alt = "Frame as captured";
  if (r.restored) { $("#img-restored").src = r.restored; $("#img-restored").alt = "Frame after restoration"; }
  const restoredBtn = $('#view-seg [data-view="restored"]'), compareBtn = $('#view-seg [data-view="compare"]');
  restoredBtn.disabled = compareBtn.disabled = !r.restored;
  restoredBtn.title = compareBtn.title = r.restored ? "" : "Nothing was restored, so there's nothing to compare";
  if (state.pendingView && r.restored) { state.view = state.pendingView; setSeg("#view-seg", "view", state.view); }
  state.pendingView = null;
  if (!r.restored && state.view !== "raw") { state.view = "raw"; setSeg("#view-seg", "view", "raw"); }
  $("#t-gt").disabled = !r.gt;
  $("#t-gt").closest(".check").title = r.gt ? "" : "Uploaded photos have no labels";

  if (r.needs_condition) return renderAsk();
  renderSign();
  renderDecide();
  renderRestore();
  renderDetect();
  renderChanges();
  applyView();
  const steps = $("#steps");
  steps.dataset.fresh = "false"; void steps.offsetWidth; steps.dataset.fresh = "true";
}

function renderAsk() {
  const sign = $("#sign");
  sign.dataset.kind = "ask";
  sign.querySelector("use").setAttribute("href", "#i-alert");
  $("#sign-title").textContent = "Which weather is this?";
  $("#sign-sub").textContent = "Pick Rain, Dust or Clear under Decide";
  $("#decide-chip").textContent = "";
  $("#decide-source").innerHTML = "The condition detector isn't on this machine, and an uploaded photo has no label. <strong>Choose the condition</strong> to run the rest of the pipeline, or set Restore to a fixed method.";
  $("#decide-bars").innerHTML = "";
  for (const id of ["#restore-chip", "#detect-chip"]) $(id).textContent = "";
  $("#restore-why").textContent = "";
  $("#counts tbody").innerHTML = ""; $("#counts tfoot")?.remove();
  $("#known").textContent = "";
  $("#changes").innerHTML = "";
  applyView();
}

function total(c) { return Object.values(c).reduce((a, b) => a + b, 0); }

function renderSign() {
  const r = state.result, cond = r.decide.condition, n = total(r.detect.counts_restored);
  const sign = $("#sign");
  sign.dataset.kind = cond === "clean" ? "clear" : "degraded";
  sign.querySelector("use").setAttribute("href", `#${COND[cond].icon}`);
  $("#sign-title").textContent = COND[cond].title;
  const objects = `${n} object${n === 1 ? "" : "s"} detected`;
  let sub;
  if (cond === "clean") sub = objects;
  else if (r.restore.method === "none") sub = `Degraded view, operator alerted\nNot restored, ${objects}`;
  else sub = `Degraded view, operator alerted\n${r.restore.label} applied, ${n} object${n === 1 ? "" : "s"}`;
  $("#sign-sub").innerHTML = esc(sub).replace("\n", "<br>");
}

function renderDecide() {
  const d = state.result.decide;
  const chip = $("#decide-chip");
  chip.textContent = NAMES[d.condition];
  chip.dataset.tone = d.condition;
  let note;
  if (d.source === "detector") note = `Condition detector says <strong>${NAMES[d.condition].toLowerCase()}</strong>. It only calls rain or dust above ${pct(d.threshold)} confidence.`;
  else if (d.source === "label") note = `Taken from the dataset's label. The condition detector isn't on this machine yet, so it can't judge the frame itself.`;
  else note = `Set by you.${d.detected ? ` The detector said <strong>${NAMES[d.detected].toLowerCase()}</strong>.` : ""}`;
  $("#decide-source").innerHTML = note;
  const bars = $("#decide-bars");
  if (!d.probs) { bars.innerHTML = ""; return; }
  const entries = Object.entries(d.probs).sort((a, b) => b[1] - a[1]);
  bars.innerHTML = entries.map(([k, v], i) => `
    <div class="bar-row" data-top="${i === 0}"><span>${NAMES[k] || esc(k)}</span>
      <span class="track"><span class="fill" style="width:${(v * 100).toFixed(1)}%"></span></span><span class="v">${pct(v)}</span></div>`).join("");
}

function renderRestore() {
  const r = state.result.restore;
  const chip = $("#restore-chip");
  chip.textContent = r.method === "none" ? "Passed through" : r.label;
  chip.dataset.tone = r.method === "none" ? "" : "on";
  const ms = r.method === "none" ? "" : ` Took ${r.seconds < 1 ? `${Math.round(r.seconds * 1000)} ms` : `${r.seconds.toFixed(1)} s`}.`;
  $("#restore-why").innerHTML = (r.why ? esc(r.why) : "Method chosen by you, not by our results.") + ms;
}

function renderDetect() {
  const d = state.result.detect;
  const classes = [...new Set([...Object.keys(d.counts_raw), ...Object.keys(d.counts_restored)])].sort();
  const restored = !!state.result.restored;
  const cell = (a, b) => {
    if (!restored) return `<td>–</td>`;
    const cls = b > a ? "up" : b < a ? "down" : "";
    return `<td class="${cls}">${b}</td>`;
  };
  const tbody = $("#counts tbody");
  tbody.innerHTML = classes.length
    ? classes.map((c) => { const a = d.counts_raw[c] || 0, b = d.counts_restored[c] || 0; return `<tr><td>${esc(c)}</td><td>${a}</td>${cell(a, b)}</tr>`; }).join("")
    : `<tr class="empty"><td colspan="3">No people or vehicles above ${state.status.show_score} confidence</td></tr>`;
  $("#counts tfoot")?.remove();
  const ta = total(d.counts_raw), tb = total(d.counts_restored);
  $("#counts").insertAdjacentHTML("beforeend", `<tfoot><tr><td>Total</td><td>${ta}</td>${cell(ta, tb)}</tr></tfoot>`);
  $("#detect-chip").textContent = `DETR, ${Math.round(d.ms[0])} ms`;
  const known = state.result.known;
  $("#known").innerHTML = known.length
    ? `On real ${state.result.decide.condition} without restoring, this detector scores ` +
      known.map((k) => `<strong>mAP50 ${k.map50.toFixed(2)}</strong> on ${esc(k.set.split(" ")[0])} and finds <strong>${pct(k.person_recall)}</strong> of people`).join("; ") + "."
    : "";
}

function renderChanges() {
  const r = state.result, dl = $("#changes");
  if (!r.restored) {
    dl.innerHTML = `<p class="lead">Nothing was restored, so the boxes are the raw detector's.</p>`;
    return;
  }
  const row = (n, text, tone) => `<dt ${tone && n ? `data-${tone}="true"` : ""}>${n}</dt><dd>${text}</dd>`;
  if (r.hallucination) {
    const h = r.hallucination;
    dl.innerHTML = `<p class="lead">Checked against the dataset's labeled objects, for boxes at ${state.status.show_score} confidence or higher.</p>` +
      row(h.lost, "<b>lost:</b> real objects found before restoring, missed after", "bad") +
      row(h.gained, "<b>gained:</b> real objects missed before, found after", "good") +
      row(h.invented, "<b>invented:</b> new false boxes where the raw frame had none", "bad") +
      row(h.relabeled, "<b>relabeled:</b> correct boxes that switched class", "bad");
  } else {
    const c = r.changes;
    dl.innerHTML = `<p class="lead">An uploaded photo has no labels, so this compares the boxes only.</p>` +
      row(c.added.length, "<b>new boxes</b> after restoring") +
      row(c.removed.length, "<b>boxes gone</b> after restoring") +
      row(c.relabeled.length, "<b>boxes that switched class</b>");
  }
}

/* ---------- viewer ---------- */
function applyView() {
  const viewer = $("#viewer");
  viewer.dataset.mode = state.view;
  $("#handle").hidden = state.view !== "compare";
  $("#legend").hidden = state.view === "compare" || !state.result;
  const r = state.result;
  const restoredView = state.view === "restored";
  document.querySelectorAll(".legend .only-restored").forEach((li) => (li.hidden = !restoredView));
  document.querySelectorAll(".legend .only-gt").forEach((li) => (li.hidden = !(state.showGt && r?.gt)));
  drawBoxes();
}

function drawBoxes() {
  const svg = $("#boxes"), r = state.result;
  if (!r || r.needs_condition || state.view === "compare") { svg.innerHTML = ""; return; }
  const [W, H] = r.size;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const frame = $("#frame");
  const k = 1 / Math.min(frame.clientWidth / W, frame.clientHeight / H || 1);
  const fs = 12 * k, pad = 4 * k;
  const parts = [];
  const rect = (b, cls) => `<rect class="${cls}" x="${b[0]}" y="${b[1]}" width="${b[2] - b[0]}" height="${b[3] - b[1]}"/>`;
  const tag = (b, text, fill, color = "#14181b") => {
    const w = text.length * fs * 0.56 + pad * 2, h = fs + pad * 1.6;
    const y = b[1] - h < 0 ? b[1] : b[1] - h;
    return `<g class="tag"><rect x="${b[0]}" y="${y}" width="${w}" height="${h}" rx="${2 * k}" style="fill:${fill}"/>` +
      `<text x="${b[0] + pad}" y="${y + h - pad * 1.05}" style="font-size:${fs}px;fill:${color}">${esc(text)}</text></g>`;
  };
  if (state.showGt && r.gt) for (const g of r.gt) parts.push(`<g>${rect(g.box, "gt")}</g>`);
  if (state.showDet) {
    const restored = state.view === "restored";
    const list = restored ? r.detect.restored : r.detect.raw;
    const added = new Set(restored ? r.changes.added : []);
    if (restored) for (const j of r.changes.removed) {
      const d = r.detect.raw[j];
      parts.push(`<g>${rect(d.box, "halo")}${rect(d.box, "rem")}${tag(d.box, `${d.label} gone`, "#ff6b5e")}</g>`);
    }
    list.forEach((d, i) => {
      const isNew = added.has(i);
      parts.push(`<g>${rect(d.box, "halo")}${rect(d.box, isNew ? "add" : "det")}` +
        `${tag(d.box, `${d.label} ${Math.round(d.score * 100)}${isNew ? " new" : ""}`, isNew ? "#5aa9ff" : "#f5c518")}</g>`);
    });
  }
  svg.innerHTML = parts.join("");
}

boot();
