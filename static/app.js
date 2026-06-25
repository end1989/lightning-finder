const state = { meta: null, fps: 30, duration: 0, frameCount: 0,
                mode: "playback", curFrame: 0, events: [], markers: {} };

const $ = (id) => document.getElementById(id);
const video = $("video");
const frameImg = $("frame-img");

function fmtTime(s) {
  if (!isFinite(s)) s = 0;
  const ms = Math.round((s - Math.floor(s)) * 1000);
  let x = Math.floor(s);
  const h = Math.floor(x / 3600); x %= 3600;
  const m = Math.floor(x / 60); const sec = x % 60;
  const p = (n, w = 2) => String(n).padStart(w, "0");
  return `${p(h)}:${p(m)}:${p(sec)}.${p(ms, 3)}`;
}

function curFrameFromVideo() { return Math.round(video.currentTime * state.fps); }

function updateTimeLabel(f) {
  state.curFrame = f;
  $("time-label").textContent = `frame ${f} · ${fmtTime(f / state.fps)}`;
}

function seekToFrame(f) {
  f = Math.max(0, Math.min((state.frameCount || 1) - 1, f));
  if (state.mode === "precision") { showFrame(f); }
  else { video.currentTime = f / state.fps; updateTimeLabel(f); }
}

// showFrame/precision are defined in Task 7; provide a fallback for the skeleton.
function showFrame(f) { updateTimeLabel(f); }

function togglePlay() {
  if (typeof exitPrecision === "function") exitPrecision();
  if (video.paused) video.play(); else video.pause();
}

async function initMeta() {
  state.meta = await (await fetch("/api/video/meta")).json();
  state.fps = state.meta.fps || 30;
  state.duration = state.meta.duration || 0;
  state.frameCount = state.meta.frame_count || 0;
  $("res-label").textContent =
    `${state.meta.width}×${state.meta.height} · ${state.fps.toFixed(2)} fps · ${state.frameCount} frames`;
  video.src = "/video";
}

function wireSkeleton() {
  $("btn-play").addEventListener("click", togglePlay);
  $("timeline").addEventListener("click", (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    const frac = (e.clientX - r.left) / r.width;
    seekToFrame(Math.round(frac * ((state.frameCount || 1) - 1)));
  });
  video.addEventListener("timeupdate", () => {
    if (state.mode === "playback") {
      const f = curFrameFromVideo();
      updateTimeLabel(f);
      setPlayhead(state.duration ? video.currentTime / state.duration : 0);
    }
  });
}

function toast(msg) {
  const t = $("toast"); t.textContent = msg; t.hidden = false;
  clearTimeout(toast._t); toast._t = setTimeout(() => (t.hidden = true), 2500);
}

// ---- Open a video (in-browser file browser) --------------------------------
function showOpenScreen() { $("open-screen").hidden = false; browseTo(""); }
function hideOpenScreen() { $("open-screen").hidden = true; }
function baseName(p) { return p.replace(/[\\/]+$/, "").split(/[\\/]/).pop() || p; }

function entryRow(icon, label, onClick, isVideo) {
  const row = document.createElement("div");
  row.className = "entry" + (isVideo ? " video" : "");
  const ic = document.createElement("span"); ic.className = "ico"; ic.textContent = icon;
  const tx = document.createElement("span"); tx.textContent = label;
  row.append(ic, tx);
  row.addEventListener("click", onClick);
  return row;
}

async function browseTo(path) {
  const r = await (await fetch(`/api/browse?path=${encodeURIComponent(path)}`)).json();
  $("open-path").textContent = r.path || "This PC";
  const list = $("open-list");
  list.innerHTML = "";
  if (r.parent !== null) list.appendChild(entryRow("⬆", "..", () => browseTo(r.parent), false));
  r.dirs.forEach((d) => list.appendChild(entryRow("📁", baseName(d), () => browseTo(d), false)));
  r.videos.forEach((v) => list.appendChild(entryRow("🎬", baseName(v), () => openVideo(v), true)));
  if (!r.dirs.length && !r.videos.length)
    list.appendChild(entryRow("", "(no folders or videos here)", () => {}, false));
}

async function openVideo(path) {
  $("open-path").textContent = "Opening…";
  const r = await fetch("/api/open", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
  if (!r.ok) { toast("Couldn't open that file"); return; }
  hideOpenScreen();
  await initVideo();
}

function wireOpen() { $("btn-open").addEventListener("click", showOpenScreen); }

async function initVideo() {
  await initMeta();
  state.events = [];
  if (typeof loadEvents === "function") await loadEvents();
}

async function main() {
  wireSkeleton();
  wireOpen();
  if (typeof wireKeyboard === "function") wireKeyboard();
  if (typeof wirePrecision === "function") wirePrecision();
  if (typeof wireExport === "function") wireExport();
  const st = await (await fetch("/api/state")).json();
  if (st.loaded) await initVideo();
  else showOpenScreen();
}
main();

// ---- Task 6: events, timeline markers, flash list, navigation ----------
async function loadEvents() {
  const ol = $("flash-list");
  ol.innerHTML =
    '<li class="scanning">Scanning for flashes… the first open of a long video can take a few minutes.</li>';
  try {
    const r = await (await fetch("/api/events")).json();
    state.events = r.events;
    renderTimeline();
    renderList();
  } catch (e) {
    ol.innerHTML = `<li class="scanning">Failed to load flashes: ${e.message}</li>`;
  }
}

function setPlayhead(frac) {
  let ph = document.querySelector("#timeline .playhead");
  if (!ph) {
    ph = document.createElement("div");
    ph.className = "playhead";
    $("timeline").appendChild(ph);
  }
  ph.style.left = `${Math.max(0, Math.min(1, frac)) * 100}%`;
}

function renderTimeline() {
  const tl = $("timeline");
  tl.querySelectorAll(".marker").forEach((m) => m.remove());
  const dur = state.duration || (state.frameCount / state.fps) || 1;
  for (const e of state.events) {
    const m = document.createElement("div");
    m.className = "marker";
    m.style.left = `${(e.peak_time / dur) * 100}%`;
    m.title = `frame ${e.peak_frame} · ${e.timecode}`;
    m.addEventListener("click", (ev) => { ev.stopPropagation(); seekToFrame(e.peak_frame); });
    tl.appendChild(m);
  }
}

function renderList() {
  const ol = $("flash-list");
  ol.innerHTML = "";
  state.events.forEach((e, i) => {
    const li = document.createElement("li");
    li.dataset.peak = e.peak_frame;
    if (e.status === "rejected") li.classList.add("rejected");
    const label = document.createElement("span");
    label.textContent = `#${i + 1} · ${e.timecode}`;
    const frameNum = document.createElement("span");
    frameNum.className = "muted";
    frameNum.textContent = `f${e.peak_frame}`;
    li.appendChild(label);
    li.appendChild(frameNum);
    li.addEventListener("click", () => seekToFrame(e.peak_frame));
    ol.appendChild(li);
  });
}

function highlightActive(frame) {
  document.querySelectorAll("#flash-list li").forEach((li) =>
    li.classList.toggle("active", Number(li.dataset.peak) === frame));
}

function jumpToFlash(dir) {
  if (!state.events.length) return;
  const cur = state.curFrame;
  const peaks = state.events.map((e) => e.peak_frame);
  let target = null;
  if (dir > 0) target = peaks.find((p) => p > cur);
  else target = [...peaks].reverse().find((p) => p < cur);
  if (target == null) target = dir > 0 ? peaks[0] : peaks[peaks.length - 1];
  seekToFrame(target);
  highlightActive(target);
}

// ---- Task 7: precision mode, frame stepping, full-quality grab ----------
function setModeLabel() { $("mode-label").textContent = state.mode; }

function enterPrecision(f) {
  state.mode = "precision";
  video.pause();
  video.hidden = true;
  frameImg.hidden = false;
  setModeLabel();
  showFrame(f);
}

function exitPrecision() {
  if (state.mode !== "precision") return;
  state.mode = "playback";
  frameImg.hidden = true;
  video.hidden = false;
  setModeLabel();
  video.currentTime = state.curFrame / state.fps;
}

// Real implementation; overrides the Task 5 skeleton fallback because this
// script runs later in the same file.
function showFrame(f) {
  f = Math.max(0, Math.min((state.frameCount || 1) - 1, f));
  state.curFrame = f;
  // Fast JPEG preview for display; Grab still pulls the lossless full-res PNG.
  frameImg.src = `/api/frame/${f}.jpg`;
  updateTimeLabel(f);
  setPlayhead(state.duration ? (f / state.fps) / state.duration : 0);
  highlightActive(f);
}

function stepFrame(d) {
  if (state.mode !== "precision") enterPrecision(curFrameFromVideo());
  else showFrame(state.curFrame + d);
}

function stepTime(dt) {
  const f = Math.round((state.curFrame / state.fps + dt) * state.fps);
  if (state.mode !== "precision") enterPrecision(f); else showFrame(f);
}

async function grab() {
  try {
    const r = await fetch("/api/grab", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ index: state.curFrame }),
    });
    if (!r.ok) { toast(`Grab failed: ${r.status}`); return; }
    const data = await r.json();
    toast(`Saved ${data.path}`);
  } catch (e) {
    toast(`Grab error: ${e.message}`);
  }
}

function wirePrecision() {
  $("btn-frame-back").addEventListener("click", () => stepFrame(-1));
  $("btn-frame-fwd").addEventListener("click", () => stepFrame(1));
  $("btn-grab").addEventListener("click", grab);
}

// ---- Task 8: export, keyboard, markers, rescan -------------------------
function wireExport() {
  $("btn-export-csv").addEventListener("click", () => { window.location = "/api/export?fmt=csv"; });
  $("btn-export-json").addEventListener("click", () => { window.location = "/api/export?fmt=json"; });

  $("btn-prev-flash").addEventListener("click", () => jumpToFlash(-1));
  $("btn-next-flash").addEventListener("click", () => jumpToFlash(1));

  $("btn-rescan").addEventListener("click", async () => {
    const sensitivity = parseFloat($("sensitivity").value);
    try {
      const resp = await fetch("/api/scan", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sensitivity }),
      });
      if (!resp.ok) { toast(`Rescan failed: ${resp.status}`); return; }
      const r = await resp.json();
      state.events = r.events;
      renderTimeline();
      renderList();
      toast(`Found ${state.events.length} flashes`);
    } catch (err) {
      toast(`Rescan error: ${err.message}`);
    }
  });

  $("btn-best-shots").addEventListener("click", startBestShots);
}

// ---- Best Shots: rank flashes by lightning-channel strength ----------------
let _boltPoll = null;

async function startBestShots() {
  try {
    const r = await (await fetch("/api/refine-bolts", { method: "POST" })).json();
    if (r.status === "done") { applyBestShots(r); return; }
    $("flash-list").innerHTML =
      '<li class="scanning">Finding best shots… checking each flash for a real channel.</li>';
    clearInterval(_boltPoll);
    _boltPoll = setInterval(pollBestShots, 1200);
  } catch (e) {
    toast(`Best Shots error: ${e.message}`);
  }
}

async function pollBestShots() {
  try {
    const r = await (await fetch("/api/refine-bolts")).json();
    if (r.status === "running") {
      const pct = r.total ? Math.round((100 * r.done) / r.total) : 0;
      $("flash-list").innerHTML =
        `<li class="scanning">Finding best shots… ${r.done}/${r.total} (${pct}%)</li>`;
    } else if (r.status === "done") {
      clearInterval(_boltPoll); _boltPoll = null;
      applyBestShots(r);
    } else if (r.status === "error") {
      clearInterval(_boltPoll); _boltPoll = null;
      toast("Best Shots failed");
    }
  } catch (e) { /* transient; keep polling */ }
}

function applyBestShots(r) {
  state.events = r.events;            // sorted by channel strength, with bolt fields
  state.bestShots = true;
  renderTimeline();
  renderBestShots(r.events);
  $("flash-count").textContent = `· ${r.n_bolts} bolts / ${r.events.length}`;
  toast(`${r.n_bolts} real bolts found`);
}

function renderBestShots(items) {
  const ol = $("flash-list");
  ol.innerHTML = "";
  items.forEach((e) => {
    const target = e.bolt_frame != null ? e.bolt_frame : e.peak_frame;
    const li = document.createElement("li");
    li.dataset.peak = e.peak_frame;
    li.dataset.target = target;
    if (!e.is_bolt) li.classList.add("glow");
    const left = document.createElement("span");
    left.textContent = `${e.is_bolt ? "⚡" : "·"} ${e.bolt_timecode || e.timecode}`;
    const right = document.createElement("span");
    right.className = "muted";
    right.textContent = e.is_bolt ? `${Math.round(e.bolt_score)}` : "glow";
    li.append(left, right);
    li.addEventListener("click", () => enterPrecision(target));
    ol.appendChild(li);
  });
}

async function setStatus(peakFrame, status) {
  await fetch("/api/markers", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ peak_frame: peakFrame, status }),
  });
  const ev = state.events.find((x) => x.peak_frame === peakFrame);
  if (ev) ev.status = status;
  renderList();
}

function currentFlash() {
  if (!state.events.length) return null;
  // the flash whose span contains the cursor, else the nearest by peak frame
  const containing = state.events.find(
    (e) => e.start_frame <= state.curFrame && state.curFrame <= e.end_frame);
  if (containing) return containing;
  let best = state.events[0];
  for (const e of state.events) {
    if (Math.abs(e.peak_frame - state.curFrame) < Math.abs(best.peak_frame - state.curFrame)) {
      best = e;
    }
  }
  return best;
}

function wireKeyboard() {
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT") return;
    switch (e.key) {
      case "ArrowLeft":  e.preventDefault(); e.shiftKey ? jumpToFlash(-1) : stepFrame(-1); break;
      case "ArrowRight": e.preventDefault(); e.shiftKey ? jumpToFlash(1)  : stepFrame(1);  break;
      case " ":          e.preventDefault(); togglePlay(); break;
      case "g": case "G": grab(); break;
      case ",": stepTime(-0.1); break;
      case ".": stepTime(0.1); break;
      case "x": case "X": { const f = currentFlash(); if (f) setStatus(f.peak_frame, "rejected"); break; }
      case "c": case "C": { const f = currentFlash(); if (f) setStatus(f.peak_frame, "confirmed"); break; }
    }
  });
}
