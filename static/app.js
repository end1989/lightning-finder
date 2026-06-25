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
  video.addEventListener("timeupdate", () => {
    if (state.mode === "playback") {
      const f = curFrameFromVideo();
      updateTimeLabel(f);
      setPlayhead((state.duration ? video.currentTime / state.duration : 0));
    }
  });
}

function wireSkeleton() {
  $("btn-play").addEventListener("click", togglePlay);
  $("timeline").addEventListener("click", (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    const frac = (e.clientX - r.left) / r.width;
    seekToFrame(Math.round(frac * ((state.frameCount || 1) - 1)));
  });
}

function toast(msg) {
  const t = $("toast"); t.textContent = msg; t.hidden = false;
  clearTimeout(toast._t); toast._t = setTimeout(() => (t.hidden = true), 2500);
}

async function main() {
  await initMeta();
  wireSkeleton();
  if (typeof loadEvents === "function") await loadEvents();   // Task 6
  if (typeof wireKeyboard === "function") wireKeyboard();     // Task 8
  if (typeof wirePrecision === "function") wirePrecision();   // Task 7
  if (typeof wireExport === "function") wireExport();         // Task 8
}
main();

// ---- Task 6: events, timeline markers, flash list, navigation ----------
async function loadEvents() {
  const r = await (await fetch("/api/events")).json();
  state.events = r.events;
  renderTimeline();
  renderList();
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
  frameImg.src = `/api/frame/${f}.png`;
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
  const r = await fetch("/api/grab", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ index: state.curFrame }),
  });
  const data = await r.json();
  toast(`Saved ${data.path}`);
}

function wirePrecision() {
  $("btn-frame-back").addEventListener("click", () => stepFrame(-1));
  $("btn-frame-fwd").addEventListener("click", () => stepFrame(1));
  $("btn-grab").addEventListener("click", grab);
}
