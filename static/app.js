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
    if (state.mode === "playback") updateTimeLabel(curFrameFromVideo());
  });
}

function wireSkeleton() {
  $("btn-play").addEventListener("click", togglePlay);
  $("timeline").addEventListener("click", (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    const frac = (e.clientX - r.left) / r.width;
    seekToFrame(Math.round(frac * (state.frameCount || 1)));
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
