/* Radar dashboard SPA. Polls /api/state every 200 ms and renders
 * tracks on a top-down canvas plot. Track colors follow the model
 * classes: green=human, blue=vehicle, gray=unknown.
 */
const CLASS_COLOR = { human: "#4caf50", vehicle: "#2196f3", unknown: "#888" };
const RANGE_RINGS = [1, 2, 4, 8, 16, 32, 64];  /* metres */
/* 800 px canvas / 100 m radius = 4 px per metre. */
const SCALE = 4;

let canvas, ctx;
let lastTracks = [];

function init() {
  canvas = document.getElementById("radar-canvas");
  ctx = canvas.getContext("2d");
  setInterval(poll, 200);
  poll();
}

async function poll() {
  try {
    const r = await fetch("/api/state");
    const s = await r.json();
    update(s);
  } catch (e) {
    console.error("poll failed:", e);
  }
}

function update(state) {
  document.getElementById("frame-count").textContent = state.frame_count ?? 0;
  document.getElementById("heap").textContent = state.free_heap ?
    Math.round(state.free_heap / 1024) : "—";
  if (state.infer_ms != null) {
    document.getElementById("infer-ms").textContent = state.infer_ms.toFixed(1);
  }
  if (state.uptime_s != null) {
    document.getElementById("uptime").textContent = Math.round(state.uptime_s);
  }
  if (state.fps != null) {
    document.getElementById("fps").textContent = state.fps.toFixed(1);
  }

  lastTracks = state.tracks || [];
  drawRadar();
  drawList();
}

function drawRadar() {
  const w = canvas.width, h = canvas.height;
  const cx = w / 2, cy = h / 2;

  ctx.fillStyle = "#0a0b0e";
  ctx.fillRect(0, 0, w, h);

  /* Range rings */
  ctx.strokeStyle = "#1c1f25";
  ctx.lineWidth = 1;
  RANGE_RINGS.forEach(r => {
    ctx.beginPath();
    ctx.arc(cx, cy, r * SCALE, 0, 2 * Math.PI);
    ctx.stroke();
  });

  /* Crosshairs */
  ctx.beginPath();
  ctx.moveTo(cx, 0); ctx.lineTo(cx, h);
  ctx.moveTo(0, cy); ctx.lineTo(w, cy);
  ctx.stroke();

  /* Tracks */
  for (const t of lastTracks) {
    const px = cx + t.x * SCALE;
    const py = cy - t.y * SCALE;  /* y inverted to "north is up" */
    const color = CLASS_COLOR[t.class] || "#888";

    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(px, py, 8, 0, 2 * Math.PI);
    ctx.fill();

    ctx.fillStyle = "#fff";
    ctx.font = "11px sans-serif";
    ctx.fillText(`#${t.id}`, px + 10, py + 4);
    ctx.fillText(`${(t.confidence * 100).toFixed(0)}%`, px + 10, py + 16);
  }
}

function drawList() {
  const ul = document.getElementById("tracks-list");
  ul.innerHTML = "";
  for (const t of lastTracks) {
    const li = document.createElement("li");
    li.className = t.class;
    if (!t.published) li.classList.add("pending");
    li.innerHTML = `
      <div class="id">Track #${t.id} <span class="conf">${(t.confidence * 100).toFixed(1)}%</span></div>
      <div>${t.class} &middot; (${t.x.toFixed(2)}, ${t.y.toFixed(2)}) m</div>
      <div>v=${t.speed.toFixed(2)} m/s &middot; age=${t.age_s.toFixed(1)} s &middot; hits=${t.hits ?? "—"}</div>
    `;
    ul.appendChild(li);
  }
}

init();
