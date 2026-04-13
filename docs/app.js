const $ = (id) => document.getElementById(id);

function fmt(x, d = 3) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return Number(x).toFixed(d);
}

function clamp(x, a, b) {
  return Math.max(a, Math.min(b, x));
}

function polarToXY(cx, cy, r, deg) {
  const rad = (deg * Math.PI) / 180;
  const x = cx + r * Math.sin(rad);
  const y = cy - r * Math.cos(rad);
  return [x, y];
}

function drawMap(canvas, run, selectedEvent, metersPerSecondOverride) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const cx = Math.round(w * 0.5);
  const cy = Math.round(h * 0.52);
  const maxR = Math.min(w, h) * 0.36;

  ctx.save();
  ctx.globalAlpha = 0.9;
  ctx.strokeStyle = "rgba(255,255,255,0.10)";
  ctx.lineWidth = 1;
  for (const k of [0.25, 0.5, 0.75, 1.0]) {
    ctx.beginPath();
    ctx.arc(cx, cy, maxR * k, 0, Math.PI * 2);
    ctx.stroke();
  }
  for (let deg = 0; deg < 360; deg += 30) {
    const [x1, y1] = polarToXY(cx, cy, maxR * 0.12, deg);
    const [x2, y2] = polarToXY(cx, cy, maxR, deg);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  }
  ctx.restore();

  ctx.save();
  ctx.fillStyle = "rgba(245,193,75,0.90)";
  ctx.beginPath();
  ctx.arc(cx, cy, 6, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  const mps = metersPerSecondOverride ?? run.meters_per_second ?? 300;
  const events = run.events || [];
  const dmax = Math.max(1e-6, ...events.map((e) => (e.distance_m ?? (e.duration_s ?? 0) * mps)));

  function drawEvent(e, isSelected) {
    const dur = e.duration_s ?? 0;
    const bearing = e.bearing_deg ?? (e.orientation_deg != null ? (e.orientation_deg + run.angle_offset_deg) % 360 : null);
    if (bearing == null) return;
    const dist = e.distance_m ?? dur * mps;
    const r = clamp((dist / dmax) * maxR, 8, maxR);
    const [x, y] = polarToXY(cx, cy, r, bearing);

    ctx.save();
    ctx.globalAlpha = isSelected ? 1.0 : 0.5;
    ctx.strokeStyle = isSelected ? "rgba(245,193,75,0.95)" : "rgba(83,209,143,0.75)";
    ctx.fillStyle = isSelected ? "rgba(245,193,75,0.95)" : "rgba(83,209,143,0.75)";
    ctx.lineWidth = isSelected ? 3 : 2;

    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(x, y);
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(x, y, isSelected ? 6 : 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  for (const e of events) drawEvent(e, false);
  if (selectedEvent) drawEvent(selectedEvent, true);
}

function setMeta(el, e, run, metersPerSecondOverride) {
  if (!e) {
    el.innerHTML = "<span class='muted'>Select an event.</span>";
    return;
  }
  const mps = metersPerSecondOverride ?? run.meters_per_second ?? 300;
  const bearing = e.bearing_deg ?? (e.orientation_deg != null ? (e.orientation_deg + run.angle_offset_deg) % 360 : null);
  const dist = e.distance_m ?? (e.duration_s ?? 0) * mps;
  el.innerHTML = [
    `<div><strong>Event</strong> #${e.id}</div>`,
    `<div><strong>Time</strong> ${fmt(e.start_s, 2)}s → ${fmt(e.end_s, 2)}s</div>`,
    `<div><strong>Confidence</strong> ${fmt(e.confidence, 3)}</div>`,
    `<div><strong>Duration</strong> ${fmt(e.duration_s, 3)} s</div>`,
    `<div><strong>Orientation</strong> ${e.orientation_deg == null ? "—" : `${fmt(e.orientation_deg, 1)}°`}</div>`,
    `<div><strong>Bearing</strong> ${bearing == null ? "—" : `${fmt(bearing, 1)}°`}</div>`,
    `<div><strong>Distance</strong> ${dist == null ? "—" : `${fmt(dist, 1)} m`}</div>`,
  ].join("");
}

async function loadJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Failed to fetch ${url}: ${r.status}`);
  return await r.json();
}

function buildTimeline(container, run, onSelect) {
  container.innerHTML = "";
  const events = run.events || [];
  for (const e of events) {
    const div = document.createElement("button");
    div.type = "button";
    div.className = "seg";
    div.title = `t=${fmt(e.start_s, 2)}→${fmt(e.end_s, 2)}  p=${fmt(e.confidence, 3)}  dur=${fmt(e.duration_s, 3)}s`;
    const pill = document.createElement("span");
    pill.className = "pill " + ((e.confidence ?? 0) >= 0.6 ? "good" : "");
    const label = document.createElement("span");
    label.textContent = `#${e.id}  ${fmt(e.start_s, 2)}–${fmt(e.end_s, 2)}s`;
    div.appendChild(pill);
    div.appendChild(label);
    div.addEventListener("click", () => onSelect(e));
    container.appendChild(div);
  }
}

async function main() {
  const clipSelect = $("clipSelect");
  const video = $("video");
  const timeline = $("timeline");
  const canvas = $("map");
  const meta = $("eventMeta");
  const scale = $("scale");
  const scaleVal = $("scaleVal");

  const manifest = await loadJSON("./data/manifest.json");
  const clips = manifest.clips || [];
  for (const c of clips) {
    const opt = document.createElement("option");
    opt.value = c.id;
    opt.textContent = c.title || c.id;
    clipSelect.appendChild(opt);
  }

  let run = null;
  let selected = null;

  function syncScaleLabel() {
    scaleVal.textContent = `${scale.value} m/s`;
  }
  syncScaleLabel();
  scale.addEventListener("input", () => {
    syncScaleLabel();
    if (run) {
      setMeta(meta, selected, run, Number(scale.value));
      drawMap(canvas, run, selected, Number(scale.value));
    }
  });

  function setActiveButton(eventId) {
    const kids = [...timeline.querySelectorAll(".seg")];
    for (const b of kids) b.classList.remove("active");
    const btn = kids.find((b) => b.textContent.startsWith(`#${eventId} `));
    if (btn) btn.classList.add("active");
  }

  async function loadClip(id) {
    const c = clips.find((x) => x.id === id) ?? clips[0];
    if (!c) return;
    selected = null;
    run = await loadJSON(`./${c.resultUrl}`);
    video.src = `./${c.videoUrl}`;
    buildTimeline(timeline, run, (e) => {
      selected = e;
      video.currentTime = e.start_s ?? 0;
      setMeta(meta, selected, run, Number(scale.value));
      drawMap(canvas, run, selected, Number(scale.value));
      setActiveButton(e.id);
    });
    setMeta(meta, null, run, Number(scale.value));
    drawMap(canvas, run, null, Number(scale.value));
  }

  clipSelect.addEventListener("change", () => loadClip(clipSelect.value));
  if (clips.length) await loadClip(clips[0].id);

  video.addEventListener("timeupdate", () => {
    if (!run) return;
    const t = video.currentTime;
    const events = run.events || [];
    const e = events.find((x) => t >= (x.start_s ?? 0) && t <= (x.end_s ?? -1));
    if (e && (!selected || selected.id !== e.id)) {
      selected = e;
      setMeta(meta, selected, run, Number(scale.value));
      drawMap(canvas, run, selected, Number(scale.value));
      setActiveButton(e.id);
    }
  });
}

main().catch((e) => {
  document.body.innerHTML = `<pre style="padding:16px;white-space:pre-wrap;color:#fff;background:#000">${String(e.stack || e)}</pre>`;
});

