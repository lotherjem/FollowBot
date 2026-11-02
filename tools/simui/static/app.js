'use strict';

/* CarryBot simulator front end.
 *
 * Everything shown here comes from the Python stack over HTTP: the costmap is
 * carrybot_nav's own cost layer, the path is what A* returned, and in
 * full-stack mode the pose is integrated from decoded encoder counts. Nothing
 * is simulated in the browser -- if it were, it would drift from the code it
 * claims to be showing. */

const RESOLUTION = 0.05;                 // metres per cell, matches PlannerConfig

const state = {
  telemetry: null,
  params: null,
  runId: -1,
  worldVersion: -1,
  costVersion: -1,
  worldImage: null,
  costImage: null,
  tool: 'goal',
  painting: false,
  drawingTarget: false,
  targetDraft: [],
  running: false,
  history: { plan: [], speed: [] },
  centre: null,
};

const $ = (id) => document.getElementById(id);

/* ------------------------------------------------------------------ api */

async function post(action, body = {}) {
  const res = await fetch('/api/cmd', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(Object.assign({ action }, body)),
  });
  return res.json();
}

function collectParams() {
  const out = {};
  document.querySelectorAll('[data-rebuild], [data-live], [data-live-nokick]')
    .forEach((el) => {
      out[el.id] = el.type === 'checkbox' ? el.checked : el.value;
    });
  out.mode = document.querySelector('#mode-group .active').dataset.mode;
  return out;
}

const rebuild = () => post('configure', { params: collectParams() });
const retune = () => post('tune', { params: collectParams() });

/* --------------------------------------------------------------- inputs */

function syncLabels() {
  const fmt = {
    sensor_range: (v) => `${(+v).toFixed(1)} m`,
    sensor_fov: (v) => (+v >= 359.5 ? 'all round' : `${(+v).toFixed(0)}\u00b0`),
    head_max_angle: (v) => `\u00b1${(+v).toFixed(0)}\u00b0`,
    target_speed: (v) => `${(+v).toFixed(2)} m/s`,
    target_standoff: (v) => `${(+v).toFixed(2)} m`,
    target_predict: (v) => (+v <= 0 ? 'off' : `${(+v).toFixed(1)} s`),
    target_pursue: (v) => (+v <= 0 ? 'stay put' : `${(+v * 100).toFixed(0)}% of the way`),
    target_radius: (v) => `${(+v * 100).toFixed(0)} cm`,
    speed: (v) => `${(+v).toFixed(2)}x`,
    wheel_error: (v) => `${((+v - 1) * 100).toFixed(0)}%`,
    track_error: (v) => `${((+v - 1) * 100).toFixed(0)}%`,
    skew: (v) => `${(+v * 100).toFixed(0)}%`,
    heuristic_weight: (v) => (+v).toFixed(2),
    unknown_cost: (v) => (+v).toFixed(0),
    turn_cost: (v) => (+v).toFixed(1),
    safety_margin: (v) => `${(+v * 1000).toFixed(0)} mm`,
    search_radius: (v) => `${(+v).toFixed(1)} m`,
    lookahead: (v) => `${(+v * 100).toFixed(0)} cm`,
    max_linear: (v) => `${(+v).toFixed(2)} m/s`,
    max_angular: (v) => `${(+v).toFixed(1)} rad/s`,
    brush: (v) => `${(+v * 100).toFixed(0)} cm`,
    zoom: (v) => `${(+v).toFixed(1)}x`,
  };
  for (const [id, format] of Object.entries(fmt)) {
    const input = $(id), out = $(`${id}-out`);
    if (input && out) out.textContent = format(input.value);
  }
  checkHeuristicWeight();
}

/* The one piece of judgement baked into the UI. A* charges
 * 1 + cost_weight x cell_cost per step but the heuristic only knows distance,
 * so a weight below the worst step multiplier stops it being goal-directed and
 * it flood-fills -- 20x slower on the same scene. Worth saying before the user
 * drags the slider there and concludes the planner is slow. */
function checkHeuristicWeight() {
  const weight = +$('heuristic_weight').value;
  const unknown = +$('unknown_cost').value;
  const worst = 1 + 0.008 * Math.max(unknown, 120);
  const box = $('weight-warning');
  if (weight < worst - 0.001) {
    box.hidden = false;
    box.textContent = `Below ${worst.toFixed(2)}, A* stops being goal-directed `
      + `and floods. Expect search time to jump by 10-20x.`;
  } else {
    box.hidden = true;
  }
}

function applyParams(params) {
  state.params = params;
  for (const [key, value] of Object.entries(params)) {
    const el = $(key);
    if (!el || document.activeElement === el) continue;
    if (el.type === 'checkbox') el.checked = !!value;
    else el.value = value;
  }
  document.querySelectorAll('#mode-group button').forEach((b) => {
    b.classList.toggle('active', b.dataset.mode === params.mode);
  });
  const full = params.mode === 'fullstack';
  $('drive-section').classList.toggle('disabled', !full);
  $('speed-row').style.display = full ? 'none' : 'block';
  $('drive-readouts').style.display = full ? 'block' : 'none';
  $('mode-hint').textContent = full
    ? 'Motors, encoders and the velocity loop are in the loop. Locked to real time.'
    : 'The robot moves exactly as commanded, so this isolates the planner. Can run faster than real time.';
  syncLabels();
}

function wireInputs() {
  document.querySelectorAll('[data-rebuild]').forEach((el) => {
    el.addEventListener('change', () => { syncLabels(); rebuild(); });
    el.addEventListener('input', syncLabels);
  });
  document.querySelectorAll('[data-live], [data-live-nokick]').forEach((el) => {
    el.addEventListener('input', () => { syncLabels(); retune(); });
  });

  $('btn-run').addEventListener('click', () => post(state.running ? 'pause' : 'run'));
  $('btn-step').addEventListener('click', () => post('step'));
  $('btn-reset').addEventListener('click', () => post('reset'));
  $('btn-random').addEventListener('click', () => post('randomise'));
  $('btn-clear-queue').addEventListener('click', () => post('clear_queue'));
  $('btn-clear-target').addEventListener('click', () => post('clear_target'));

  document.querySelectorAll('#mode-group button').forEach((b) => {
    b.addEventListener('click', () => {
      document.querySelectorAll('#mode-group button')
        .forEach((x) => x.classList.remove('active'));
      b.classList.add('active');
      rebuild();
    });
  });

  document.querySelectorAll('#tool-group button').forEach((b) => {
    b.addEventListener('click', () => {
      document.querySelectorAll('#tool-group button')
        .forEach((x) => x.classList.remove('active'));
      b.classList.add('active');
      state.tool = b.dataset.tool;
    });
  });

  ['brush', 'zoom'].forEach((id) => $(id).addEventListener('input', syncLabels));
  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    if (e.code === 'Space') { e.preventDefault(); post(state.running ? 'pause' : 'run'); }
    if (e.key === 's') post('step');
    if (e.key === 'r') post('reset');
  });
}

/* ---------------------------------------------------------------- canvas */

function view(canvas) {
  const t = state.telemetry;
  const worldSize = (state.params ? state.params.size : 160) * RESOLUTION;
  const zoom = +$('zoom').value;
  const span = worldSize / zoom;

  let cx = worldSize / 2, cy = worldSize / 2;
  if ($('follow').checked && t) { cx = t.true_pose[0]; cy = t.true_pose[1]; }

  const scale = Math.min(canvas.width, canvas.height) / span;
  return {
    scale,
    x0: cx - canvas.width / (2 * scale),
    y0: cy - canvas.height / (2 * scale),
    toCanvas(wx, wy) {
      return [(wx - this.x0) * this.scale, canvas.height - (wy - this.y0) * this.scale];
    },
    toWorld(px, py) {
      return [this.x0 + px / this.scale, this.y0 + (canvas.height - py) / this.scale];
    },
  };
}

function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.round(rect.width * dpr));
  const h = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
}

function drawImageWorld(ctx, v, img, x0, y0, w, h) {
  // Row 0 of the array is the *lowest* y, but drawImage puts row 0 at the top
  // and canvas y grows downward. Without the flip the map is mirrored about
  // its horizontal centre, which looks plausible on a symmetric world and is
  // instantly obvious the moment you overlay the robot on it.
  const [px, py] = v.toCanvas(x0, y0);      // bottom-left of the destination
  ctx.save();
  ctx.imageSmoothingEnabled = false;
  ctx.translate(px, py);
  ctx.scale(1, -1);
  ctx.drawImage(img, 0, 0, w * v.scale, h * v.scale);
  ctx.restore();
}

function drawPolyline(ctx, v, points, colour, width, alpha = 1) {
  if (!points || points.length < 2) return;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = colour;
  ctx.lineWidth = width;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.beginPath();
  points.forEach((p, i) => {
    const [x, y] = v.toCanvas(p[0], p[1]);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();
  ctx.restore();
}

function drawRobot(ctx, v, pose, colour) {
  const [x, y] = v.toCanvas(pose[0], pose[1]);
  const size = 0.2 * v.scale;
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(-pose[2]);
  ctx.fillStyle = colour;
  ctx.globalAlpha = 0.85;
  ctx.fillRect(-size / 2, -size / 2, size, size);
  ctx.globalAlpha = 1;
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.lineTo(size * 1.1, 0);
  ctx.stroke();
  ctx.restore();
}

function drawTarget(ctx, v, t) {
  const path = t.target_path || [];
  if (path.length > 1) {
    const closed = state.params && state.params.target_loop === 'loop';
    const pts = closed ? path.concat([path[0]]) : path;
    ctx.save();
    ctx.setLineDash([6, 5]);
    drawPolyline(ctx, v, pts, '#c27bff', 1.6, 0.75);
    ctx.restore();
  }
  if (!t.target) return;

  const [x, y] = v.toCanvas(t.target[0], t.target[1]);
  const r = Math.max(3, (state.params ? +state.params.target_radius : 0.22) * v.scale);

  // Colour by whether the robot can currently see them. Distinguishing "the
  // follow logic is wrong" from "the person is behind a wall" at a glance is
  // most of what this view is for.
  const tracked = t.follow_state === 'FOLLOWING' || t.follow_state === 'HOLDING';
  const chasing = t.follow_state === 'PURSUING';
  ctx.save();
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = tracked ? 'rgba(194,123,255,.85)' : 'rgba(194,123,255,.28)';
  ctx.fill();
  ctx.lineWidth = 2;
  ctx.strokeStyle = tracked ? '#e3c7ff' : 'rgba(194,123,255,.6)';
  ctx.stroke();

  if (tracked || chasing) {
    const [rx, ry] = v.toCanvas(t.true_pose[0], t.true_pose[1]);
    ctx.beginPath();
    ctx.moveTo(rx, ry);
    ctx.lineTo(x, y);
    ctx.setLineDash([3, 4]);
    ctx.strokeStyle = chasing ? 'rgba(255,160,60,.6)' : 'rgba(194,123,255,.5)';
    ctx.lineWidth = 1.2;
    ctx.stroke();
  }
  ctx.restore();

  // While chasing, show the spot the robot has committed to reaching. It is
  // usually nowhere near the person any more, and seeing that is the whole
  // point -- "why is it driving over there" has an answer on screen.
  if (chasing && t.goal) {
    const [gx, gy] = v.toCanvas(t.goal[0], t.goal[1]);
    ctx.save();
    ctx.beginPath();
    ctx.arc(gx, gy, 6, 0, Math.PI * 2);
    ctx.strokeStyle = '#ffa03c';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 3]);
    ctx.stroke();
    ctx.restore();
  }
}

function drawMarker(ctx, v, point, colour, glyph) {
  const [x, y] = v.toCanvas(point[0], point[1]);
  ctx.save();
  ctx.fillStyle = colour;
  ctx.font = '600 16px system-ui';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(glyph, x, y);
  ctx.restore();
}

function renderWorld() {
  const canvas = $('canvas-world');
  fitCanvas(canvas);
  const ctx = canvas.getContext('2d');
  const t = state.telemetry;
  const v = view(canvas);

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (state.worldImage) {
    const side = (state.params ? state.params.size : 160) * RESOLUTION;
    drawImageWorld(ctx, v, state.worldImage, -RESOLUTION / 2, -RESOLUTION / 2, side, side);
  }
  if (!t) return;

  if ($('layer-sensor').checked) {
    const [x, y] = v.toCanvas(t.true_pose[0], t.true_pose[1]);
    ctx.save();
    ctx.strokeStyle = 'rgba(78,161,255,.22)';
    ctx.lineWidth = 1;
    // Draw the actual cone, not a circle. A 75-degree camera dressed up as
    // a 360-degree ring is the single most misleading thing this view could
    // show: it makes "the planner ignored the wall behind it" look like a
    // planner bug rather than a robot that never saw the wall.
    //
    // The cone follows the HEAD, not the body. Canvas y runs downwards while
    // the world's runs up, so every world angle is negated on the way in --
    // head yaw included, or the cone swings the wrong way and the picture
    // disagrees with the costmap beside it.
    const reach = (state.params.sensor_range || 2) * v.scale;
    const fov = (state.params.sensor_fov === undefined ? 360 : +state.params.sensor_fov);
    const heading = -((t.true_pose[2] || 0) + (t.head_yaw || 0));
    ctx.beginPath();
    if (fov >= 359.5) {
      ctx.arc(x, y, reach, 0, Math.PI * 2);
    } else {
      const half = (fov * Math.PI) / 360;
      ctx.moveTo(x, y);
      ctx.arc(x, y, reach, heading - half, heading + half);
      ctx.closePath();
      ctx.fillStyle = 'rgba(78,161,255,.07)';
      ctx.fill();
    }
    ctx.stroke();
    ctx.restore();
  }
  if ($('layer-target').checked) drawTarget(ctx, v, t);
  if ($('layer-trail').checked) drawPolyline(ctx, v, t.trail_true, '#4ea1ff', 1.8);
  if ($('layer-odom').checked && t.mode === 'fullstack') {
    drawPolyline(ctx, v, t.trail_odom, '#ff9f45', 1.4, 0.85);
  }
  if ($('layer-path').checked) drawPolyline(ctx, v, t.path, '#ffd166', 2.2, 0.95);

  (t.queue || []).forEach((p, i) => drawMarker(ctx, v, p, '#8a94a0', String(i + 1)));
  drawMarker(ctx, v, t.start, '#63d297', '■');
  drawMarker(ctx, v, t.goal, '#ffd166', '★');
  if (t.mode === 'fullstack' && $('layer-odom').checked) {
    drawRobot(ctx, v, t.odom_pose, '#ff9f45');
  }
  drawRobot(ctx, v, t.true_pose, '#4ea1ff');
}

function renderMap() {
  const canvas = $('canvas-map');
  fitCanvas(canvas);
  const ctx = canvas.getContext('2d');
  const t = state.telemetry;
  const v = view(canvas);

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (state.costImage && t) {
    // The costmap lives in the planner's frame, anchored where the robot
    // started. Offset it into world coordinates so both panels line up.
    const [oy, ox] = [t.map_origin[1], t.map_origin[0]];
    const [rows, cols] = t.map_shape;
    drawImageWorld(
      ctx, v, state.costImage,
      ox + t.start[0] - RESOLUTION / 2,
      oy + t.start[1] - RESOLUTION / 2,
      cols * RESOLUTION, rows * RESOLUTION,
    );
  }
  if (!t) return;

  if ($('layer-path').checked) drawPolyline(ctx, v, t.path, '#ffd166', 2.2);
  drawMarker(ctx, v, t.goal, '#ffd166', '★');
  drawRobot(ctx, v, t.mode === 'fullstack' ? t.odom_pose : t.true_pose, '#4ea1ff');
}

/* ------------------------------------------------------------ pointer io */

function wireCanvas() {
  const canvas = $('canvas-world');

  const locate = (event) => {
    const rect = canvas.getBoundingClientRect();
    const px = (event.clientX - rect.left) * (canvas.width / rect.width);
    const py = (event.clientY - rect.top) * (canvas.height / rect.height);
    return view(canvas).toWorld(px, py);
  };

  /* The target path is drawn rather than clicked: dragging lays down a
     polyline, simplified as it goes so a slow hand does not produce four
     hundred waypoints a centimetre apart. It is sent on pointerup, once,
     because a path that updates mid-drag restarts the target on every
     sample and the robot never gets a coherent thing to follow. */
  const pushTargetPoint = (event, force) => {
    const [x, y] = locate(event);
    const pts = state.targetDraft;
    if (!force && pts.length) {
      const last = pts[pts.length - 1];
      if (Math.hypot(x - last[0], y - last[1]) < 0.18) return;
    }
    pts.push([x, y]);
  };

  const act = (event) => {
    const [x, y] = locate(event);
    if (state.tool === 'goal') post('goal', { x, y });
    else if (state.tool === 'wall') post('paint', { x, y, radius: +$('brush').value, value: 'wall' });
    else if (state.tool === 'erase') post('paint', { x, y, radius: +$('brush').value, value: 'free' });
    else if (state.tool === 'queue') post('queue', { x, y });
  };

  canvas.addEventListener('pointerdown', (e) => {
    if (state.tool === 'none') return;
    canvas.setPointerCapture(e.pointerId);
    if (state.tool === 'target') {
      state.drawingTarget = true;
      state.targetDraft = [];
      pushTargetPoint(e, true);
      return;
    }
    state.painting = state.tool === 'wall' || state.tool === 'erase';
    act(e);
  });
  canvas.addEventListener('pointermove', (e) => {
    if (state.drawingTarget) pushTargetPoint(e, false);
    else if (state.painting) act(e);
  });
  const finish = (e) => {
    if (state.drawingTarget) {
      state.drawingTarget = false;
      if (e) pushTargetPoint(e, true);
      if (state.targetDraft.length >= 2) {
        post('target', { points: state.targetDraft });
      }
      state.targetDraft = [];
    }
    state.painting = false;
  };
  canvas.addEventListener('pointerup', finish);
  canvas.addEventListener('pointercancel', () => finish(null));
}

/* -------------------------------------------------------------- readouts */

const metres = (v) => `${(v * 1000).toFixed(0)} mm`;
const degrees = (v) => `${(v * 57.2958).toFixed(1)}°`;

function poseText(p) {
  return `${p[0].toFixed(2)}, ${p[1].toFixed(2)} m  ${degrees(p[2])}`;
}

function setAlert(id, on) { $(id).classList.toggle('alert', !!on); }

function renderReadouts() {
  const t = state.telemetry;
  if (!t) return;

  const pill = $('status-pill');
  pill.textContent = t.reached ? 'ARRIVED' : t.plan_status;
  pill.className = 'pill ' + ({
    PATH_OK: 'ok', AT_GOAL: 'ok', FRONTIER: 'warn', PARTIAL: 'warn',
    NO_PATH: 'bad', IDLE: '',
  }[t.reached ? 'AT_GOAL' : t.plan_status] || '');

  $('clock-label').textContent = `${t.sim_time.toFixed(1)} s sim`;
  $('btn-run').textContent = t.running ? 'Pause' : 'Run';
  $('btn-run').classList.toggle('running', t.running);
  state.running = t.running;

  $('r-status').textContent = t.plan_status + (t.reused ? ' (reused)' : '');
  $('r-plan-ms').textContent = `${t.plan_ms.toFixed(1)} ms`;
  $('r-expansions').textContent = t.expansions.toLocaleString();
  $('r-plan-seq').textContent = t.plan_seq.toLocaleString();
  $('r-known').textContent = `${(t.known_fraction * 100).toFixed(1)}%`;
  $('r-arrivals').textContent = t.arrivals;
  $('queue-label').textContent = t.queued
    ? `${t.queued} waypoint${t.queued === 1 ? '' : 's'} queued`
    : 'queue empty';

  $('r-true').textContent = poseText(t.true_pose);
  $('r-odom').textContent = poseText(t.odom_pose);
  $('r-drift').textContent = `${metres(t.drift)} / ${degrees(t.heading_drift)}`;
  $('r-distance').textContent = `${t.distance.toFixed(2)} m`;
  $('r-collisions').textContent = t.collisions;
  setAlert('drift-stat', t.drift > 0.1);
  setAlert('contact-stat', t.collisions > 0);

  if (t.mode === 'fullstack') {
    $('r-wheels').textContent = `${t.left_speed.toFixed(2)} / ${t.right_speed.toFixed(2)} rad/s`;
    $('r-duty').textContent = `${t.left_duty.toFixed(0)} / ${t.right_duty.toFixed(0)} %`;
    $('r-counts').textContent = `${t.left_counts} / ${t.right_counts}`;
    $('r-enc').textContent = `${t.encoder_errors} / ${t.desyncs}`;
    $('r-overruns').textContent = t.overruns;
    setAlert('enc-stat', t.encoder_errors + t.desyncs > 0);
    setAlert('ovr-stat', t.overruns > 20);
  }

  const box = $('error-box');
  box.hidden = !t.error;
  if (t.error) box.textContent = t.error;

  if (t.plan_ms > 0) push(state.history.plan, t.plan_ms, 220);
  push(state.history.speed, Math.abs(t.linear), 220);
  spark('spark-plan', state.history.plan, '#ffd166');
  spark('spark-speed', state.history.speed, '#4ea1ff');
}

function push(series, value, cap) {
  series.push(value);
  if (series.length > cap) series.shift();
}

function spark(id, series, colour) {
  const canvas = $(id);
  if (!canvas) return;
  fitCanvas(canvas);
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (series.length < 2) return;

  const top = Math.max(...series, 1e-6) * 1.15;
  const stepX = canvas.width / (series.length - 1);
  ctx.strokeStyle = colour;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  series.forEach((value, i) => {
    const x = i * stepX;
    const y = canvas.height - (value / top) * (canvas.height - 6) - 3;
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();

  ctx.fillStyle = '#6d7885';
  ctx.font = '10px ui-monospace, monospace';
  ctx.fillText(top.toFixed(top < 10 ? 2 : 0), 4, 11);
}

/* ------------------------------------------------------------------ loop */

async function loadImage(url) {
  const res = await fetch(url, { cache: 'no-store' });
  return createImageBitmap(await res.blob());
}

let mapBusy = false;

async function poll() {
  try {
    const res = await fetch('/api/state', { cache: 'no-store' });
    const t = await res.json();
    state.telemetry = t;
    if (t.params) applyParams(t.params);

    if (t.run_id !== state.runId) {
      state.runId = t.run_id;
      state.history.plan.length = 0;
      state.history.speed.length = 0;
      state.worldVersion = -1;
      state.costVersion = -1;
    }
    if (t.world_version !== state.worldVersion) {
      state.worldVersion = t.world_version;
      state.worldImage = await loadImage('/api/world.png');
    }
    if (t.costmap_version !== state.costVersion && !mapBusy) {
      mapBusy = true;
      state.costVersion = t.costmap_version;
      loadImage('/api/costmap.png')
        .then((img) => { state.costImage = img; })
        .finally(() => { mapBusy = false; });
    }
    renderReadouts();
  } catch (err) {
    /* the server is restarting a scenario; the next tick will pick it up */
  }
}

function frame() {
  renderWorld();
  renderMap();
  requestAnimationFrame(frame);
}

wireInputs();
wireCanvas();
syncLabels();
poll();
setInterval(poll, 80);
requestAnimationFrame(frame);
