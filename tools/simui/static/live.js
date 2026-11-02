'use strict';

/* CarryBot live interface.
 *
 * Everything here is display and intent. The robot decides what actually
 * reaches the motors: it starts disarmed, disarms itself when this page stops
 * polling, brakes when a manual command is let go or not refreshed, and
 * latches on STOP.
 * This page can only ask. That is deliberate -- a browser is the least
 * reliable component in the loop, so it is given the least authority. */

const state = {
  t: null,
  lastOk: 0,
  mode: 'manual',
  pitchDir: 1,
  keys: new Set(),
  pad: null,            // {x, y} in -1..1 while the pad is held
  lastTeleopZero: true,
  teleopSeq: 0,         // orders the commands: the robot drops any older than the newest
  cameraSeq: -1,
  cameraBusy: false,
  depthBusy: false,
  costVersion: -1,
  costImage: null,
  mapBusy: false,
  speed: [],
};

const $ = (id) => document.getElementById(id);

/* Which open page this is. The robot listens only to the page that armed it:
 * its silence disarms, and only it may drive. Made fresh on every load, so a
 * reload is a new page -- and a reload of the driving page disarms. Built
 * from getRandomValues because crypto.randomUUID only exists on HTTPS pages,
 * and this is served over plain HTTP on the LAN. */
const CLIENT_ID = Array.from(crypto.getRandomValues(new Uint8Array(16)),
  (b) => b.toString(16).padStart(2, '0')).join('');

async function api(url, options = {}) {
  const headers = Object.assign({ 'X-CarryBot-Client': CLIENT_ID }, options.headers || {});
  const res = await fetch(url, Object.assign({ cache: 'no-store' }, options, { headers }));
  if (res.status === 401) {
    location.replace('/login');
    throw new Error('login required');
  }
  return res;
}

async function post(action, body = {}) {
  try {
    const res = await api('/api/cmd', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(Object.assign({ action }, body)),
    });
    const reply = await res.json();
    if (!reply.ok && reply.error) flash(reply.error);
    return reply;
  } catch (err) {
    flash('could not reach the robot');
    return { ok: false };
  }
}

let flashTimer = null;
function flash(text) {
  const box = $('error-box');
  box.hidden = false;
  box.textContent = text;
  clearTimeout(flashTimer);
  flashTimer = setTimeout(() => { if (!(state.t && state.t.error)) box.hidden = true; }, 5000);
}

/* ------------------------------------------------------------- settings */

const formats = {
  max_linear: (v) => `${(+v).toFixed(2)} m/s`,
  max_angular: (v) => `${(+v).toFixed(1)} rad/s`,
  standoff: (v) => `${(+v).toFixed(2)} m`,
  predict: (v) => (+v <= 0 ? 'off' : `${(+v).toFixed(1)} s`),
  pursue: (v) => (+v <= 0 ? 'stay put' : `${(+v * 100).toFixed(0)}% of the way`),
  head_max_angle: (v) => `±${(+v).toFixed(0)}°`,
  heuristic_weight: (v) => (+v).toFixed(2),
  unknown_cost: (v) => (+v).toFixed(0),
  lookahead: (v) => `${(+v * 100).toFixed(0)} cm`,
  zoom: (v) => `${(+v).toFixed(1)}x`,
  'head-angle': (v) => `${+v > 0 ? 'left ' : +v < 0 ? 'right ' : ''}${Math.abs(+v).toFixed(0)}°`,
};

function syncLabels() {
  for (const [id, fmt] of Object.entries(formats)) {
    const input = $(id), out = $(`${id}-out`);
    if (input && out) out.textContent = fmt(input.value);
  }
}

function collectParams() {
  const out = {};
  document.querySelectorAll('[data-live]').forEach((el) => {
    out[el.id] = el.type === 'checkbox' ? el.checked : el.value;
  });
  return out;
}

function applyParams(params) {
  for (const [key, value] of Object.entries(params)) {
    const el = $(key);
    if (!el || !el.hasAttribute('data-live') || document.activeElement === el) continue;
    if (el.type === 'checkbox') el.checked = !!value;
    else el.value = value;
  }
  const limit = +params.head_max_angle || 60;
  $('head-angle').min = -limit;
  $('head-angle').max = limit;
  syncLabels();
}

const MODE_HINTS = {
  manual: 'Drive with W A S D or the arrow keys, or drag the pad. Every command '
    + 'expires in a third of a second, so letting go stops it.',
  goto: 'Click the map and it plans a route there around what the camera has '
    + 'seen. Unknown space counts as passable unless you untick it below.',
  follow: 'Tracks the person in view and keeps the standoff behind them. If '
    + 'they are lost it carries their last velocity, then chases part of the way.',
};

function showMode(mode) {
  state.mode = mode;
  document.querySelectorAll('#mode-group button').forEach((b) => {
    b.classList.toggle('active', b.dataset.mode === mode);
  });
  $('mode-hint').textContent = MODE_HINTS[mode] || '';
  $('pad').hidden = mode !== 'manual';
  $('btn-clear-goal').hidden = mode !== 'goto';
  $('map-hint').textContent = mode === 'goto'
    ? 'Click the map to send it there.'
    : 'Switch to "Go to" to send it somewhere by clicking the map.';
}

function wireInputs() {
  document.querySelectorAll('[data-live]').forEach((el) => {
    const send = () => { syncLabels(); post('tune', { params: collectParams() }); };
    el.addEventListener(el.type === 'checkbox' ? 'change' : 'input', send);
  });
  $('zoom').addEventListener('input', syncLabels);

  document.querySelectorAll('#mode-group button').forEach((b) => {
    b.addEventListener('click', async () => {
      const reply = await post('mode', { mode: b.dataset.mode });
      if (reply.ok) showMode(b.dataset.mode);
    });
  });

  $('btn-arm').addEventListener('click', () => {
    post(state.t && state.t.armed ? 'disarm' : 'arm');
  });
  $('btn-estop').addEventListener('click', () => post('estop'));
  $('btn-clear-estop').addEventListener('click', () => post('clear_estop'));
  $('btn-clear-goal').addEventListener('click', () => post('clear_goal'));
  $('btn-reset').addEventListener('click', () => post('reset'));
  $('btn-logout').addEventListener('click', async () => {
    await api('/api/logout', { method: 'POST' }).catch(() => {});
    location.replace('/login');
  });

  document.querySelectorAll('#head-group button').forEach((b) => {
    b.addEventListener('click', async () => {
      const manual = b.dataset.head === 'manual';
      const reply = await post('head', { angle: manual ? +$('head-angle').value : null });
      if (reply.ok) showHead(manual);
    });
  });
  $('head-angle').addEventListener('input', () => {
    syncLabels();
    post('head', { angle: +$('head-angle').value });
  });

  // Space is STOP from anywhere on the page, including a focused slider --
  // the one key that must never be swallowed by whatever has focus.
  document.addEventListener('keydown', (e) => {
    // Inside the switch-on dialog the motors are by definition off, so Space
    // can do its ordinary job of ticking the box instead of latching STOP.
    const inModal = !$('motor-modal').hidden;
    if (e.code === 'Space' && !inModal) { e.preventDefault(); post('estop'); return; }
    if (inModal) return;
    if (e.key === 'Escape') {
      if (!$('motor-modal').hidden) { closeMotorModal(); return; }
      post('disarm');
      return;
    }
    if (e.target.tagName === 'INPUT' && e.target.type !== 'range') return;
    const k = driveKey(e);
    if (k) {
      e.preventDefault();
      // Act on a new key at once rather than at the next repeat; auto-repeat
      // keydowns for a key already held change nothing.
      if (!state.keys.has(k)) { state.keys.add(k); sendTeleop(); }
    }
  });
  document.addEventListener('keyup', (e) => {
    const k = driveKey(e);
    // Letting go is sent the instant it happens: the robot brakes on it.
    if (k) { state.keys.delete(k); sendTeleop(); }
  });
  window.addEventListener('blur', () => { state.keys.clear(); state.pad = null; sendTeleop(true); });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) { state.keys.clear(); state.pad = null; sendTeleop(true); }
  });
}

function showHead(manual) {
  document.querySelectorAll('#head-group button').forEach((b) => {
    b.classList.toggle('active', (b.dataset.head === 'manual') === manual);
  });
  $('head-angle-row').hidden = !manual;
}

/* ----------------------------------------------------------------- motors */

/* The switch is only ever a request. The robot re-checks everything when it
 * arrives and refuses if anything is wrong, so a stale page cannot talk it
 * into switching on. Switching OFF is never blocked and never asks. */

function motorState() {
  const t = state.t;
  const hw = (t && t.hardware) || {};
  return {
    active: t && t.drive_kind === 'hardware',
    sim: t && t.drive_kind === 'sim',
    blockers: hw.blockers || [],
    warnings: hw.warnings || [],
    detail: hw.detail || '',
    known: !!(t && t.hardware && 'blockers' in t.hardware),
  };
}

function fillList(el, items) {
  el.replaceChildren(...items.map((text) => {
    const li = document.createElement('li');
    li.textContent = text;
    return li;
  }));
  el.hidden = items.length === 0;
}

function renderMotors(t) {
  const m = motorState();
  const button = $('btn-motors');
  const blocked = !m.active && m.blockers.length > 0;
  button.classList.toggle('on', m.active);
  button.classList.toggle('blocked', blocked);
  button.setAttribute('aria-checked', m.active ? 'true' : 'false');
  button.setAttribute('aria-disabled', blocked ? 'true' : 'false');
  $('motors-state').textContent = m.active ? 'MOTORS ON' : 'MOTORS OFF';
  $('motors-sub').textContent = m.active ? (t.armed ? 'armed — will move' : 'hardware, disarmed')
    : m.sim ? 'simulated motors' : blocked ? 'blocked — see why' : 'dry run';

  const count = $('motors-count');
  const issues = blocked ? m.blockers.length : 0;
  count.hidden = issues === 0;
  count.textContent = issues;

  $('tip-head').textContent = !m.known ? 'Checking the hardware…'
    : m.active ? 'The real motors are on. Click to switch them off — that always works.'
    : blocked ? `Can't switch the motors on (${m.blockers.length} problem${m.blockers.length === 1 ? '' : 's'}):`
    : 'Ready. Click to switch the real motors on — it asks first, and stays disarmed.';
  fillList($('tip-blockers'), m.active ? [] : m.blockers);
  fillList($('tip-warnings'), m.warnings);
  $('tip-warn-head').hidden = m.warnings.length === 0;
  $('tip-detail').textContent = m.detail;
  $('tip-detail').hidden = !m.detail;
  document.body.classList.toggle('hot', m.active);
}

function openMotorModal() {
  const m = motorState();
  $('modal-speed').textContent =
    `${(+$('max_linear').value).toFixed(2)} m/s, ${(+$('max_angular').value).toFixed(1)} rad/s`;
  fillList($('modal-warnings'), m.warnings);
  $('modal-warn-box').hidden = m.warnings.length === 0;
  $('modal-clear').checked = false;
  $('modal-confirm').disabled = true;
  $('motor-modal').hidden = false;
  $('modal-cancel').focus();
}

function closeMotorModal() { $('motor-modal').hidden = true; }

function wireMotors() {
  const box = $('motors');
  $('btn-motors').addEventListener('click', async () => {
    const m = motorState();
    if (m.active) { post('hardware', { enable: false }); return; }
    if (m.sim || m.blockers.length || !m.known) {
      // Phones cannot hover, so a tap on a blocked switch pins the list open.
      box.classList.add('pinned', 'shake');
      setTimeout(() => box.classList.remove('shake'), 400);
      return;
    }
    openMotorModal();
  });
  document.addEventListener('click', (e) => {
    if (!box.contains(e.target)) {
      box.classList.remove('pinned');
      // Focus alone keeps the list open (for keyboard users), and a tapped
      // button keeps focus on a phone -- so let go of it too.
      if (document.activeElement === $('btn-motors')) $('btn-motors').blur();
    }
  });
  $('modal-clear').addEventListener('change', () => {
    $('modal-confirm').disabled = !$('modal-clear').checked;
  });
  $('modal-cancel').addEventListener('click', closeMotorModal);
  $('motor-modal').addEventListener('click', (e) => {
    if (e.target === $('motor-modal')) closeMotorModal();
  });
  $('modal-confirm').addEventListener('click', async () => {
    $('modal-confirm').disabled = true;
    const reply = await post('hardware', { enable: true, confirm: true });
    closeMotorModal();
    if (!reply.ok) box.classList.add('pinned');
  });
}

/* ---------------------------------------------------------------- driving */

function driveKey(e) {
  const map = {
    KeyW: 'up', ArrowUp: 'up', KeyS: 'down', ArrowDown: 'down',
    KeyA: 'left', ArrowLeft: 'left', KeyD: 'right', ArrowRight: 'right',
  };
  return map[e.code];
}

function teleopVector() {
  if (state.pad) return { linear: -state.pad.y, angular: -state.pad.x };
  const k = state.keys;
  return {
    linear: (k.has('up') ? 1 : 0) - (k.has('down') ? 1 : 0),
    angular: (k.has('left') ? 1 : 0) - (k.has('right') ? 1 : 0),
  };
}

function sendTeleop(force = false) {
  if (state.mode !== 'manual') return;
  if (state.t && state.t.control && state.t.control.other) return;   // not ours to drive
  const v = teleopVector();
  const zero = Math.abs(v.linear) < 1e-3 && Math.abs(v.angular) < 1e-3;
  // Keep sending while held -- the robot drops a command that is not renewed
  // within teleop_timeout (0.15 s) -- and send an explicit zero the moment
  // everything is let go, which it brakes on. The sequence number stops a
  // "go" that was overtaken in flight from landing after its "stop".
  if (!zero || !state.lastTeleopZero || force) {
    state.teleopSeq += 1;
    post('teleop', Object.assign({ seq: state.teleopSeq }, v));
  }
  state.lastTeleopZero = zero;
}

function wireRangeCal() {
  $('btn-range-cal').addEventListener('click', () =>
    post('range_cal', { distance: +$('range-distance').value }));
  $('btn-range-clear').addEventListener('click', () => post('range_cal', { clear: true }));
  $('btn-floor-apply').addEventListener('click', async () => {
    const reply = await post('floor_apply');
    if (reply.ok) state.mountDirty = false;        // show what the floor set
  });
  for (const id of ['mount-pitch', 'mount-height']) {
    $(id).addEventListener('input', () => { state.mountDirty = true; });
  }
  // Up or down is a toggle, not a minus sign: phone number pads often have
  // no minus key. A minus typed anyway still works -- it flips the toggle.
  for (const b of document.querySelectorAll('#pitch-dir button')) {
    b.addEventListener('click', (e) => {
      e.preventDefault();
      setPitchDir(+b.dataset.dir);
      state.mountDirty = true;
    });
  }
  $('btn-mount-apply').addEventListener('click', async () => {
    let size = +$('mount-pitch').value;
    if (size < 0) { setPitchDir(-state.pitchDir); size = -size; $('mount-pitch').value = size; }
    const reply = await post('camera_mount', {
      pitch_deg: state.pitchDir * size, height: +$('mount-height').value });
    if (reply.ok) state.mountDirty = false;
  });
}

function setPitchDir(dir) {
  state.pitchDir = dir < 0 ? -1 : 1;
  for (const b of document.querySelectorAll('#pitch-dir button')) {
    b.classList.toggle('active', +b.dataset.dir === state.pitchDir);
  }
}

function renderRangeCal(t) {
  const pinned = $('range-pinned');
  pinned.hidden = !t.range_pinned;
  pinned.textContent = t.range_pinned || '';
  const readings = (t.range_samples || []).map(([z, d]) => `${z} m -> ${d} px`).join(', ');
  $('range-note').textContent = [readings && `Readings: ${readings}.`, t.range_cal]
    .filter(Boolean).join(' ');
  // Start from the robot's own values, and follow them -- but never
  // overwrite something typed and not yet applied.
  if (!state.mountDirty && t.mount_pitch_deg !== undefined) {
    $('mount-pitch').value = Math.abs(t.mount_pitch_deg);
    setPitchDir(t.mount_pitch_deg < 0 ? -1 : 1);
    $('mount-height').value = t.mount_height;
  }
  $('floor-check').textContent = t.floor_check || '';
  $('floor-check').classList.toggle('warn', !!t.floor_off);
  $('btn-floor-apply').hidden = !t.floor_off;
}

function wirePerception() {
  for (const [id, key] of [['sense-terrain', 'terrain'], ['sense-people', 'people']]) {
    $(id).addEventListener('change', async (e) => {
      const want = e.target.checked;
      state.perceptionPending = performance.now();
      const reply = await post('perception', { [key]: want });
      if (!reply.ok) e.target.checked = !want;     // refused: put it back
    });
  }
}

function renderPerception(t) {
  // Do not fight a click that has not reached the robot yet.
  if (!state.perceptionPending || performance.now() - state.perceptionPending > 600) {
    $('sense-terrain').checked = t.terrain_on;
    $('sense-people').checked = t.people_on && t.tracker_installed;
  }
  $('sense-people').disabled = !t.tracker_installed;
  const notes = [];
  if (!t.terrain_on) notes.push('Not mapping obstacles: nothing new is added, and what is already on the map can be crossed at a cost.');
  if (!t.people_on) notes.push('Not looking for people: Follow me is off.');
  $('sense-note').hidden = !notes.length;
  $('sense-note').textContent = notes.join(' ');
}

function wirePad() {
  const pad = $('pad'), knob = $('knob');
  const place = (e) => {
    const r = pad.getBoundingClientRect();
    let x = ((e.clientX - r.left) / r.width) * 2 - 1;
    let y = ((e.clientY - r.top) / r.height) * 2 - 1;
    const len = Math.hypot(x, y);
    if (len > 1) { x /= len; y /= len; }
    state.pad = { x, y };
    knob.style.left = `${50 + x * 40}%`;
    knob.style.top = `${50 + y * 40}%`;
  };
  const release = () => {
    state.pad = null;
    knob.style.left = '50%';
    knob.style.top = '50%';
    sendTeleop();
  };
  pad.addEventListener('pointerdown', (e) => { pad.setPointerCapture(e.pointerId); place(e); sendTeleop(); });
  pad.addEventListener('pointermove', (e) => { if (state.pad) place(e); });
  pad.addEventListener('pointerup', release);
  pad.addEventListener('pointercancel', release);
}

/* -------------------------------------------------------------------- map */

function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.round(rect.width * dpr));
  const h = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
}

function view(canvas) {
  const t = state.t;
  const span = 8 / +$('zoom').value;             // metres across the short side
  let cx = 0, cy = 0;
  if (t && $('follow-robot').checked) { cx = t.pose[0]; cy = t.pose[1]; }
  const scale = Math.min(canvas.width, canvas.height) / span;
  return {
    scale,
    x0: cx - canvas.width / (2 * scale),
    y0: cy - canvas.height / (2 * scale),
    toCanvas(wx, wy) { return [(wx - this.x0) * this.scale, canvas.height - (wy - this.y0) * this.scale]; },
    toWorld(px, py) { return [this.x0 + px / this.scale, this.y0 + (canvas.height - py) / this.scale]; },
  };
}

function drawImageWorld(ctx, v, img, x0, y0, w, h) {
  // Row 0 of the costmap is the lowest y; canvas row 0 is the top. Flip.
  const [px, py] = v.toCanvas(x0, y0);
  ctx.save();
  ctx.imageSmoothingEnabled = false;
  ctx.translate(px, py);
  ctx.scale(1, -1);
  ctx.drawImage(img, 0, 0, w * v.scale, h * v.scale);
  ctx.restore();
}

function polyline(ctx, v, pts, colour, width, alpha = 1) {
  if (!pts || pts.length < 2) return;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = colour;
  ctx.lineWidth = width;
  ctx.lineJoin = ctx.lineCap = 'round';
  ctx.beginPath();
  pts.forEach((p, i) => { const [x, y] = v.toCanvas(p[0], p[1]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.stroke();
  ctx.restore();
}

function drawRobot(ctx, v, pose) {
  const [x, y] = v.toCanvas(pose[0], pose[1]);
  const size = Math.max(8, 0.3 * v.scale);
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(-pose[2]);
  ctx.fillStyle = state.t && state.t.armed ? '#63d297' : '#4ea1ff';
  ctx.globalAlpha = 0.9;
  ctx.fillRect(-size / 2, -size / 2, size, size);
  ctx.globalAlpha = 1;
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.lineTo(size * 1.1, 0);
  ctx.stroke();
  ctx.restore();
}

function renderMap() {
  const canvas = $('canvas-map');
  fitCanvas(canvas);
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const t = state.t;
  if (!t) return;
  const v = view(canvas);
  const res = t.resolution || 0.05;

  if (state.costImage && t.map_shape) {
    const [rows, cols] = t.map_shape;
    drawImageWorld(ctx, v, state.costImage,
      t.map_origin[0] - res / 2, t.map_origin[1] - res / 2, cols * res, rows * res);
  }

  if ($('layer-sensor').checked) {
    // The cone follows the head, and canvas angles run the other way.
    const [x, y] = v.toCanvas(t.pose[0], t.pose[1]);
    const heading = -(t.pose[2] + (t.head_yaw || 0));
    const half = (t.hfov || 1.3) / 2;
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.arc(x, y, (t.sensor_range || 6) * v.scale, heading - half, heading + half);
    ctx.closePath();
    ctx.fillStyle = 'rgba(78,161,255,.06)';
    ctx.strokeStyle = 'rgba(78,161,255,.25)';
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }
  if ($('layer-trail').checked) polyline(ctx, v, t.trail, '#4ea1ff', 1.6, 0.8);
  if ($('layer-path').checked) polyline(ctx, v, t.path, '#ffd166', 2.4);

  if (t.goal) {
    const [gx, gy] = v.toCanvas(t.goal[0], t.goal[1]);
    ctx.save();
    ctx.fillStyle = '#ffd166';
    ctx.font = '600 18px system-ui';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('★', gx, gy);
    ctx.restore();
  }
  if (t.target) {
    const [px, py] = v.toCanvas(t.target[0], t.target[1]);
    ctx.save();
    ctx.beginPath();
    ctx.arc(px, py, Math.max(5, 0.22 * v.scale), 0, Math.PI * 2);
    ctx.fillStyle = t.seen ? 'rgba(194,123,255,.85)' : 'rgba(194,123,255,.3)';
    ctx.fill();
    ctx.strokeStyle = t.predicted ? '#ffa03c' : '#e3c7ff';
    ctx.lineWidth = 2;
    if (t.predicted) ctx.setLineDash([4, 3]);
    ctx.stroke();
    ctx.restore();
  }
  drawRobot(ctx, v, t.pose);
}

function wireMap() {
  const canvas = $('canvas-map');
  canvas.addEventListener('click', (e) => {
    if (state.mode !== 'goto') { flash('Switch to "Go to" to send it somewhere.'); return; }
    const rect = canvas.getBoundingClientRect();
    const px = (e.clientX - rect.left) * (canvas.width / rect.width);
    const py = (e.clientY - rect.top) * (canvas.height / rect.height);
    const [x, y] = view(canvas).toWorld(px, py);
    post('goal', { x, y });
  });
}

/* ---------------------------------------------------------------- images */

async function refreshImage(url, img, empty, busyKey) {
  if (state[busyKey]) return;
  state[busyKey] = true;
  try {
    const res = await api(url);
    if (res.status === 200) {
      const blob = await res.blob();
      const old = img.src;
      img.src = URL.createObjectURL(blob);
      if (old && old.startsWith('blob:')) URL.revokeObjectURL(old);
      empty.hidden = true;
    }
  } catch (err) {
    /* next frame */
  } finally {
    state[busyKey] = false;
  }
}

/* -------------------------------------------------------------- readouts */

const deg = (r) => `${(r * 57.2958).toFixed(1)}°`;
const vel = (p) => `${p[0].toFixed(2)} m/s  ${p[1].toFixed(2)} rad/s`;

function setAlert(id, on) { $(id).classList.toggle('alert', !!on); }

function render() {
  const t = state.t;
  const linked = performance.now() - state.lastOk < 1000;
  $('link-dot').className = 'link-dot ' + (linked ? 'ok' : 'bad');
  $('banner-link').hidden = linked || !t;
  if (!t) return;

  renderMotors(t);
  $('banner-dry').hidden = t.drive_kind === 'hardware';
  $('banner-dry').textContent = t.drive_kind === 'dry'
    ? 'Motors off: every command is computed and shown, and nothing moves. '
      + 'Use the MOTORS switch to drive the robot.'
    : 'Simulated motors: the pose moves but the camera does not, so the map '
      + 'will smear as it drives. For testing the control loops only.';
  $('banner-estop').hidden = !t.estopped;
  // The drive can latch itself (runaway or stall guard); say why, since
  // nobody pressed STOP.
  $('estop-text').textContent = t.estopped && t.disarm_reason && t.disarm_reason !== 'emergency stop'
    ? `Stopped: ${t.disarm_reason}` : 'Emergency stop is latched.';
  $('banner-calib').hidden = t.depth_calibrated;
  $('banner-geometry').hidden = t.geometry_trusted;
  $('banner-geometry').textContent = `The camera's lenses have moved since calibration: ${t.registration}`;

  const control = t.control || {};
  document.body.classList.toggle('viewer', !!control.other);
  $('banner-viewer').hidden = !control.other;
  $('link-dot').title = `connection to the robot · ${control.pages || 1} page${control.pages === 1 ? '' : 's'} open`;
  $('btn-logout').hidden = !t.auth;

  const arm = $('btn-arm');
  arm.textContent = t.armed ? 'Disarm' : 'Arm';
  arm.classList.toggle('armed', t.armed);
  arm.disabled = t.estopped;

  const pill = $('status-pill');
  pill.textContent = t.estopped ? 'E-STOP'
    : t.armed ? (control.other ? 'ARMED ELSEWHERE' : `ARMED · ${t.mode}`) : 'DISARMED';
  pill.className = 'pill ' + (t.estopped ? 'bad' : t.armed ? (control.other ? 'warn' : 'ok') : '');

  if (t.mode !== state.mode) showMode(t.mode);
  showHead(t.head_manual !== null && t.head_manual !== undefined);
  $('head-section').classList.toggle('disabled', !t.head_available);
  $('head-note').hidden = t.head_available;
  document.querySelector('#mode-group [data-mode="follow"]').disabled = !t.follow_available;
  $('follow-note').hidden = t.follow_available;
  $('follow-note').textContent = t.follow_note || '';

  $('r-armed').textContent = t.estopped ? 'E-STOP' : t.armed ? 'armed' : 'disarmed';
  $('r-reason').textContent = t.note || t.disarm_reason || '—';
  $('r-watchdog').textContent = t.watchdog ? 'TRIPPED' : 'ok';
  setAlert('wd-stat', t.watchdog && t.armed);
  $('r-motor-mode').textContent = t.motor_mode;

  $('r-pose').textContent = `${t.pose[0].toFixed(2)}, ${t.pose[1].toFixed(2)} m  ${deg(t.pose[2])}`;
  $('r-wanted').textContent = vel(t.wanted);
  $('r-commanded').textContent = vel(t.commanded);
  $('r-measured').textContent = vel(t.measured);
  $('r-wheels').textContent = `${t.wheels[0].toFixed(2)} / ${t.wheels[1].toFixed(2)} rad/s`;
  $('r-duty').textContent = `${t.duty[0].toFixed(0)} / ${t.duty[1].toFixed(0)} %`;
  $('r-enc').textContent = `${t.encoder_errors} / ${t.desyncs}`;
  $('r-overruns').textContent = t.overruns;
  setAlert('enc-stat', t.encoder_errors + t.desyncs > 0);
  if (t.edge_lag) {
    $('r-edge-lag').textContent = `${Math.round(t.edge_lag[0])} / ${Math.round(t.edge_lag[1])} ms`;
    setAlert('lag-stat', Math.max(t.edge_lag[0], t.edge_lag[1]) > 100);
  }
  setAlert('ovr-stat', t.overruns > 20);

  $('r-fps').textContent = `${t.vision_fps.toFixed(1)} fps`;
  $('r-latency').textContent = t.vision_seq ? `${(t.vision_latency * 1000).toFixed(0)} ms` : '—';
  $('r-frames').textContent = `${t.vision_frames} / ${t.vision_dropped}`;
  renderStages(t.vision_stages || [], t.parallel && t.terrain_on && t.people_on);
  renderPerception(t);
  renderRangeCal(t);
  $('r-obstacles').textContent = t.obstacles;
  $('r-registration').textContent = t.registration;
  setAlert('reg-stat', !t.geometry_trusted);
  $('r-camera').textContent = t.camera || '';

  $('r-follow').textContent = t.follow_state + (t.predicted ? ' (predicted)' : '');
  $('r-target').textContent = t.target_range > 0
    ? `${t.target_range.toFixed(2)} m  ${deg(-t.target_bearing)}`
    : '—';
  $('r-tspeed').textContent = t.target_speed ? `${t.target_speed.toFixed(2)} m/s` : '—';
  $('r-follow-reason').textContent = t.follow_reason || '';

  $('r-head').textContent = deg(t.head_yaw);
  $('r-head-mode').textContent = t.head_mode;
  $('r-head-reason').textContent = t.head_reason || '';

  $('r-plan').textContent = t.plan_status;
  renderPlanning(t.planning);
  $('r-known').textContent = `${(t.known_fraction * 100).toFixed(1)}%`;

  const err = t.error || t.vision_error;
  if (err) { $('error-box').hidden = false; $('error-box').textContent = err; }

  state.speed.push(Math.abs(t.measured[0]));
  if (state.speed.length > 200) state.speed.shift();
  spark($('spark-speed'), state.speed, '#4ea1ff');
}

/* Where each vision frame's time goes: one bar per stage, widths in
 * proportion, the biggest one named so it is obvious what to make faster. */
const STAGE_COLOURS = { capture: '#7a8699', stereo: '#4ea1ff', detect: '#e0a43a', map: '#63d297' };
function renderStages(stages, parallel) {
  const box = $('r-stages');
  box.title = parallel
    ? 'stereo and detect run side by side, so a frame takes about the slower of the two, not their sum.'
    : "Where each frame's time goes, smoothed.";
  if (!stages.length) { box.replaceChildren(); return; }
  const total = stages.reduce((sum, [, ms]) => sum + ms, 0) || 1;
  const bar = document.createElement('div');
  bar.className = 'stage-bar';
  const legend = document.createElement('div');
  legend.className = 'stage-legend';
  for (const [name, ms] of stages) {
    const seg = document.createElement('span');
    seg.style.width = `${(100 * ms) / total}%`;
    seg.style.background = STAGE_COLOURS[name] || '#999';
    bar.appendChild(seg);
    const key = document.createElement('span');
    key.innerHTML = `<i style="background:${STAGE_COLOURS[name] || '#999'}"></i>${name} <b>${ms < 10 ? ms.toFixed(1) : Math.round(ms)}</b>`;
    legend.appendChild(key);
  }
  box.replaceChildren(bar, legend);
}

/* Pathfinding latency. Shown as typical and p95 rather than the last value,
 * which flickers: one slow search among fast ones is what p95 is for. */
const PLAN_SLOW_MS = 250;     // frame -> path above this: the robot is steering on old news
function renderPlanning(p) {
  const ms = (v) => (v == null ? '—' : v < 10 ? v.toFixed(1) : Math.round(v).toString());
  const pair = (s) => (s ? `${ms(s.mean)} · ${ms(s.p95)} ms` : '—');
  if (!p) return;
  const idle = p.idle && !p.plans;
  $('r-plan-ms').textContent = idle ? 'idle — no goal' : pair(p.search_ms);
  $('r-plan-total').textContent = idle ? '—' : pair(p.total_ms);
  $('r-plan-rate').textContent = p.plans
    ? `${p.rate_hz.toFixed(1)} · ${Math.round(p.reused_fraction * 100)}%` : '—';
  $('r-plan-age').textContent = p.path_age_ms == null ? '—' : `${ms(p.path_age_ms)} ms`;
  $('plan-total-stat').classList.toggle('alert', !!(p.total_ms && p.total_ms.p95 > PLAN_SLOW_MS));
  $('plan-age-stat').classList.toggle('alert', p.path_age_ms != null && p.path_age_ms > 2 * PLAN_SLOW_MS);
  sparkMulti($('spark-plan'), [
    { series: p.history.map((h) => h[1]), colour: '#4ea1ff' },
    // Dots: most cycles re-use the route, so searches are scattered points.
    { series: p.history.map((h) => h[0]), colour: '#e0a43a', dots: true },
  ]);
}

/* Several series on one shared scale; null is a gap (a reused plan has no
 * search time). A series with dots: true is drawn as points, not a line. */
function sparkMulti(canvas, lines) {
  fitCanvas(canvas);
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const values = lines.flatMap((l) => l.series.filter((v) => v != null));
  if (values.length < 2) return;
  const top = Math.max(...values, 1) * 1.15;
  for (const { series, colour, dots } of lines) {
    if (series.length < 2) continue;
    if (dots) {
      ctx.fillStyle = colour;
      series.forEach((value, i) => {
        if (value == null) return;
        const x = (i * canvas.width) / (series.length - 1);
        const r = 2 * (window.devicePixelRatio || 1);
        const y = Math.min(canvas.height - r,
          canvas.height - (value / top) * (canvas.height - 6) - 3);
        ctx.beginPath();
        ctx.arc(Math.max(r, Math.min(canvas.width - r, x)), y, r, 0, 2 * Math.PI);
        ctx.fill();
      });
      continue;
    }
    ctx.strokeStyle = colour;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    let pen = false;
    series.forEach((value, i) => {
      if (value == null) { pen = false; return; }
      const x = (i * canvas.width) / (series.length - 1);
      const y = canvas.height - (value / top) * (canvas.height - 6) - 3;
      pen ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      pen = true;
    });
    ctx.stroke();
  }
}

function spark(canvas, series, colour) {
  fitCanvas(canvas);
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (series.length < 2) return;
  const top = Math.max(...series, 0.05) * 1.15;
  ctx.strokeStyle = colour;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  series.forEach((value, i) => {
    const x = (i * canvas.width) / (series.length - 1);
    const y = canvas.height - (value / top) * (canvas.height - 6) - 3;
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();
}

/* ------------------------------------------------------------------ loop */

async function loadBitmap(url) {
  const res = await api(url);
  return createImageBitmap(await res.blob());
}

async function poll() {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 800);
    const res = await api('/api/state', { signal: controller.signal });
    clearTimeout(timer);
    const t = await res.json();
    state.t = t;
    state.lastOk = performance.now();
    if (t.params) applyParams(t.params);

    if (t.vision_seq !== state.cameraSeq) {
      state.cameraSeq = t.vision_seq;
      refreshImage('/api/camera.jpg', $('img-camera'), $('camera-empty'), 'cameraBusy');
      refreshImage('/api/depth.jpg', $('img-depth'), $('depth-empty'), 'depthBusy');
    }
    if (t.costmap_version !== state.costVersion && !state.mapBusy) {
      state.mapBusy = true;
      state.costVersion = t.costmap_version;
      loadBitmap('/api/costmap.png')
        .then((img) => { state.costImage = img; })
        .catch(() => {})
        .finally(() => { state.mapBusy = false; });
    }
  } catch (err) {
    /* render() shows the lost link */
  }
  render();
}

function frame() {
  renderMap();
  requestAnimationFrame(frame);
}

wireInputs();
wireMotors();
wirePad();
wirePerception();
wireRangeCal();
wireMap();
syncLabels();
showMode('manual');
poll();
setInterval(poll, 100);
setInterval(sendTeleop, 50);     // repeats while held; key-up and release send at once
requestAnimationFrame(frame);
