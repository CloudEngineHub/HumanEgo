// HumanEgo Sim — UI: configs, initial-condition explorer, timeline, policy-input sync.
import { SimViewer } from './viewer.js';

const $ = (s) => document.querySelector(s);
const el = (tag, attrs = {}, html = '') => { const e = document.createElement(tag); Object.assign(e, attrs); if (html) e.innerHTML = html; return e; };
const fmt = (x, d = 1) => (x == null ? '—' : Number(x).toFixed(d));
const pct = (x) => `${Math.round(100 * x)}%`;

const state = { manifest: null, robot: null, camera: null, mode: null, cfg: null, rollout: null, ro: null };
const viewer = new SimViewer($('#viewport'));
const video = $('#pip-video');

// ------------------------------------------------------------------ boot
async function boot() {
  const man = await (await fetch('data/manifest.json', { cache: 'no-cache' })).json();
  state.manifest = man;
  const d = man.default || {};
  state.robot = d.robot || man.configs[0].robot;
  state.camera = d.camera || man.configs[0].camera;
  state.mode = d.mode || man.configs[0].mode;
  const qs = new URLSearchParams(location.search);
  for (const k of ['robot', 'camera', 'mode']) if (qs.get(k)) state[k] = qs.get(k);
  renderStats();
  renderPills();
  renderResultsTable();
  renderTuning();
  await selectConfig(qs.get('rollout') || d.rollout);
  if (qs.get('view')) viewer.setView(qs.get('view'), true);
  if (qs.get('t') !== null) { pause(); viewer.setTime(parseFloat(qs.get('t'))); }
  document.body.dataset.ready = '1';
}

function renderStats() {
  const m = state.manifest;
  const all = m.configs.flatMap(c => c.rollouts);
  const sides = all.flatMap(r => Object.values(r.per_side || {}));
  const ok = sides.filter(s => s.success).length;
  $('#stats').innerHTML = [
    `<span class="chip"><b>${all.length}</b> policy rollouts</span>`,
    `<span class="chip"><b>${pct(ok / Math.max(1, sides.length))}</b> breads served</span>`,
    `<span class="chip"><b>${new Set(m.configs.map(c => c.robot)).size}</b> robots · <b>${new Set(m.configs.map(c => c.camera)).size}</b> cameras</span>`,
    `<span class="chip">checkpoint <b>Leo-TX/HumanEgo · serve_bread</b></span>`,
  ].join('');
}

function cfgOf(robot, camera, mode) {
  return state.manifest.configs.find(c => c.robot === robot && c.camera === camera && c.mode === mode);
}

function renderPills() {
  const m = state.manifest;
  const mk = (id, key, items) => {
    const box = $(id); box.innerHTML = '';
    for (const [val, label] of items) {
      const b = el('button', { className: 'pill' + (state[key] === val ? ' on' : ''), textContent: label });
      const probe = { robot: state.robot, camera: state.camera, mode: state.mode, [key]: val };
      b.disabled = !cfgOf(probe.robot, probe.camera, probe.mode);
      b.onclick = () => { state[key] = val; renderPills(); selectConfig(); };
      box.appendChild(b);
    }
  };
  mk('#pills-robot', 'robot', Object.entries(m.robots));
  mk('#pills-camera', 'camera', Object.entries(m.cameras));
  mk('#pills-mode', 'mode', [['bimanual', 'Two arms'], ['single', 'One arm']]);
}

async function selectConfig(rolloutId) {
  let cfg = cfgOf(state.robot, state.camera, state.mode);
  if (!cfg) {   // fall back to any config of this robot
    cfg = state.manifest.configs.find(c => c.robot === state.robot) || state.manifest.configs[0];
    Object.assign(state, { robot: cfg.robot, camera: cfg.camera, mode: cfg.mode });
    renderPills();
  }
  state.cfg = cfg;
  const ok = cfg.rollouts.flatMap(r => Object.values(r.per_side)).filter(s => s.success).length;
  const n = cfg.rollouts.flatMap(r => Object.values(r.per_side)).length;
  $('#count').textContent = `${cfg.rollouts.length} rollouts · ${pct(ok / Math.max(1, n))} served`;
  showLoading('Loading the robot…');
  const model = await (await fetch(`data/models/${cfg.model}.json`)).json();
  await viewer.loadModel(model, (p) => showLoading(`Loading meshes… ${Math.round(p * 100)}%`));
  const pick = cfg.rollouts.find(r => r.id === rolloutId) || cfg.rollouts.find(r => r.featured) || cfg.rollouts[0];
  await selectRollout(pick.id, true);
}

function showLoading(msg) { $('#loading-msg').textContent = msg; $('#loading').classList.remove('hide'); }
function hideLoading() { $('#loading').classList.add('hide'); }

async function selectRollout(id, autoplay = true) {
  const meta = state.cfg.rollouts.find(r => r.id === id);
  state.rollout = meta;
  showLoading('Loading the rollout…');
  const ro = await (await fetch(`data/rollouts/${id}.json`)).json();
  state.ro = ro;
  viewer.setRollout(ro);
  // policy-input video
  const pip = $('#pip');
  pip.classList.toggle('single', state.cfg.mode === 'single');
  if (ro.policy_video) { video.src = ro.policy_video; video.load(); }
  $('#pip-cap').textContent = state.cfg.mode === 'single' ? 'right arm' : 'right arm  ·  left arm (mirrored)';
  renderResult();
  renderMinimap();
  drawTrack();
  hideLoading();
  if (autoplay) play(true);
}

// ------------------------------------------------------------------ result card
function renderResult() {
  const ro = state.ro, res = ro.result;
  const box = $('#result'); box.innerHTML = '';
  for (const side of Object.keys(res.per_side)) {
    const r = res.per_side[side];
    const ok = r.success;
    box.appendChild(el('div', { className: 'res-arm' }, `<div class="k">${side} arm</div>
      <div class="v ${ok ? 'ok' : 'bad'}">${ok ? '✓ Served' : '✗ Missed'}</div>
      <div class="s">${fmt(100 * r.dist_to_plate_center, 1)} cm from plate centre</div>`));
  }
  box.appendChild(el('div', { className: 'res-sum' }, `<span>Episode <b>${fmt(res.duration, 1)} s</b> sim time</span><span>${ro.policy.length} re-plans</span>`));
  $('#res-meta').textContent = res.success ? 'all breads served' : 'partial';
  const t = viewer.ro;
  $('#hud-title').innerHTML = `<span class="dot"></span><span>${state.manifest.robots[state.cfg.robot]} · ${state.manifest.cameras[state.cfg.camera]}</span>`;
}

// ------------------------------------------------------------------ minimap (top-down)
// world x (forward) -> up, world y (left) -> left
const MM = { w: 340, h: 300, x0: 0.18, x1: 0.80, y0: -0.52, y1: 0.52 };
const mmX = (y) => 18 + (MM.y1 - y) / (MM.y1 - MM.y0) * (MM.w - 36);
const mmY = (x) => 16 + (MM.x1 - x) / (MM.x1 - MM.x0) * (MM.h - 64);
function renderMinimap() {
  const svg = $('#minimap');
  const ro = state.ro, sc = ro.scene;
  const sides = Object.keys(ro.result.per_side);
  let s = `<defs><radialGradient id="pg" cx="50%" cy="45%"><stop offset="0%" stop-color="#cfe0f3"/><stop offset="100%" stop-color="#8fb0d6"/></radialGradient></defs>`;
  s += `<rect x="6" y="6" width="${MM.w - 12}" height="${MM.h - 50}" rx="10" fill="#1a1510" stroke="#3a2c1d"/>`;
  // grid
  for (let x = 0.2; x <= 0.8; x += 0.1) s += `<line x1="10" x2="${MM.w - 10}" y1="${mmY(x)}" y2="${mmY(x)}" stroke="#2a2118" stroke-width="1"/>`;
  // robot + camera at the bottom
  s += `<rect x="${mmX(0.40)}" y="${MM.h - 40}" width="${mmX(-0.40) - mmX(0.40)}" height="7" rx="3" fill="#30353f"/>`;
  for (const yb of [0.3238, -0.3238]) s += `<circle cx="${mmX(yb)}" cy="${MM.h - 36}" r="9" fill="#11141a" stroke="#4b5363"/>`;
  s += `<rect x="${mmX(0) - 7}" y="${MM.h - 30}" width="14" height="9" rx="2" fill="#ff8c2b"/><text x="${mmX(0)}" y="${MM.h - 8}" fill="#6b7586" font-size="10" text-anchor="middle" font-family="Inter">camera</text>`;
  // all rollouts of this config as dots (bread positions)
  for (const r of state.cfg.rollouts) {
    for (const side of sides) {
      const b = r.scene[`bread_${side}`]; if (!b) continue;
      const okk = r.per_side[side]?.success;
      const cur = r.id === ro.id;
      s += `<circle class="mm-dot" data-id="${r.id}" data-side="${side}" cx="${mmX(b[1])}" cy="${mmY(b[0])}" r="${cur ? 0 : 4.2}"
              fill="${okk ? '#3ddc97' : '#ff5d6c'}" fill-opacity=".55" stroke="${okk ? '#3ddc97' : '#ff5d6c'}" stroke-opacity=".9" style="cursor:pointer"/>`;
    }
  }
  // current layout
  const p = sc.plate; const pr = 0.1225;
  const rx = (mmX(-pr) - mmX(pr)) / 2, ry = (mmY(-pr) - mmY(pr)) / 2;
  s += `<ellipse cx="${mmX(p[1])}" cy="${mmY(p[0])}" rx="${Math.abs(rx)}" ry="${Math.abs(ry)}" fill="url(#pg)" stroke="#e6f0fb" stroke-opacity=".7"/>`;
  for (const side of sides) {
    const b = sc[`bread_${side}`]; if (!b) continue;
    const L = side === 'right' && state.cfg.mode === 'bimanual' ? 0.12 : 0.138, W = 0.066;
    const cx = mmX(b[1]), cy = mmY(b[0]);
    const sx = (mmX(-1) - mmX(1)) / 2 / -1, sy = (mmY(-1) - mmY(1)) / 2;
    const ang = -(b[2]) + 90;   // map yaw (about +z, x->up) into svg rotation
    s += `<g class="mm-bread" data-side="${side}" transform="translate(${cx},${cy}) rotate(${ang})" style="cursor:grab">
            <ellipse rx="${Math.abs(L / 2 * sx)}" ry="${Math.abs(W / 2 * sy)}" fill="#d98a3a" stroke="#ffcf9e" stroke-width="1.5"/></g>`;
  }
  svg.innerHTML = s;
  $('#mm-meta').textContent = `${state.cfg.rollouts.length} scenes`;
  svg.querySelectorAll('.mm-dot').forEach(d => d.addEventListener('click', () => selectRollout(d.dataset.id)));
  // drag a bread -> snap to the nearest rollout (by that arm's bread position)
  svg.querySelectorAll('.mm-bread').forEach(g => {
    g.addEventListener('pointerdown', (ev) => {
      ev.preventDefault();
      const side = g.dataset.side;
      const pt = svg.createSVGPoint();
      const move = (e) => {
        pt.x = e.clientX; pt.y = e.clientY;
        const q = pt.matrixTransform(svg.getScreenCTM().inverse());
        g.setAttribute('transform', g.getAttribute('transform').replace(/translate\([^)]*\)/, `translate(${q.x},${q.y})`));
        g.dataset.x = q.x; g.dataset.y = q.y;
      };
      const up = () => {
        window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up);
        if (g.dataset.x === undefined) return;
        let best = null, bd = 1e9;
        for (const r of state.cfg.rollouts) {
          const b = r.scene[`bread_${side}`]; if (!b) continue;
          const d = Math.hypot(mmX(b[1]) - g.dataset.x, mmY(b[0]) - g.dataset.y);
          if (d < bd) { bd = d; best = r; }
        }
        if (best) selectRollout(best.id);
      };
      window.addEventListener('pointermove', move); window.addEventListener('pointerup', up);
    });
  });
}

// ------------------------------------------------------------------ timeline
const track = $('#track'), tcv = $('#track-canvas'), scrub = $('#scrub');
function drawTrack() {
  const ro = state.ro; if (!ro) return;
  const dpr = window.devicePixelRatio || 1;
  const W = track.clientWidth, H = track.clientHeight;
  tcv.width = W * dpr; tcv.height = H * dpr;
  const g = tcv.getContext('2d'); g.scale(dpr, dpr);
  const t0 = viewer.t0, t1 = viewer.t1, X = (t) => (t - t0) / (t1 - t0) * W;
  g.fillStyle = 'rgba(255,255,255,.12)'; g.beginPath(); g.roundRect(0, H / 2 - 2, W, 4, 2); g.fill();
  const tn = viewer.time;
  const grad = g.createLinearGradient(0, 0, W, 0); grad.addColorStop(0, '#ffae52'); grad.addColorStop(1, '#ff6a00');
  g.fillStyle = grad; g.beginPath(); g.roundRect(0, H / 2 - 2, Math.max(4, X(tn)), 4, 2); g.fill();
  // grasp-probability sparklines (first predicted step) per arm
  const cols = { right: 'rgba(255,174,82,.95)', left: 'rgba(34,211,238,.95)' };
  for (const side of Object.keys(ro.result.per_side)) {
    g.strokeStyle = cols[side]; g.lineWidth = 1.2; g.beginPath(); let started = false;
    for (const p of ro.policy) {
      const e = p.sides[side]; if (!e) { started = false; continue; }
      const gp = Math.max(...e.grasp.slice(0, 10));
      const x = X(p.t + (ro.policy_period || 0.3)), y = H / 2 - 4 - gp * (H / 2 - 6);
      if (!started) { g.moveTo(x, y); started = true; } else g.lineTo(x, y);
    }
    g.stroke();
  }
  // t=0 marker (policy starts)
  g.fillStyle = 'rgba(255,255,255,.55)'; g.fillRect(X(0) - 0.5, H / 2 - 7, 1, 14);
  // knob
  g.fillStyle = '#fff'; g.beginPath(); g.arc(X(tn), H / 2, 6, 0, Math.PI * 2); g.fill();
  g.fillStyle = '#ff6a00'; g.beginPath(); g.arc(X(tn), H / 2, 3, 0, Math.PI * 2); g.fill();
}
scrub.addEventListener('input', () => {
  const a = scrub.value / 1000;
  viewer.setTime(viewer.t0 + a * (viewer.t1 - viewer.t0));
  pause();
});
viewer.onTime = (t) => {
  if (!state.ro) return;
  scrub.value = Math.round((t - viewer.t0) / (viewer.t1 - viewer.t0) * 1000);
  $('#tlabel').textContent = t < 0 ? `scene  ${fmt(t, 1)} s` : `t = ${fmt(t, 1)} s`;
  syncVideo(t);
  updateArmBadges(t);
  if (!drawTrack.raf) drawTrack.raf = requestAnimationFrame(() => { drawTrack.raf = 0; drawTrack(); });
};
viewer.onEnded = () => setPlayIcon(false);

function syncVideo(t) {
  const ro = state.ro; if (!ro || !ro.policy_video || !video.duration) return;
  const k = viewer.currentPolicy;
  if (k === syncVideo.k) return;
  syncVideo.k = k;
  const fps = 1 / (ro.policy_period || 0.3);
  const tt = Math.max(0, Math.min(video.duration - 0.01, (Math.max(k, 0) + 0.5) / fps));
  try { video.currentTime = tt; } catch (e) { /* not ready */ }
  $('#pip').style.opacity = k >= 0 ? 1 : 0.35;
}

function updateArmBadges(t) {
  const ro = state.ro, box = $('#arm-state');
  const k = viewer.currentPolicy;
  let html = '';
  for (const side of Object.keys(ro.result.per_side)) {
    const e = k >= 0 ? ro.policy[k].sides[side] : null;
    let phase = 'waiting', cls = '';
    if (t < 0) phase = 'scene setup';
    else if (e) { phase = e.grasping ? 'carrying' : (e.grasp && Math.max(...e.grasp) > 0.5 ? 'reaching to grasp' : 'reaching'); if (e.grasping) cls = 'grasp'; }
    else if (k >= 0) { const r = ro.result.per_side[side]; phase = r.success ? 'served ✓' : 'done'; cls = r.success ? 'done' : 'fail'; }
    if (side === 'left' && t >= 0 && !e && (ro.policy[k]?.sides.right) && !(k > 0 && ro.policy.slice(0, k).some(p => p.sides.left))) phase = 'starts at 6 s';
    html += `<span class="arm-badge ${cls}"><b>${side === 'right' ? 'R' : 'L'}</b>${phase}</span>`;
  }
  if (box.innerHTML !== html) box.innerHTML = html;
}

// ------------------------------------------------------------------ transport
function setPlayIcon(on) { $('#play-ico').innerHTML = on ? '<path d="M7 5h4v14H7zM13 5h4v14h-4z"/>' : '<path d="M8 5v14l11-7z"/>'; }
function play(fromStart = false) {
  if (fromStart || viewer.time >= viewer.t1 - 1e-3) viewer.setTime(viewer.t0);
  viewer.playing = true; setPlayIcon(true);
}
function pause() { viewer.playing = false; setPlayIcon(false); }
$('#play').onclick = () => (viewer.playing ? pause() : play());
$('#speed').querySelectorAll('button').forEach(b => b.onclick = () => {
  viewer.speed = parseFloat(b.dataset.s);
  $('#speed').querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
});
document.querySelectorAll('[data-cam]').forEach(b => b.onclick = () => {
  viewer.setView(b.dataset.cam);
  document.querySelectorAll('[data-cam]').forEach(x => x.classList.toggle('on', x === b));
});
viewer.onUserCamera = () => document.querySelectorAll('[data-cam]').forEach(x => x.classList.remove('on'));
document.querySelectorAll('[data-ov]').forEach(b => b.onclick = () => {
  const on = !b.classList.contains('on'); b.classList.toggle('on', on);
  if (b.dataset.ov === 'pip') $('#pip-wrap').style.display = on ? '' : 'none';
  else viewer.setOverlay(b.dataset.ov, on);
});
$('#btn-shuffle').onclick = () => {
  const rs = state.cfg.rollouts.filter(r => r.id !== state.rollout.id);
  selectRollout(rs[Math.floor(Math.random() * rs.length)].id);
};
$('#btn-replay').onclick = () => play(true);
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  if (e.code === 'Space') { e.preventDefault(); viewer.playing ? pause() : play(); }
  if (e.code === 'ArrowRight') viewer.setTime(viewer.time + 1);
  if (e.code === 'ArrowLeft') viewer.setTime(viewer.time - 1);
});
window.addEventListener('resize', drawTrack);

// ------------------------------------------------------------------ results + tuning sections
function renderResultsTable() {
  const m = state.manifest, tb = $('#res-table');
  let h = '<tr><th>Robot</th><th>Camera</th><th>Arms</th><th>Rollouts</th><th>Right arm</th><th>Left arm</th><th>Both served</th></tr>';
  for (const c of (m.summary || [])) {
    const bar = (v) => v == null ? '<span class="meta">—</span>' : `<span class="bar-s" style="width:${Math.round(60 * v)}px"></span>${pct(v)}`;
    h += `<tr><td>${m.robots[c.robot]}</td><td>${m.cameras[c.camera]}</td><td>${c.mode === 'bimanual' ? 'two' : 'one'}</td><td>${c.n}</td>
          <td>${bar(c.right)}</td><td>${bar(c.left)}</td><td>${bar(c.both)}</td></tr>`;
  }
  tb.innerHTML = h;
  $('#res-note').innerHTML = m.summary_note || '';
}

function renderTuning() {
  const tun = state.manifest.tuning; const cv = $('#tune-plot');
  if (!tun) { cv.parentElement.parentElement.style.display = 'none'; return; }
  const draw = () => {
    const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
    cv.width = W * dpr; cv.height = H * dpr; const g = cv.getContext('2d'); g.scale(dpr, dpr);
    g.fillStyle = '#0c0f14'; g.fillRect(0, 0, W, H);
    const L = 48, R = 18, T = 18, B = 34, n = tun.batches.length;
    const X = (i) => L + (W - L - R) * (n <= 1 ? 0.5 : i / (n - 1)), Y = (v) => T + (H - T - B) * (1 - v);
    g.strokeStyle = '#242a35'; g.fillStyle = '#6b7586'; g.font = '11px Inter'; g.lineWidth = 1;
    for (let v = 0; v <= 1.001; v += 0.25) { g.beginPath(); g.moveTo(L, Y(v)); g.lineTo(W - R, Y(v)); g.stroke(); g.fillText(pct(v), 8, Y(v) + 4); }
    tun.batches.forEach((b, i) => {
      g.fillStyle = 'rgba(152,162,179,.35)';
      for (const v of b.scores) { g.beginPath(); g.arc(X(i) + (Math.random() - .5) * 8, Y(v), 2.6, 0, 7); g.fill(); }
      g.fillStyle = '#6b7586'; g.fillText(`b${i}`, X(i) - 7, H - 12);
      if (b.action) { g.fillStyle = b.action === 'revise_global' ? '#c084fc' : '#4b5363'; g.fillText(b.action.replace('_', ' '), X(i) - 26, H - 1); }
    });
    g.strokeStyle = '#ff8c2b'; g.lineWidth = 2.2; g.beginPath();
    let best = -1;
    tun.batches.forEach((b, i) => { best = Math.max(best, Math.max(...b.scores)); i ? g.lineTo(X(i), Y(best)) : g.moveTo(X(i), Y(best)); });
    g.stroke();
  };
  draw(); window.addEventListener('resize', draw);
  $('#tune-note').innerHTML = tun.note || '';
}

boot().catch(e => { console.error(e); showLoading('Failed to load: ' + e.message); });
