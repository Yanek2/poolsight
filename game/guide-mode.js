// ===================== GUIDE MODE (aiming guide) =====================
// Tap a ball = lock it (pocket lines meet there). Tap a pocket = line into that pocket.
// LINE on = dragging on the table swings the guide line. ◀◀ ◀ ▶ ▶▶ fine-turn it.
// Values tuned by the player in PoolSight Guide on the PC.
(function () {
  const TUNE = {bounce: 1.04, slow: 1.59, mouth: 0.5, corner: 53, midMax: 65};
  const G = {on: true, line: false, power: 1.03, bounces: 6, point: null, dir: null, label: null, drag: null};
  try { Object.assign(G, JSON.parse(localStorage.getItem('guideMode') || '{}'), {point: null, dir: null, label: null, drag: null}); } catch (e) {}
  const store = () => { try { localStorage.setItem('guideMode', JSON.stringify({on: G.on, line: G.line, power: G.power, bounces: G.bounces})); } catch (e) {} };

  // ---- measured model (game units: felt 980 wide) ----
  const POWER_V0 = [[0, 0], [.1, 264], [.2, 446], [.3, 644], [.4, 858], [.5, 1090], [.6, 1338], [.7, 1604], [.8, 1886], [.9, 2184], [1, 2500]];
  const SLIDE = 1260, ROLLD = 190, RFRAC = 5 / 7;
  const BOUNCE_K = [[0, .643], [299, .643], [488, .663], [877, .733], [1271, .755], [1690, .765], [2142, .771], [3000, .775]];
  const CV = [299, 488, 877, 1271, 1690, 2142], CA = [5, 10, 15, 20, 30, 40, 50, 60, 70, 75, 80, 84, 87];
  const C_ROLL = [[298, 296, 293, 288, 276, 260, 242, 224, 207, 201, 196, 194, 193], [459, 456, 451, 444, 426, 403, 377, 350, 327, 318, 311, 307, 306], [693, 689, 684, 675, 653, 624, 593, 561, 534, 523, 515, 511, 510], [955, 950, 942, 932, 903, 867, 827, 787, 753, 740, 730, 725, 723], [1243, 1237, 1227, 1214, 1179, 1133, 1083, 1034, 992, 975, 963, 957, 954], [1561, 1554, 1541, 1525, 1481, 1423, 1362, 1299, 1246, 1225, 1210, 1202, 1198]];
  const C_SLIDE = [[1, 2, 2, 3, 5, 6, 6, 7, 7, 7, 7, 7, 7], [11, 11, 12, 13, 16, 18, 19, 20, 21, 21, 21, 20, 20], [113, 113, 112, 110, 107, 102, 96, 91, 87, 85, 84, 83, 83], [277, 275, 271, 267, 254, 239, 222, 207, 194, 189, 186, 183, 183], [517, 513, 506, 497, 471, 441, 407, 377, 351, 342, 335, 331, 330], [848, 841, 830, 815, 772, 723, 667, 618, 576, 561, 549, 543, 541]];
  const C_SHIFT = [[.1, .1, .2, .3, .4, .5, .5, .4, .3, .2, .2, .1, .1], [.9, 1, 1, 1.1, 1.3, 1.3, 1.3, 1.1, .8, .6, .4, .3, .1], [4, 4, 3.9, 3.8, 3.5, 3.1, 2.5, 2, 1.3, 1, .7, .4, .2], [5.1, 5, 4.9, 4.7, 4.3, 3.7, 3, 2.3, 1.5, 1.1, .8, .5, .2], [5.6, 5.5, 5.4, 5.2, 4.7, 4, 3.2, 2.4, 1.6, 1.2, .8, .5, .2], [5.9, 5.8, 5.7, 5.5, 4.9, 4.2, 3.4, 2.5, 1.7, 1.3, .9, .5, .2]];
  const RV = [490, 780, 1200, 1500, 2200], RA = [10, 20, 30, 45, 60, 75, 87];
  const R_K = [[.643, .643, .643, .643, .643, .643, .643], [.643, .643, .643, .643, .643, .643, .643], [.711, .707, .700, .680, .643, .643, .643], [.710, .707, .700, .690, .696, .672, .672], [.730, .730, .725, .720, .720, .722, .722]];
  const R_KEEP = [[.99, .965, .925, .84, .75, .67, .643], [.99, .965, .925, .84, .75, .67, .644], [.83, .82, .80, .77, .75, .67, .644], [.80, .79, .77, .73, .68, .64, .62], [.75, .74, .72, .68, .62, .60, .59]];
  const R_SLIDE = [[5, 9, 12, 14, 12, 9, 9], [12, 22, 31, 38, 39, 37, 37], [189, 178, 160, 122, 82, 81, 81], [300, 290, 270, 230, 244, 192, 192], [700, 680, 650, 600, 560, 520, 509]];

  const lerp1 = (xs, ys, x) => { if (x <= xs[0]) return ys[0]; for (let i = 1; i < xs.length; i++) if (x <= xs[i]) return ys[i - 1] + (ys[i] - ys[i - 1]) * (x - xs[i - 1]) / (xs[i] - xs[i - 1]); return ys[ys.length - 1]; };
  const interp = (tab, x) => lerp1(tab.map(t => t[0]), tab.map(t => t[1]), x);
  const grid = (tab, vs, as, v, a, scaleUp) => { let sc = 1; if (v > vs[vs.length - 1]) { sc = v / vs[vs.length - 1]; v = vs[vs.length - 1]; } const rows = tab.map(r => lerp1(as, r, a)); return lerp1(vs, rows, v) * (scaleUp ? sc : 1); };
  const startSpeed = p => p <= 1 ? interp(POWER_V0, p) : 2500 + (p - 1) * 3160;

  function Roll(v) { this.v = v; this.rollAt = RFRAC * v; this.free = 0; }
  Roll.prototype.go = function (dist) {
    const sl = SLIDE * TUNE.slow, rl = ROLLD * TUNE.slow; let done = 0;
    if (this.free > 0) { const f = Math.min(this.free, dist); this.free -= f; done = f; dist -= f; if (dist <= 0) return done; }
    if (this.v > this.rollAt) { const ds = (this.v * this.v - this.rollAt * this.rollAt) / (2 * sl);
      if (dist <= ds) { this.v = Math.sqrt(Math.max(this.v * this.v - 2 * sl * dist, 0)); return done + dist; }
      this.v = this.rollAt; done += ds; dist -= ds; }
    const reach = this.v * this.v / (2 * rl);
    if (dist >= reach) { this.v = 0; return done + reach; }
    this.v = Math.sqrt(this.v * this.v - 2 * rl * dist); return done + dist;
  };
  Roll.prototype.rolling = function () { return this.free <= 0 && this.v <= this.rollAt + 1e-6; };
  Roll.prototype.bounced = function (a) {
    const v = this.v; let k;
    if (this.rolling()) { k = grid(R_K, RV, RA, v, a); this.v = this.rollAt = v * grid(R_KEEP, RV, RA, v, a); this.free = grid(R_SLIDE, RV, RA, v, a); }
    else { k = interp(BOUNCE_K, v); this.v = this.rollAt = grid(C_ROLL, CV, CA, v, a, true); this.free = grid(C_SLIDE, CV, CA, v, a, true); }
    return k;
  };

  // pockets: where shots really drop (measured), outward axis, mouth half-width
  const AIM = [[X0 + 14.15, Y0 + 12.15], [MX, Y0 + 5.95], [X1 - 14.15, Y0 + 12.15], [X0 + 14.15, Y1 - 12.15], [MX, Y1 - 5.95], [X1 - 14.15, Y1 - 12.15]];
  const AX = [[-1, -1], [0, -1], [1, -1], [-1, 1], [0, 1], [1, 1]].map(([x, y]) => { const m = Math.hypot(x, y); return [x / m, y / m]; });
  const mouth = i => (i === 1 || i === 4 ? 12.6 : 18.8) * TUNE.mouth;

  function pocketHit(p, d, limit) {
    let best = null;
    for (let i = 0; i < 6; i++) {
      const wx = AIM[i][0] - p[0], wy = AIM[i][1] - p[1], t = wx * d[0] + wy * d[1];
      if (t <= 0 || t > limit + 3 * R) continue;
      if (Math.abs(d[0] * wy - d[1] * wx) > mouth(i)) continue;
      const ang = Math.acos(Math.max(-1, Math.min(1, d[0] * AX[i][0] + d[1] * AX[i][1]))) * 180 / Math.PI;
      if (ang > (i === 1 || i === 4 ? TUNE.midMax : TUNE.corner)) continue;
      if (!best || t < best.t) best = {t, i};
    }
    return best;
  }

  // straight line + cushion rebounds; returns {pts, pot (pocket index or -1), stop}
  function path(start, d, nb, speed) {
    const ix0 = X0 + R, iy0 = Y0 + R, ix1 = X1 - R, iy1 = Y1 - R;
    let p = start.slice(); d = d.slice(); const pts = [p.slice()]; const ball = speed ? new Roll(speed) : null;
    for (let b = 0; b <= nb; b++) {
      const ts = [];
      if (d[0] > 1e-9) ts.push([(ix1 - p[0]) / d[0], 0]); if (d[0] < -1e-9) ts.push([(ix0 - p[0]) / d[0], 0]);
      if (d[1] > 1e-9) ts.push([(iy1 - p[1]) / d[1], 1]); if (d[1] < -1e-9) ts.push([(iy0 - p[1]) / d[1], 1]);
      if (!ts.length) break;
      ts.sort((a, c) => a[0] - c[0]); const tt = Math.max(ts[0][0], 0), axis = ts[0][1];
      const hit = pocketHit(p, d, tt);
      if (ball) { const need = hit ? hit.t : tt, got = ball.go(need);
        if (got < need - 1e-6) { pts.push([p[0] + d[0] * got, p[1] + d[1] * got]); return {pts, pot: -1, stop: true}; } }
      if (hit) { pts.push(AIM[hit.i].slice()); return {pts, pot: hit.i, stop: false}; }
      p = [p[0] + d[0] * tt, p[1] + d[1] * tt]; pts.push(p.slice());
      const ang = Math.asin(Math.min(1, Math.abs(d[axis]))) * 180 / Math.PI;
      let k = 0.643;
      if (ball) { const shift = ball.rolling() ? 0 : grid(C_SHIFT, CV, CA, ball.v, ang); k = ball.bounced(ang);
        const tg = 1 - axis; if (Math.abs(d[tg]) > 1e-9) p[tg] -= Math.sign(d[tg]) * shift; }
      k = Math.min(.98, k * TUNE.bounce);
      d[axis] *= -k; const m = Math.hypot(d[0], d[1]); d = [d[0] / m, d[1] / m];
    }
    return {pts, pot: -1, stop: false};
  }

  function objSpeed(c, d) {
    const cb = cue(), gx = c[0] - d[0] * D, gy = c[1] - d[1] * D;
    let L = Math.hypot(gx - cb.x, gy - cb.y), cs = 1;
    if (L > 1) cs = Math.max(.05, ((gx - cb.x) * d[0] + (gy - cb.y) * d[1]) / L); else L = 300;
    const b = new Roll(startSpeed(G.power)); if (b.go(L) < L - 1e-6) return 0;
    return b.v * cs * .975;
  }

  // first ball touched along a polyline (a ball is 2 radii wide, not a thin line)
  function firstBlock(pts, skip) {
    for (let s = 0; s + 1 < pts.length; s++) {
      const a = pts[s], b = pts[s + 1], L = Math.hypot(b[0] - a[0], b[1] - a[1]); if (L < 1e-6) continue;
      const u = [(b[0] - a[0]) / L, (b[1] - a[1]) / L]; let best = null;
      for (const q of balls) { if (!q.on || skip.includes(q)) continue;
        const wx = q.x - a[0], wy = q.y - a[1], along = wx * u[0] + wy * u[1], side = Math.abs(u[0] * wy - u[1] * wx);
        if (side >= D) continue; const t = along - Math.sqrt(D * D - side * side);
        if (t > 0 && t < L && (!best || t < best.t)) best = {t, q}; }
      if (best) return {seg: s, at: [a[0] + u[0] * best.t, a[1] + u[1] * best.t], q: best.q};
    }
    return null;
  }
  const lockedBall = () => G.point && balls.find(b => b.on && Math.hypot(b.x - G.point[0], b.y - G.point[1]) < 1.5);

  function solvePocket(tgt) {
    const c = G.point, nb = Math.max(G.bounces, 1), lb = lockedBall(), skip = [cue(), lb].filter(Boolean);
    const hits = []; const N = 1440;
    for (let i = 0; i < N; i++) { const a = i * 2 * Math.PI / N, d = [Math.cos(a), Math.sin(a)];
      const r = path(c, d, nb, objSpeed(c, d)); hits.push(r.pot === tgt ? r.pts.length - 2 : -1); }
    let best = null;
    for (let i = 0; i < N;) { if (hits[i] < 0) { i++; continue; }
      let j = i; while (j + 1 < N && hits[j + 1] === hits[i]) j++;
      const mid = (i + j) / 2 * 2 * Math.PI / N, d = [Math.cos(mid), Math.sin(mid)];
      const r = path(c, d, nb, objSpeed(c, d)), blk = firstBlock(r.pts, skip) ? 1 : 0, width = (j - i + 1) * 360 / N;
      const key = [blk, hits[i], -width];
      if (!best || key[0] < best.key[0] || (key[0] === best.key[0] && (key[1] < best.key[1] || (key[1] === best.key[1] && key[2] < best.key[2])))) best = {key, d, nb: hits[i], width, blk};
      i = j + 1; }
    return best;
  }

  // ---- toolbar on the bottom rail ----
  const BY = Y1 + CD + WOOD / 2 + 1;
  const TB = [
    {id: 'guide', x: 232, w: 76}, {id: 'line', x: 314, w: 60},
    {id: 'l2', x: 372, w: 40}, {id: 'l1', x: 416, w: 40}, {id: 'r1', x: 460, w: 40}, {id: 'r2', x: 504, w: 40},
    {id: 'pm', x: 760, w: 40}, {id: 'pw', x: 830, w: 92}, {id: 'pp', x: 900, w: 40},
    {id: 'bm', x: 980, w: 40}, {id: 'bn', x: 1036, w: 64}, {id: 'bp', x: 1092, w: 40}];
  const inTB = p => TB.find(b => Math.abs(p.x - b.x) < b.w / 2 && Math.abs(p.y - BY) < 16);
  const turn = deg => { if (!G.dir) return; const a = deg * Math.PI / 180, c = Math.cos(a), s = Math.sin(a);
    G.dir = [c * G.dir[0] - s * G.dir[1], s * G.dir[0] + c * G.dir[1]]; G.label = null; };

  function press(b) {
    if (b.id === 'guide') G.on = !G.on;
    else if (!G.on) return;
    else if (b.id === 'line') G.line = !G.line;
    else if (b.id === 'l2') turn(-1); else if (b.id === 'l1') turn(-.1);
    else if (b.id === 'r1') turn(.1); else if (b.id === 'r2') turn(1);
    else if (b.id === 'pm') G.power = Math.max(.05, +(G.power - .05).toFixed(2));
    else if (b.id === 'pp') G.power = Math.min(2, +(G.power + .05).toFixed(2));
    else if (b.id === 'bm') G.bounces = Math.max(1, G.bounces - 1);
    else if (b.id === 'bp') G.bounces = Math.min(8, G.bounces + 1);
    store();
  }

  function tap(p) {
    for (let i = 0; i < 6; i++) if (Math.hypot(p.x - PCK[i].x, p.y - PCK[i].y) < 34 && G.point) {
      const s = solvePocket(i);
      if (s) { G.dir = s.d; G.label = (s.nb ? s.nb + (s.nb > 1 ? ' banks' : ' bank') : 'direct') + '  ±' + (s.width / 2).toFixed(1) + '°' + (s.blk ? '  BLOCKED' : ''); }
      else { G.dir = null; G.label = 'no way into that pocket'; }
      return true;
    }
    let best = null;
    for (const b of balls) if (b.on) { const d = Math.hypot(p.x - b.x, p.y - b.y); if (d < 2.5 * R && (!best || d < best.d)) best = {d, b}; }
    if (best) { G.point = [best.b.x, best.b.y]; G.dir = null; G.label = null; return true; }
    return false;
  }

  const onTable = p => p.x > OX && p.x < OX + OW && p.y > OY && p.y < OY + OH;
  cv.addEventListener('pointerdown', e => {
    if (mode !== 'play' || paused) return;
    const p = pos(e), b = inTB(p);
    if (b) { press(b); e.stopImmediatePropagation(); return; }
    if (!G.on || !onTable(p)) return;
    G.drag = {x: p.x, y: p.y, t: performance.now(), aim: aimAng, line: G.line && !!G.point};
    if (G.drag.line) { e.stopImmediatePropagation(); cv.setPointerCapture(e.pointerId); }
  }, {capture: true});
  cv.addEventListener('pointermove', e => {
    if (!G.drag || !G.drag.line) return;
    const p = pos(e), dx = p.x - G.point[0], dy = p.y - G.point[1], m = Math.hypot(dx, dy);
    if (m > 3) { G.dir = [dx / m, dy / m]; G.label = null; }
    e.stopImmediatePropagation();
  }, {capture: true});
  cv.addEventListener('pointerup', e => {
    if (!G.drag) return; const d = G.drag; G.drag = null; const p = pos(e);
    const isTap = Math.hypot(p.x - d.x, p.y - d.y) < 8 && performance.now() - d.t < 400;
    if (isTap && tap(d)) { if (!d.line) aimAng = d.aim; }
    if (d.line) e.stopImmediatePropagation();
  }, {capture: true});

  // ---- drawing ----
  function drawToolbar() {
    ctx.save(); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.font = 'bold 13px Arial';
    const lbl = {guide: G.on ? 'GUIDE ON' : 'GUIDE OFF', line: 'LINE', l2: '◀◀', l1: '◀', r1: '▶', r2: '▶▶',
      pm: 'PWR−', pw: 'power ' + Math.round(G.power * 100) + '%', pp: 'PWR+', bm: '−', bn: G.bounces + ' bounce', bp: '+'};
    for (const b of TB) {
      if (!G.on && b.id !== 'guide') continue;
      const lit = (b.id === 'guide' && G.on) || (b.id === 'line' && G.line);
      const plain = b.id === 'pw' || b.id === 'bn';
      if (!plain) { ctx.fillStyle = lit ? 'rgba(40,200,90,.9)' : 'rgba(10,16,28,.82)'; ctx.beginPath(); ctx.roundRect(b.x - b.w / 2, BY - 13, b.w, 26, 8); ctx.fill();
        ctx.strokeStyle = 'rgba(255,255,255,.35)'; ctx.lineWidth = 1; ctx.stroke(); }
      ctx.fillStyle = '#fff'; ctx.fillText(lbl[b.id], b.x, BY + 1);
    }
    ctx.restore();
  }

  function drawGuideMode() {
    if (!G.on || !G.point) return;
    const c = G.point, lb = lockedBall();
    ctx.save(); ctx.lineCap = 'round';
    ctx.strokeStyle = 'rgba(190,120,255,.85)'; ctx.lineWidth = 2;
    for (const a of AIM) { ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(c[0], c[1]); ctx.stroke(); }
    ctx.strokeStyle = '#3cff6e'; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.arc(c[0], c[1], R + 2, 0, 7); ctx.stroke();
    if (G.dir) {
      const skip = [cue(), lb].filter(Boolean), r = path(c, G.dir, Math.max(G.bounces, 1), objSpeed(c, G.dir));
      const blk = firstBlock(r.pts, skip);
      // where the cue ball has to hit
      const gx = c[0] - G.dir[0] * D, gy = c[1] - G.dir[1] * D, cb = cue();
      ctx.strokeStyle = '#ffe100'; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.arc(gx, gy, R, 0, 7); ctx.stroke();
      const L = Math.hypot(gx - cb.x, gy - cb.y);
      if (L > D) { const ux = (gx - cb.x) / L, uy = (gy - cb.y) / L, cblk = firstBlock([[cb.x, cb.y], [gx, gy]], [cb, lb].filter(Boolean));
        ctx.setLineDash([8, 6]); ctx.strokeStyle = cblk ? '#ff3a3a' : '#ffe100';
        ctx.beginPath(); ctx.moveTo(cb.x + ux * R, cb.y + uy * R); ctx.lineTo(gx - ux * R, gy - uy * R); ctx.stroke(); ctx.setLineDash([]); }
      ctx.lineWidth = 3.5;
      for (let i = 0; i + 1 < r.pts.length; i++) {
        const a = r.pts[i], b = r.pts[i + 1];
        if (blk && i === blk.seg) { ctx.strokeStyle = '#00e5fe'; ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(blk.at[0], blk.at[1]); ctx.stroke();
          ctx.strokeStyle = '#ff3a3a'; ctx.beginPath(); ctx.moveTo(blk.at[0], blk.at[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); }
        else { ctx.strokeStyle = blk && i > blk.seg ? '#ff3a3a' : '#00e5fe'; ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); }
      }
      const end = r.pts[r.pts.length - 1];
      if (blk) { ctx.strokeStyle = '#ff3a3a'; ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(blk.q.x, blk.q.y, R + 4, 0, 7); ctx.stroke(); }
      else if (r.pot >= 0) { ctx.strokeStyle = '#29fe51'; ctx.lineWidth = 4; ctx.beginPath(); ctx.arc(end[0], end[1], R * 1.6, 0, 7); ctx.stroke(); }
      else if (r.stop) { ctx.setLineDash([5, 4]); ctx.strokeStyle = '#ffaa00'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(end[0], end[1], R, 0, 7); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = '#ffaa00'; ctx.font = 'bold 13px Arial'; ctx.fillText('too soft', end[0] + R + 4, end[1] + 4); }
      if (G.label) { ctx.font = 'bold 15px Arial'; ctx.fillStyle = /BLOCK|no way/.test(G.label) ? '#ff5050' : '#29fe51';
        ctx.fillText(G.label, c[0] + R + 8, c[1] - R - 6); }
    }
    ctx.restore();
  }

  const baseDraw = draw;
  draw = function () {
    baseDraw();
    if (mode === 'play' && !paused && balls.length) {
      ctx.save(); ctx.setTransform(dpr * scale, 0, 0, dpr * scale, 0, 0);
      drawGuideMode(); drawToolbar(); ctx.restore();
    }
  };
})();
