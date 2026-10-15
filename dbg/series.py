"""Play a series of real shots at chosen cut angles; log predicted vs actual."""
import json
import math
import sys
import time

import mss
import numpy as np
from Xlib import X
from Xlib.ext import xtest

sys.path.insert(0, '/home/cobalt_y/poolsight')
sys.path.insert(0, '/home/cobalt_y/poolsight/dbg')
import drive  # noqa: E402
import poolsight as P  # noqa: E402

BAR_X, BAR_TOP, BAR_BOT = 222, 425, 930


def fresh(img):
    for k in ('own', 'own_crop', 'balls', 'table', 'smooth'):
        P.STATE[k] = None
    P.STATE['own'] = P.own_mask(img)          # ignore PoolSight's own lines on screen
    return P.analyse(img)


def balls_in(img):
    P.STATE['balls'] = None
    P.STATE['own'] = P.own_mask(img)
    res, _ = P.analyse(img)
    if not res:
        return []
    o = np.array(res[0].off, float)
    return [np.array(b[:2]) + o for b in res[1]]


def track(dets, start, r):
    tracks = [[p] for p in start]
    for ds in dets:
        guess = [tr[-1] + (tr[-1] - tr[-2]) if len(tr) > 1 else tr[-1] for tr in tracks]
        pairs = sorted((np.linalg.norm(g - d), i, j) for i, g in enumerate(guess) for j, d in enumerate(ds))
        ut, ud = set(), set()
        for dist, i, j in pairs:
            if i in ut or j in ud or dist > 3 * r:
                continue
            tracks[i].append(ds[j])
            ut.add(i)
            ud.add(j)
        for i, tr in enumerate(tracks):
            if i not in ut:
                tr.append(tr[-1])
    return [np.array(t) for t in tracks]


def start_index(tr, r):
    for k in range(len(tr)):
        if np.linalg.norm(tr[k] - tr[0]) > 0.25 * r:
            return k
    return None


def dir_from(tr, k0, r, dist=3.0):
    p0 = tr[k0]
    for k in range(k0, len(tr)):
        if np.linalg.norm(tr[k] - p0) > dist * r:
            v = tr[k] - p0
            return math.degrees(math.atan2(v[1], v[0]))
    return None


def angdiff(a, b):
    return None if a is None or b is None else (a - b + 180) % 360 - 180


def bar_geometry():
    """Power bar position on screen (from PoolSight's own detection when available)."""
    pb = P.STATE.get('pbar')
    if pb is not None:
        x0, y0, x1, y1 = pb
        return (x0 + x1) // 2, y0 + 15, y1 - 5
    return BAR_X, BAR_TOP, BAR_BOT


def shoot(power):
    bx, top, bot = bar_geometry()
    y1 = top + power * (bot - top)
    drive.move(bx, top + 5)
    time.sleep(0.05)
    xtest.fake_input(drive.d, X.ButtonPress, 1)
    drive.d.sync()
    for k in range(1, 11):
        drive.move(bx, top + 5 + (y1 - top - 5) * k / 10)
        time.sleep(0.02)
    time.sleep(0.1)
    xtest.fake_input(drive.d, X.ButtonRelease, 1)
    drive.d.sync()
    drive.move(80, 600)


def one_shot(k_off, power, log):
    img = drive.grab()
    res, st = fresh(img)
    if not res or res[2] is None:
        print('cannot see cue ball:', st)
        return False
    t, balls, ci, _ = res
    o = np.array(t.off, float)
    r = t.r
    cue = np.array(balls[ci][:2]) + o
    allb = [np.array(b[:2]) + o for b in balls]
    ix0, iy0, ix1, iy1 = [v + o[i % 2] for i, v in enumerate(t.inner)]
    best = None
    for i, B in enumerate(allb):
        if i == ci:
            continue
        L = np.linalg.norm(B - cue)
        u = (B - cue) / L
        nrm = np.array([-u[1], u[0]])
        blocked = any(abs((q - cue) @ nrm) < 2.2 * r and 0 < (q - cue) @ u < L
                      for j, q in enumerate(allb) if j not in (i, ci))
        space = min([np.linalg.norm(B - q) for j, q in enumerate(allb) if j not in (i, ci)] +
                    [B[0] - ix0, ix1 - B[0], B[1] - iy0, iy1 - B[1]])
        if not blocked and 120 < L < 900 and space > 2.3 * r and (best is None or space > best[0]):
            best = (space, i, B, u, nrm)
    if best is None:
        print('no clear target')
        return False
    _, ti, B, u, nrm = best
    aim_pt = B + nrm * k_off * 2 * r          # k_off = sin(cut angle)
    drive.drag(aim_pt[0] + 3, aim_pt[1] + 3, aim_pt[0], aim_pt[1], steps=6)
    drive.move(80, 600)
    time.sleep(0.5)
    img = drive.grab()
    res, st = fresh(img)
    res, st = P.analyse(img)
    if not res or not res[3]:
        print('no prediction:', st)
        return False
    t, balls, ci, pr = res
    o = np.array(t.off, float)
    allb = [np.array(b[:2]) + o for b in balls]
    pred_obj = None if pr['obj_ball'] is None else pr['obj_ball'] + o
    vec = lambda pts: (math.degrees(math.atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0]))
                       if len(pts) > 1 else None)
    entry = {'k': k_off, 'power': power, 'status': st, 'aim': vec(pr['cue_path']),
             'pred_obj_dir': vec(pr['obj_path']), 'pred_cue_after': vec(pr['cue_after']),
             'ghost': (pr['cue_path'][1] + o).tolist(), 'hint': pr.get('hint', False)}
    shoot(power)
    frames = []
    with mss.MSS() as s:
        t0 = time.time()
        while time.time() - t0 < 4.0:
            frames.append(np.ascontiguousarray(np.asarray(s.grab(s.monitors[0]))[..., :3]))
    dets = [balls_in(f) for f in frames]
    tracks = track(dets, allb, r)
    ci_tr = tracks[ci]
    entry['cue_dir_actual'] = dir_from(ci_tr, 0, r)
    if pred_obj is not None:
        oi = int(np.argmin([np.linalg.norm(b - pred_obj) for b in allb]))
        otr = tracks[oi]
        k0 = start_index(otr, r)
        if k0 is not None:
            entry['obj_dir_actual'] = dir_from(otr, k0, r)
            entry['cue_after_actual'] = dir_from(ci_tr, k0, r, dist=2.0)
            entry['contact_actual'] = ci_tr[k0].tolist()
            # how the cue ball's speed splits right after contact (for follow/stun)
            k1 = min(k0 + 3, len(ci_tr) - 1)
            entry['cue_after_disp'] = (ci_tr[k1] - ci_tr[k0]).tolist()
            entry['obj_after_disp'] = (otr[k1] - otr[k0]).tolist()
    entry['err_obj'] = angdiff(entry.get('obj_dir_actual'), entry['pred_obj_dir'])
    entry['err_cue_after'] = angdiff(entry.get('cue_after_actual'), entry['pred_cue_after'])
    print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in entry.items()
                      if k not in ('ghost', 'contact_actual')}))
    log.append(entry)
    time.sleep(1.0)
    return True


if __name__ == '__main__':
    drive.raise_chrome()
    plan = [(0.0, 0.3), (0.3, 0.3), (0.5, 0.3), (0.7, 0.3), (-0.5, 0.3), (0.85, 0.3),
            (0.5, 0.6), (0.3, 0.15)]
    log = []
    for k, pw in plan:
        one_shot(k, pw, log)
    json.dump(log, open('dbg/series.json', 'w'), indent=1)
