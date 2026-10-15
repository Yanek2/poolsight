"""Measure the game's real physics with dedicated shots.
  physics.py bank N    - cue ball into a cushion: incoming vs outgoing angle
"""
import json
import math
import sys
import time

import cv2
import mss
import numpy as np

sys.path.insert(0, '/home/cobalt_y/poolsight')
sys.path.insert(0, '/home/cobalt_y/poolsight/dbg')
import drive  # noqa: E402
import poolsight as P  # noqa: E402
from series import fresh, shoot  # noqa: E402


def film(sec=4.0):
    frames, ts = [], []
    with mss.MSS() as s:
        t0 = time.time()
        while time.time() - t0 < sec:
            frames.append(np.ascontiguousarray(np.asarray(s.grab(s.monitors[0]))[..., :3]))
            ts.append(time.time() - t0)
    return frames, ts


def white_track(frames, start, r):
    """Follow the white cue ball only (bright, unsaturated blob of ball size)."""
    pos = [np.array(start, float)]
    for f in frames:
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        m = ((hsv[..., 1] < 45) & (hsv[..., 2] > 200)).astype(np.uint8)
        n, _, st, ce = cv2.connectedComponentsWithStats(m)
        best = None
        for i in range(1, n):
            a = st[i, 4]
            if not (0.8 * r * r < a < 4.2 * r * r):
                continue
            w, h = st[i, 2], st[i, 3]
            if not 0.6 < w / max(h, 1) < 1.6:
                continue
            d = np.linalg.norm(ce[i] - pos[-1])
            if d < 6 * r and (best is None or d < best[0]):
                best = (d, ce[i])
        pos.append(best[1] if best is not None else pos[-1])
    return np.array(pos)


def segments(tr, r):
    """Split a track into straight runs; return list of (direction deg, start, end)."""
    pts = [tr[0]]
    for p in tr[1:]:
        if np.linalg.norm(p - pts[-1]) > 0.5:
            pts.append(p)
    pts = np.array(pts)
    if len(pts) < 4:
        return []
    out, s0 = [], 0
    for k in range(2, len(pts)):
        a = pts[k - 1] - pts[s0]
        b = pts[k] - pts[k - 1]
        if np.linalg.norm(a) > 2 * r and np.linalg.norm(b) > 1 and \
                (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)) < math.cos(math.radians(15)):
            out.append((pts[s0], pts[k - 1]))
            s0 = k - 1
    out.append((pts[s0], pts[-1]))
    return [(math.degrees(math.atan2(e[1] - s[1], e[0] - s[0])), s, e) for s, e in out
            if np.linalg.norm(e - s) > 2 * r]


def bank(n):
    drive.raise_chrome()
    log = []
    angles = [30, 45, 60, 75, 20, 50, 70, 40][:n]
    for k, inc in enumerate(angles):
        time.sleep(0.5)
        img = drive.grab()
        res, st = fresh(img)
        if not res or res[2] is None:
            drive.drag(960, 600, 963, 603, steps=3)
            drive.move(80, 600)
            time.sleep(0.5)
            res, st = fresh(drive.grab())
            if not res or res[2] is None:
                print('no cue ball', st)
                shoot(0.3)
                time.sleep(6)
                continue
        t, balls, ci, _ = res
        o = np.array(t.off, float)
        r = t.r
        cue = np.array(balls[ci][:2]) + o
        ix0, iy0, ix1, iy1 = [v + o[i % 2] for i, v in enumerate(t.inner)]
        # choose the cushion that is farthest away vertically, aim at it with incidence `inc`
        best = None
        for wall, wy in (('t', iy0), ('b', iy1)):
            dy = wy - cue[1]
            for sgn in (1, -1):
                dx = abs(dy) * math.tan(math.radians(90 - inc)) * sgn
                hit = np.array([cue[0] + dx, wy])
                if not ix0 + 4 * r < hit[0] < ix1 - 4 * r:
                    continue
                others = [np.array(b[:2]) + o for j, b in enumerate(balls) if j != ci]
                u = (hit - cue) / np.linalg.norm(hit - cue)
                blocked = any(abs((q - cue) @ np.array([-u[1], u[0]])) < 2.2 * r and
                              0 < (q - cue) @ u < np.linalg.norm(hit - cue) for q in others)
                if not blocked and (best is None or abs(dy) > best[0]):
                    best = (abs(dy), hit, wall)
        if best is None:
            print('no clear bank line for', inc)
            continue
        _, hit, wall = best
        drive.drag(hit[0] + 2, hit[1] + 2, hit[0], hit[1], steps=6)
        drive.move(80, 600)
        time.sleep(0.35)
        img = drive.grab()
        res, st = fresh(img)
        res, st = P.analyse(img)
        if not res or not res[3]:
            print('no prediction', st)
            shoot(0.3)
            time.sleep(6)
            continue
        pr = res[3]
        o = np.array(res[0].off, float)
        cp = [q + o for q in pr['cue_path']]
        pred_in = math.degrees(math.atan2(*(cp[1] - cp[0])[::-1]))
        pred_out = math.degrees(math.atan2(*(cp[2] - cp[1])[::-1])) if len(cp) > 2 else None
        shoot(0.3)
        frames, ts = film(3.5)
        tr = white_track(frames, cue, r)
        segs = segments(tr, r)
        e = {'inc': inc, 'wall': wall, 'pred_in': pred_in, 'pred_out': pred_out,
             'segs': [(round(sd, 2), s.round(1).tolist(), en.round(1).tolist()) for sd, s, en in segs],
             'pred_bounce_pt': cp[1].round(1).tolist()}
        if len(segs) >= 2:
            e['act_in'], e['act_out'] = segs[0][0], segs[1][0]
            e['act_bounce_pt'] = segs[0][2].round(1).tolist()
            # angle to the cushion normal, in and out
            ni = abs(math.sin(math.radians(e['act_in'])))
            no = abs(math.sin(math.radians(e['act_out'])))
            e['in_from_rail'] = math.degrees(math.asin(min(ni, 1)))
            e['out_from_rail'] = math.degrees(math.asin(min(no, 1)))
            e['pred_out_err'] = None if pred_out is None else (e['act_out'] - pred_out + 180) % 360 - 180
        print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in e.items() if k != 'segs'}),
              flush=True)
        log.append(e)
        json.dump(log, open('dbg/bank.json', 'w'), indent=1)
        time.sleep(3)


if __name__ == '__main__':
    if sys.argv[1] == 'bank':
        bank(int(sys.argv[2]))


def cuts(n):
    from series import fresh as fr
    drive.raise_chrome()
    log = []
    offs = [0.3, 0.5, 0.7, 0.4, 0.6, 0.2, 0.8, 0.5][:n]
    for k_off in offs:
        time.sleep(0.5)
        img = drive.grab()
        res, st = fr(img)
        if not res or res[2] is None:
            drive.drag(960, 600, 963, 603, steps=3)
            drive.move(80, 600)
            time.sleep(0.5)
            res, st = fr(drive.grab())
            if not res or res[2] is None:
                shoot(0.3)
                time.sleep(6)
                continue
        t, balls, ci, _ = res
        o = np.array(t.off, float)
        r = t.r
        allb = [np.array(b[:2]) + o for b in balls]
        cue = allb[ci]
        ix0, iy0, ix1, iy1 = [v + o[i % 2] for i, v in enumerate(t.inner)]
        best = None
        for i, B in enumerate(allb):
            if i == ci:
                continue
            L = np.linalg.norm(B - cue)
            u = (B - cue) / L
            nrm = np.array([-u[1], u[0]])
            others = [q for j, q in enumerate(allb) if j not in (i, ci)]
            blocked = any(abs((q - cue) @ nrm) < 2.2 * r and 0 < (q - cue) @ u < L for q in others)
            room = min([np.linalg.norm(B - q) for q in others] + [B[0] - ix0, ix1 - B[0], B[1] - iy0, iy1 - B[1]])
            if not blocked and 150 < L < 700 and (best is None or room > best[0]):
                best = (room, B, u, nrm)
        if best is None:
            shoot(0.3)
            time.sleep(6)
            continue
        _, B, u, nrm = best
        aim = B + nrm * k_off * 2 * r
        drive.drag(aim[0] + 2, aim[1] + 2, aim[0], aim[1], steps=6)
        drive.move(80, 600)
        time.sleep(0.35)
        img = drive.grab()
        res, st = fr(img)
        res, st = P.analyse(img)
        if not res or not res[3] or res[3]['ghost'] is None:
            shoot(0.3)
            time.sleep(6)
            continue
        pr = res[3]
        o = np.array(res[0].off, float)
        ca = pr['cue_after']
        pred_after = (math.degrees(math.atan2(ca[1][1] - ca[0][1], ca[1][0] - ca[0][0]))
                      if len(ca) > 1 else None)
        g = pr['ghost'] + o
        ob = pr['obj_ball'] + o if pr['obj_ball'] is not None else None
        shoot(0.3)
        frames, ts = film(3.5)
        tr = white_track(frames, cue, r)
        segs = segments(tr, r)
        e = {'k': k_off, 'pred_after': pred_after}
        if len(segs) >= 2 and ob is not None:
            vin = math.radians(segs[0][0])
            n = (ob - g) / np.linalg.norm(ob - g)
            cut = math.degrees(math.acos(abs(math.cos(vin) * n[0] + math.sin(vin) * n[1])))
            e['cut'] = cut
            e['act_in'], e['act_after'] = segs[0][0], segs[1][0]
            tdir = np.array([math.cos(vin), math.sin(vin)])
            tang = tdir - (tdir @ n) * n
            tang /= np.linalg.norm(tang)
            a = np.array([math.cos(math.radians(e['act_after'])), math.sin(math.radians(e['act_after']))])
            dev = math.degrees(math.atan2(a @ n, a @ tang))     # + = toward the object ball's direction
            e['dev_toward_n'] = dev
            e['k_follow'] = math.tan(math.radians(dev)) * math.tan(math.radians(cut))
            e['err'] = None if pred_after is None else (e['act_after'] - pred_after + 180) % 360 - 180
        print(json.dumps({kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in e.items()}), flush=True)
        log.append(e)
        json.dump(log, open('dbg/cuts.json', 'w'), indent=1)
        time.sleep(3)


if __name__ == '__main__' and sys.argv[1] == 'cuts':
    cuts(int(sys.argv[2]))


def speed(powers):
    """Cue ball alone at different powers: initial speed, slowdown, cushion loss."""
    drive.raise_chrome()
    log = []
    for pw in powers:
        time.sleep(0.5)
        img = drive.grab()
        res, st = fresh(img)
        if not res or res[2] is None:
            drive.drag(960, 600, 963, 603, steps=3); drive.move(80, 600); time.sleep(0.5)
            res, st = fresh(drive.grab())
            if not res or res[2] is None:
                shoot(0.2); time.sleep(6); continue
        t, balls, ci, _ = res
        o = np.array(t.off, float); r = t.r
        cue = np.array(balls[ci][:2]) + o
        others = [np.array(b[:2]) + o for j, b in enumerate(balls) if j != ci]
        ix0, iy0, ix1, iy1 = [v + o[i % 2] for i, v in enumerate(t.inner)]
        # longest clear straight run from the cue ball (try many directions)
        best = None
        for a in np.radians(np.arange(0, 360, 5)):
            u = np.array([math.cos(a), math.sin(a)])
            ts_ = [((ix1 if u[0] > 0 else ix0) - cue[0]) / u[0] if abs(u[0]) > 1e-6 else 1e9,
                   ((iy1 if u[1] > 0 else iy0) - cue[1]) / u[1] if abs(u[1]) > 1e-6 else 1e9]
            L = min(ts_)
            blocked = any(abs((q - cue) @ np.array([-u[1], u[0]])) < 2.2 * r and 0 < (q - cue) @ u < L for q in others)
            if not blocked and (best is None or L > best[0]):
                best = (L, cue + u * L)
        L, hit = best
        drive.drag(hit[0] + 2, hit[1] + 2, hit[0], hit[1], steps=6); drive.move(80, 600); time.sleep(0.35)
        shoot(pw)
        frames, ts = film(7.0)
        tr = white_track(frames, cue, r)[1:]
        ts = np.array(ts)
        # speeds (px/s) smoothed over 3 frames
        d = np.linalg.norm(np.diff(tr, axis=0), axis=1)
        dt = np.diff(ts)
        v = np.convolve(d, np.ones(3), 'same') / np.convolve(dt, np.ones(3), 'same')
        moving = np.flatnonzero(v > 20)
        if len(moving) == 0:
            print('did not move'); continue
        k0, k1 = moving[0], moving[-1]
        path_len = d[k0:k1 + 1].sum()
        e = {'power': pw, 'run_to_cushion': float(L), 'v0': float(v[k0 + 1:k0 + 4].max()),
             'travel': float(path_len), 'time': float(ts[k1] - ts[k0]),
             'v_series': v[k0:k1 + 1:2].round(0).tolist(), 't_series': (ts[k0:k1 + 1:2] - ts[k0]).round(3).tolist(),
             'track': tr[k0:k1 + 1:2].round(1).tolist()}
        print(json.dumps({kk: vv for kk, vv in e.items() if kk not in ('v_series', 't_series', 'track')}), flush=True)
        log.append(e)
        json.dump(log, open('dbg/speed.json', 'w'), indent=1)
        time.sleep(2)


if __name__ == '__main__' and sys.argv[1] == 'speed':
    speed([float(x) for x in sys.argv[2:]])
