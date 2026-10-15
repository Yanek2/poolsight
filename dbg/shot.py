"""Play a real shot and compare PoolSight's prediction with what actually happens.

usage: shot.py POWER(0..1) [aim_x aim_y]
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

BAR_X, BAR_TOP, BAR_BOT = 222, 425, 930


def analyse_full(img):
    P.STATE['own'] = None
    P.STATE['own_crop'] = None
    P.STATE['balls'] = None
    P.STATE['table'] = None
    return P.analyse(img)


def balls_in(img):
    """All ball centres (screen coords) in a frame, no aim needed."""
    P.STATE['balls'] = None
    res, st = P.analyse(img)
    if not res:
        return []
    t = res[0]
    o = np.array(t.off, float)
    return [np.array(b[:2]) + o for b in res[1]]


def track(frames, start_pts, r):
    """Greedy nearest-neighbour tracks from the given start positions."""
    tracks = [[p] for p in start_pts]
    alive = [True] * len(start_pts)
    for dets in frames:
        dets = list(dets)
        pairs = sorted(((np.linalg.norm(tr[-1] - d), i, j) for i, tr in enumerate(tracks) if alive[i]
                        for j, d in enumerate(dets)), key=lambda x: x[0])
        used_t, used_d = set(), set()
        for dist, i, j in pairs:
            if i in used_t or j in used_d or dist > 5 * r:
                continue
            tracks[i].append(dets[j])
            used_t.add(i)
            used_d.add(j)
        for i in range(len(tracks)):
            if alive[i] and i not in used_t:
                tracks[i].append(tracks[i][-1])          # not seen: assume it stayed (or got potted)
    return tracks


def first_dir(tr, r, start=0):
    """Direction of travel once a ball has moved 3r from where it was at index `start`."""
    p0 = tr[start]
    for k in range(start, len(tr)):
        if np.linalg.norm(tr[k] - p0) > 3 * r:
            v = tr[k] - p0
            return math.degrees(math.atan2(v[1], v[0])), k
    return None, None


def main():
    power = float(sys.argv[1])
    drive.raise_chrome()
    if len(sys.argv) >= 4:
        ax, ay = float(sys.argv[2]), float(sys.argv[3])
        drive.drag(ax + 3, ay + 3, ax, ay, steps=6)
    drive.move(80, 600)
    time.sleep(0.4)
    img = drive.grab()
    for _ in range(2):
        res, st = analyse_full(img)
    if not res or not res[3]:
        print('no prediction:', st)
        return
    t, balls, ci, pr = res
    o = np.array(t.off, float)
    r = t.r
    cue0 = np.array(balls[ci][:2]) + o
    pred = {
        'aim': math.degrees(math.atan2(pr['aim'][1], pr['aim'][0])),
        'ghost': (pr['cue_path'][1] + o).tolist() if len(pr['cue_path']) > 1 else None,
        'obj': None if pr['obj_ball'] is None else (pr['obj_ball'] + o).tolist(),
        'obj_path': [(q + o).tolist() for q in pr['obj_path']],
        'cue_after': [(q + o).tolist() for q in pr['cue_after']],
        'cue_path': [(q + o).tolist() for q in pr['cue_path']],
        'pot': None if pr['pot'] is None else (np.array(pr['pot']) + o).tolist(),
        'scratch': pr['cue_pot'] is not None,
        'locked': pr['locked'],
    }
    cv2.imwrite('dbg/shot_before.png', P.draw_cv(img, res))
    all_balls = [np.array(b[:2]) + o for b in balls]

    # shoot: pull the power bar down and release
    y1 = BAR_TOP + power * (BAR_BOT - BAR_TOP)
    drive.move(BAR_X, BAR_TOP + 5)
    time.sleep(0.05)
    from Xlib import X
    from Xlib.ext import xtest
    xtest.fake_input(drive.d, X.ButtonPress, 1)
    drive.d.sync()
    for k in range(1, 11):
        drive.move(BAR_X, BAR_TOP + 5 + (y1 - BAR_TOP - 5) * k / 10)
        time.sleep(0.02)
    time.sleep(0.1)
    xtest.fake_input(drive.d, X.ButtonRelease, 1)
    drive.d.sync()
    drive.move(80, 600)

    # film it
    reg = {'left': int(t.off[0]), 'top': int(t.off[1]),
           'width': int(t.bx1 - t.bx0 + 2 * (t.bx0)), 'height': int(t.by1 - t.by0 + 2 * t.by0)}
    frames, stamps = [], []
    with mss.MSS() as s:
        t0 = time.time()
        while time.time() - t0 < 5.0:
            frames.append(np.ascontiguousarray(np.asarray(s.grab(s.monitors[0]))[..., :3]))
            stamps.append(time.time() - t0)
    print(f'filmed {len(frames)} frames ({len(frames) / 5:.0f} fps)')
    cv2.imwrite('dbg/shot_after.png', frames[-1])
    dets = [balls_in(f) for f in frames]
    tracks = track(dets, all_balls, r)

    # what happened
    cue_tr = tracks[[np.linalg.norm(b - cue0) for b in all_balls].index(min(np.linalg.norm(b - cue0) for b in all_balls))]
    out = {'pred': pred, 'r': r}
    cd, ck = first_dir(cue_tr, r)
    out['cue_dir_actual'] = cd
    if pred['obj'] is not None:
        oi = int(np.argmin([np.linalg.norm(b - np.array(pred['obj'])) for b in all_balls]))
        otr = tracks[oi]
        od, ok_ = first_dir(otr, r)
        out['obj_dir_actual'] = od
        pn = np.array(pred['obj_path'][1]) - np.array(pred['obj_path'][0]) if len(pred['obj_path']) > 1 else None
        out['obj_dir_pred'] = None if pn is None else math.degrees(math.atan2(pn[1], pn[0]))
        out['obj_final'] = otr[-1].tolist()
        out['obj_seen_at_end'] = any(np.linalg.norm(d - otr[-1]) < r for d in dets[-1])
        # cue ball direction after contact: when the object ball starts moving
        if ok_ is not None:
            k0 = next(k for k in range(len(otr)) if np.linalg.norm(otr[k] - otr[0]) > 0.3 * r)
            cad, _ = first_dir(cue_tr, r, start=min(k0, len(cue_tr) - 1))
            out['cue_after_actual'] = cad
            ca = pred['cue_after']
            out['cue_after_pred'] = (math.degrees(math.atan2(ca[1][1] - ca[0][1], ca[1][0] - ca[0][0]))
                                     if len(ca) > 1 else None)
    out['cue_final'] = cue_tr[-1].tolist()
    out['moved'] = [i for i, tr in enumerate(tracks) if np.linalg.norm(tr[-1] - tr[0]) > r]
    json.dump(out, open('dbg/shot_result.json', 'w'), indent=1)
    np.save('dbg/shot_tracks.npy', np.array([np.array(tr) for tr in tracks], dtype=object), allow_pickle=True)
    for k, v in out.items():
        if k != 'pred':
            print(k, v)
    print('pred obj_path', [tuple(np.round(q, 0)) for q in pred['obj_path']])
    print('pred pot', pred['pot'], 'scratch', pred['scratch'], 'locked', pred['locked'])


if __name__ == '__main__':
    main()
