"""Play real shots (both sides): pick a potting shot, check PoolSight's prediction
against what really happens. usage: autoplay.py N_SHOTS [power]"""
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
from series import fresh, balls_in, track, start_index, dir_from, angdiff, shoot  # noqa: E402


def seg_clear(a, b, pts, rad):
    v = b - a
    L = np.linalg.norm(v)
    if L < 1e-6:
        return True
    u = v / L
    for q in pts:
        s = (q - a) @ u
        if 0 < s < L and abs((q - a) @ np.array([-u[1], u[0]])) < rad:
            return False
    return True


def plan(res):
    t, balls, ci, _ = res
    o = np.array(t.off, float)
    r = t.r
    allb = [np.array(b[:2]) + o for b in balls]
    cue = allb[ci]
    pockets = [np.array(p[:2]) + o for p in t.pockets]
    best = []
    for i, B in enumerate(allb):
        if i == ci:
            continue
        others = [q for j, q in enumerate(allb) if j not in (i, ci)]
        for pk in pockets:
            to_p = pk - B
            dp = np.linalg.norm(to_p)
            if dp < 2 * r:
                continue
            g = B - to_p / dp * 2 * r                      # ghost ball position
            dc = g - cue
            L = np.linalg.norm(dc)
            if L < 3 * r:
                continue
            cut = math.degrees(math.acos(np.clip((dc / L) @ (to_p / dp), -1, 1)))
            if cut > 65:
                continue
            if not seg_clear(cue, g, others, 2 * r) or not seg_clear(B, pk, others, 2 * r):
                continue
            dark = balls[i][2] < -2.4      # very dark ball: probably the 8, avoid early
            best.append((cut + 0.02 * (L + dp) + (200 if dark else 0), g, i, pk, cut))
    best.sort(key=lambda x: x[0])
    return best, allb, r


def main():
    n_shots = int(sys.argv[1])
    power = float(sys.argv[2]) if len(sys.argv) > 2 else 0.45
    drive.raise_chrome()
    log = []
    for shot in range(n_shots):
        time.sleep(0.5)
        # wait for our turn (the computer plays in between): need a cue ball + aim line
        for _ in range(25):
            img = drive.grab()
            res, st = fresh(img)
            if res and res[2] is not None:
                break
            time.sleep(1.0)
        else:
            print(shot, 'never got a turn:', st)
            cv2.imwrite(f'dbg/ap_fail{shot}.png', img)
            continue
        cands, allb, r = plan(res)
        if not cands:
            print(shot, 'no potting shot found; skipping')
            cv2.imwrite(f'dbg/ap_noshot{shot}.png', img)
            break
        _, g, ti, pk, cut = cands[0]
        drive.drag(g[0] + 3, g[1] + 3, g[0], g[1], steps=6)
        drive.move(80, 600)
        time.sleep(0.35)
        img = drive.grab()
        P.STATE['smooth'] = None
        res, st = fresh(img)
        res2, st2 = P.analyse(img)
        res, st = (res2, st2) if res2 and res2[3] else (res, st)
        if not res or not res[3]:
            print(shot, 'no prediction', st)
            cv2.imwrite(f'dbg/ap_nopred{shot}.png', img)
            shoot(power)                      # keep the game moving
            time.sleep(6)
            continue
        t, balls, ci, pr = res
        o = np.array(t.off, float)
        cv2.imwrite(f'dbg/ap_before{shot}.png', P.draw_cv(img, res))
        cv2.imwrite(f'dbg/ap_raw{shot}.png', img)
        vec = lambda pts: (math.degrees(math.atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0]))
                           if len(pts) > 1 else None)
        pred_obj = None if pr['obj_ball'] is None else pr['obj_ball'] + o
        e = {'shot': shot, 'cut': round(cut, 1), 'target': allb[ti].tolist(), 'pocket': pk.tolist(),
             'status': st, 'pred_pot': None if pr['pot'] is None else (np.array(pr['pot']) + o).tolist(),
             'pred_obj_ball': None if pred_obj is None else pred_obj.tolist(),
             'pred_obj_dir': vec(pr['obj_path']), 'pred_cue_after': vec(pr['cue_after']),
             'hint': pr.get('hint', False),
             'pred_obj_path': [(q + o).tolist() for q in pr['obj_path']],
             'pred_cue_after_path': [(q + o).tolist() for q in pr['cue_after']],
             'pred_scratch': pr['cue_pot'] is not None}
        start = [np.array(b[:2]) + o for b in balls]
        shoot(power)
        frames = []
        with mss.MSS() as s:
            t0 = time.time()
            while time.time() - t0 < 5.0:
                frames.append(np.ascontiguousarray(np.asarray(s.grab(s.monitors[0]))[..., :3]))
        dets = [balls_in(f) for f in frames]
        tracks = track(dets, start, r)
        np.save(f'dbg/ap_tracks{shot}.npy', np.array(tracks, dtype=object), allow_pickle=True)
        cv2.imwrite(f'dbg/ap_after{shot}.png', frames[-1])
        tgt = int(np.argmin([np.linalg.norm(b - allb[ti]) for b in start]))
        otr = tracks[tgt]
        k0 = start_index(otr, r)
        e['obj_dir_actual'] = dir_from(otr, k0, r) if k0 is not None else None
        e['obj_moved'] = k0 is not None
        # potted = no longer on the table at the end, last seen near a pocket
        final = dets[-1]
        e['obj_on_table_end'] = any(np.linalg.norm(d - otr[-1]) < r for d in final)
        pockets = [np.array(p[:2]) + o for p in t.pockets]
        e['obj_last_pocket_dist'] = float(min(np.linalg.norm(otr[-1] - pk_) for pk_ in pockets))
        e['actual_pot'] = (not e['obj_on_table_end']) and e['obj_last_pocket_dist'] < 4 * r
        ctr = tracks[ci]
        e['cue_after_actual'] = dir_from(ctr, k0, r, 2.0) if k0 is not None else None
        e['cue_on_table_end'] = any(np.linalg.norm(d - ctr[-1]) < r for d in final)
        e['err_obj_dir'] = angdiff(e['obj_dir_actual'], e['pred_obj_dir'])
        e['err_cue_after'] = angdiff(e['cue_after_actual'], e['pred_cue_after'])
        log.append(e)
        print(shot, f"cut {cut:.0f}  pred pot {e['pred_pot'] is not None}  actual pot {e['actual_pot']}  "
                    f"obj err {e['err_obj_dir']}  cue-after pred {e['pred_cue_after']} actual {e['cue_after_actual']}  "
                    f"hint {e['hint']}", flush=True)
        json.dump(log, open('dbg/autoplay.json', 'w'), indent=1)
        time.sleep(2.5)                               # let everything settle / turn change


if __name__ == '__main__':
    main()
