#!/usr/bin/env python3
"""Regression check for PoolSight against hand-checked screenshots.

Each case: image, true cue-ball centre, true ghost-ball centre (the game's own
ring). Groups share calibration state like the live app would.
Run:  venv/bin/python eval.py [-v]
"""
import sys
import time

import cv2
import numpy as np

import poolsight as P

GROUPS = [
    # green UI (Chrome screenshots, 1568px wide)
    [('dbg/a1.jpg', (524, 403), (1048, 203)),
     ('dbg/a2.jpg', (524, 402), (955, 395)),
     ('dbg/a3.jpg', (524, 401), (620, 603))],
    # green UI, full-res screen capture
    [('dbg/live.png', (641.5, 693), (1195, 666))],
    # blue UI (Chrome screenshots)
    [('dbg/b1.jpg', (482, 424), (1034, 395)),
     ('dbg/b2.jpg', (482, 423), (1148, 186)),
     ('dbg/b3.jpg', (482, 423), (302, 186))],
    # blue UI, full-res screen captures with windows on top
    [('dbg/g0.png', (1218, 661), (1261, 652))],
    [('dbg/n0.png', (1442, 924), (1393, 913))],
    # newest UI: ghost drawn with a red X (aiming at the opponent's ball)
    [('dbg/t_x.png', (540, 756), (1305, 628))],
    # cue ball near a cushion (stick mostly over the rail) + 3 balls in a row
    [('dbg/t_rail.png', (1313, 938), (1453, 871))],
    [('dbg/t_8ball.png', (667.5, 439.5), (403, 669))],
    # stick lying across another ball (green 6) + a ball beside the stick
    [('dbg/t_stickover.png', (1247, 621), (1220, 689.5))],
    # user's F7 snapshot (new UI, off-centre stick): used to flip to 'unsure'
    [('dbg/t_user1.png', (804, 638), (428, 617))],
    # cue ball hidden under a window: must draw nothing rather than guess
    [('dbg/T0.png', None, None)],
]


def run(verbose=False):
    fails, n, times = 0, 0, []
    for grp in GROUPS:
        P.STATE.clear()
        P.STATE['r'] = None
        P.CAL.clear()
        for f, cue, ghost in grp:
            P.STATE['table'] = None
            img = cv2.imread(f)
            t0 = time.time()
            res, st = P.analyse(img)
            times.append((time.time() - t0) * 1000)
            n += 1
            if cue is None:                     # expected: no prediction
                ok = not res or not res[3]
                fails += not ok
                print(f"{'ok  ' if ok else 'FAIL'} {f}: expected nothing drawn, got '{st}'")
                continue
            if not res or not res[3]:
                print(f'FAIL {f}: {st}')
                fails += 1
                continue
            t, balls, ci, pr = res
            o = np.array(t.off, float)
            c = np.array(balls[ci][:2]) + o
            g = pr.get('game_ghost', pr['cue_path'][1] if len(pr['cue_path']) > 1 else c) + o
            ec, eg = np.linalg.norm(c - cue), np.linalg.norm(g - ghost)
            ok = ec < 4 and eg < 3
            fails += not ok
            print(f"{'ok  ' if ok else 'FAIL'} {f}: cue err {ec:5.1f}px  ghost err {eg:5.1f}px  "
                  f"{'locked' if pr['locked'] else 'unlocked'}  r={t.r:.1f} balls={len(balls)}")
            if verbose:
                print('     cue', c.round(1), 'first contact', g.round(1))
    print(f'\n{n - fails}/{n} passed, {np.mean(times):.0f} ms/frame avg')
    return fails


if __name__ == '__main__':
    sys.exit(run('-v' in sys.argv))
