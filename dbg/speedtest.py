"""Real shots with the power read off the bar: predicted vs actual rest positions / pots."""
import json
import math
import sys
import time

import cv2
import numpy as np
from Xlib import X
from Xlib.ext import xtest

sys.path.insert(0, '/home/cobalt_y/poolsight')
sys.path.insert(0, '/home/cobalt_y/poolsight/dbg')
import drive  # noqa: E402
import poolsight as P  # noqa: E402
from series import fresh, balls_in, track, shoot, BAR_X, BAR_TOP, BAR_BOT  # noqa: E402
from autoplay import plan  # noqa: E402
from physics import film  # noqa: E402


def main(n, powers):
    drive.raise_chrome()
    log = []
    for shot in range(n):
        p_pull = powers[shot % len(powers)]
        time.sleep(0.5)
        img = drive.grab()
        res, st = fresh(img)
        if not res or res[2] is None:
            drive.drag(960, 600, 963, 603, steps=3); drive.move(80, 600); time.sleep(0.5)
            res, st = fresh(drive.grab())
            if not res or res[2] is None:
                print(shot, 'no cue'); shoot(0.3); time.sleep(6); continue
        cands, allb, r = plan(res)
        if not cands:
            print(shot, 'no pot plan; plain shot'); shoot(0.3); time.sleep(6); continue
        _, g, ti, pk, cut = cands[0]
        drive.drag(g[0] + 3, g[1] + 3, g[0], g[1], steps=6); drive.move(80, 600); time.sleep(0.3)
        # pull the bar and hold
        y1 = BAR_TOP + p_pull * (BAR_BOT - BAR_TOP)
        drive.move(BAR_X, BAR_TOP + 5); time.sleep(0.05)
        xtest.fake_input(drive.d, X.ButtonPress, 1); drive.d.sync()
        for k in range(1, 11):
            drive.move(BAR_X, BAR_TOP + 5 + (y1 - BAR_TOP - 5) * k / 10); time.sleep(0.02)
        time.sleep(0.25)
        img = drive.grab()
        res, st = fresh(img)
        pb = P.STATE.get('pbar')
        P.STATE['power'] = P.read_power(img, pb) if pb else None
        res, st = P.analyse(img)
        xtest.fake_input(drive.d, X.ButtonRelease, 1); drive.d.sync(); drive.move(80, 600)
        frames, _ = film(6.0)
        if not res or not res[3]:
            print(shot, 'no prediction', st); time.sleep(2); continue
        t, balls, ci, pr = res
        o = np.array(t.off, float)
        cv2.imwrite(f'dbg/st_before{shot}.png', P.draw_cv(img, res))
        start = [np.array(b[:2]) + o for b in balls]
        dets = [balls_in(f) for f in frames[::2]]
        tracks = track(dets, start, r)
        e = {'shot': shot, 'power_read': P.STATE['power'], 'cut': round(cut, 1)}
        final = dets[-1]
        def gone(tr):
            return not any(np.linalg.norm(d - tr[-1]) < r for d in final)
        if pr['obj_ball'] is not None:
            oi = int(np.argmin([np.linalg.norm(b - (pr['obj_ball'] + o)) for b in start]))
            otr = tracks[oi]
            e['obj_pred_pot'] = pr['pot'] is not None
            e['obj_actual_pot'] = bool(gone(otr))
            if pr.get('obj_stops'):
                e['obj_pred_rest'] = (pr['obj_path'][-1] + o).round(0).tolist()
                e['obj_actual_rest'] = otr[-1].round(0).tolist()
                e['obj_rest_err_px'] = float(np.linalg.norm(otr[-1] - (pr['obj_path'][-1] + o)))
            e['obj_travel_actual'] = float(np.linalg.norm(np.diff(otr, axis=0), axis=1).sum())
            e['obj_travel_pred'] = float(sum(np.linalg.norm(pr['obj_path'][k + 1] - pr['obj_path'][k])
                                             for k in range(len(pr['obj_path']) - 1)))
        ctr = tracks[ci]
        e['cue_actual_rest'] = ctr[-1].round(0).tolist()
        if pr.get('cue_after_stops'):
            e['cue_pred_rest'] = (pr['cue_after'][-1] + o).round(0).tolist()
            e['cue_rest_err_px'] = float(np.linalg.norm(ctr[-1] - (pr['cue_after'][-1] + o)))
        print(json.dumps(e), flush=True)
        log.append(e)
        json.dump(log, open('dbg/speedtest.json', 'w'), indent=1)
        time.sleep(2)


if __name__ == '__main__':
    main(int(sys.argv[1]), [float(x) for x in sys.argv[2:]])
