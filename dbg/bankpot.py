"""Find bank-shot pots with PoolSight itself, shoot them, check if they drop."""
import json, math, sys, time
import cv2, numpy as np
sys.path.insert(0, '/home/cobalt_y/poolsight'); sys.path.insert(0, '/home/cobalt_y/poolsight/dbg')
import drive, poolsight as P
from series import fresh, shoot, balls_in, track
from physics import film

def predict_at(aim):
    drive.drag(aim[0] + 2, aim[1] + 2, aim[0], aim[1], steps=5); drive.move(80, 600); time.sleep(0.3)
    img = drive.grab(); res, st = fresh(img); res, st = P.analyse(img)
    return img, res, st

def main(n, power):
    drive.raise_chrome(); log = []
    for shot in range(n):
        time.sleep(0.5)
        res, st = fresh(drive.grab())
        if not res or res[2] is None:
            drive.drag(960, 600, 963, 603, steps=3); drive.move(80, 600); time.sleep(0.5)
            res, st = fresh(drive.grab())
            if not res or res[2] is None:
                print(shot, 'no cue'); shoot(0.3); time.sleep(6); continue
        t, balls, ci, _ = res; o = np.array(t.off, float); r = t.r
        allb = [np.array(b[:2]) + o for b in balls]; cue = allb[ci]
        found = None; t_start = time.time()
        order = sorted([i for i in range(len(allb)) if i != ci], key=lambda i: np.linalg.norm(allb[i] - cue))
        for i in order:
            B = allb[i]; u = (B - cue) / np.linalg.norm(B - cue); nrm = np.array([-u[1], u[0]])
            for k in np.linspace(-0.85, 0.85, 13):
                img, res, st = predict_at(B + nrm * k * 2 * r)
                if res and res[3] and res[3]['pot'] is not None and len(res[3]['obj_path']) == 3:
                    found = (img, res, i, k); break
                if time.time() - t_start > 25: break     # stay inside the game's shot clock
            if found or time.time() - t_start > 25: break
        if not found:
            print(shot, 'no bank pot found quickly; plain shot'); shoot(0.3); time.sleep(6); continue
        img, res, i, k = found
        t, balls, ci, pr = res; o = np.array(t.off, float)
        start = [np.array(b[:2]) + o for b in balls]
        oi = int(np.argmin([np.linalg.norm(b - (pr['obj_ball'] + o)) for b in start]))
        cv2.imwrite(f'dbg/bp_before{shot}.png', P.draw_cv(img, res)); cv2.imwrite(f'dbg/bp_raw{shot}.png', img)
        shoot(power)
        frames, _ = film(5.0)
        dets = [balls_in(f) for f in frames]
        tracks = track(dets, start, r)
        otr = tracks[oi]
        on_end = any(np.linalg.norm(d - otr[-1]) < r for d in dets[-1])
        pk = np.array(pr['pot']) + o
        e = {'shot': shot, 'bounces': len(pr['obj_path']) - 2, 'pred_pocket': pk.round(0).tolist(),
             'potted': bool((not on_end) and np.linalg.norm(otr[-1] - pk) < 5 * r),
             'obj_last': otr[-1].round(0).tolist(), 'obj_on_table': bool(on_end),
             'pred_path': [(q + o).round(0).tolist() for q in pr['obj_path']]}
        np.save(f'dbg/bp_track{shot}.npy', otr)
        print(json.dumps(e), flush=True); log.append(e)
        json.dump(log, open('dbg/bankpot.json', 'w'), indent=1); time.sleep(3)

if __name__ == '__main__':
    main(int(sys.argv[1]), float(sys.argv[2]))
