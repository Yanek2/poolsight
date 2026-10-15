"""Live tests: drive the game with the real mouse, analyse frames in-process."""
import math
import sys
import time

import numpy as np

sys.path.insert(0, '/home/cobalt_y/poolsight')
sys.path.insert(0, '/home/cobalt_y/poolsight/dbg')
import drive  # noqa: E402
import poolsight as P  # noqa: E402


def frame():
    img = drive.grab()
    P.STATE['own'] = None
    P.STATE['own_crop'] = None
    res, st = P.analyse(img)
    return img, res, st


def state(n=3):
    """Analyse a few frames; return the last one with its summary."""
    for _ in range(n):
        img, res, st = frame()
    out = {'st': st}
    if res and res[3]:
        t, b, ci, pr = res
        o = np.array(t.off, float)
        out['cue'] = np.array(b[ci][:2]) + o
        d = pr['cue_path'][1] - pr['cue_path'][0]
        out['aim'] = math.degrees(math.atan2(d[1], d[0]))
        out['ghost'] = pr.get('game_ghost', pr['cue_path'][1]) + o
        out['locked'] = pr['locked']
        if len(pr['obj_path']) > 1:
            n_ = pr['obj_path'][1] - pr['obj_path'][0]
            out['obj_dir'] = math.degrees(math.atan2(n_[1], n_[0]))
            out['obj'] = pr['obj_ball'] + o
            out['hint'] = pr.get('hint', False)
        out['pot'] = pr['pot'] is not None
        out['res'] = res
    return img, out


if __name__ == '__main__':
    drive.raise_chrome()
    img, s0 = state()
    print('before', {k: v for k, v in s0.items() if k != 'res'})
    c = s0['cue']
    # drag across the table above the cue ball, left to right
    drive.drag(c[0] + 200, c[1] - 200, c[0] + 260, c[1] - 200)
    time.sleep(0.3)
    img, s1 = state()
    print('after drag right', {k: v for k, v in s1.items() if k != 'res'})
    drive.drag(c[0] + 300, c[1] + 100, c[0] + 300, c[1] + 160)
    time.sleep(0.3)
    img, s2 = state()
    print('after drag down', {k: v for k, v in s2.items() if k != 'res'})
