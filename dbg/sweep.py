"""Cut-angle sweep + jitter measurement on the live game."""
import math
import sys
import time

import numpy as np

sys.path.insert(0, '/home/cobalt_y/poolsight/dbg')
import drive  # noqa: E402
from live_test import state, frame  # noqa: E402
import poolsight as P  # noqa: E402

drive.raise_chrome()
img, s = state(4)
res = s['res']
t, balls, ci, pr = res
o = np.array(t.off, float)
r = t.r
cue = s['cue']
# pick a target ball with a clear straight path from the cue ball
best = None
for i, b in enumerate(balls):
    if i == ci:
        continue
    B = np.array(b[:2]) + o
    v = B - cue
    L = np.linalg.norm(v)
    u = v / L
    clear = all(abs((np.array(q[:2]) + o - cue) @ np.array([-u[1], u[0]])) > 3 * r or
                not 0 < (np.array(q[:2]) + o - cue) @ u < L - r
                for j, q in enumerate(balls) if j not in (i, ci))
    if clear and 250 < L < 900 and (best is None or L < best[0]):
        best = (L, B)
L, B = best
u = (B - cue) / L
nrm = np.array([-u[1], u[0]])
print(f'cue {cue.round(1)} target {B.round(1)} dist {L:.0f} r {r:.1f}')
offsets = [0, 0.5, 1.0, 1.4, 1.7, 1.85, 1.95, -1.0, -1.7, -1.95]
for k in offsets:
    P_ = B + nrm * k * r
    drive.drag(P_[0] + 4, P_[1] + 4, P_[0], P_[1], steps=6)
    drive.move(P_[0] + 400, P_[1] + 150)        # get the pointer off the line
    time.sleep(0.25)
    rows = []
    for _ in range(12):
        _, sx = state(1)
        rows.append(sx)
    st = [x['st'] for x in rows]
    ok = [x for x in rows if 'aim' in x]
    line = f'offset {k:+.2f}r: ' + ', '.join(f'{v}x{st.count(v)}' for v in sorted(set(st)))
    if ok:
        aims = np.array([x['aim'] for x in ok])
        gh = np.array([x['ghost'] for x in ok])
        line += f' | aim sd {aims.std():.3f} deg  ghost sd {gh.std(axis=0).max():.2f}px'
        od = [x['obj_dir'] for x in ok if 'obj_dir' in x]
        if od:
            line += f' | obj dir {np.mean(od):.1f} sd {np.std(od):.2f} deg  hint {sum(x.get("hint", False) for x in ok)}/{len(ok)}'
            lens = [np.linalg.norm(x['res'][3]['obj_path'][-1] - x['res'][3]['obj_path'][0]) for x in ok if 'obj_dir' in x]
            line += f' len {np.mean(lens):.0f}px'
        else:
            line += ' | NO OBJECT LINE'
    print(line, flush=True)
