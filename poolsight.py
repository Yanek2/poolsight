#!/usr/bin/env python3
"""PoolSight - screen-reading aim predictor for the local 8 Ball Pool game.

It never touches the game's code: it only looks at the screen, finds the
table, the balls and the game's own short aim line, then draws a longer
prediction (object-ball path, cue-ball path, cushion bounces, pots) on a
click-through overlay. Table colour, ball size and cushion positions are
all measured from the screen, so it keeps working when the game's look changes.

    F8  show / hide overlay
    F9  quit

Run with --test IMAGE... to analyse screenshots and write IMAGE.out.png.
"""
import math
import sys
import threading

import cv2
import numpy as np

# ---------------------------------------------------------------- tuning
BAND_RATIO = 0.033         # default cushion width / felt height (until calibrated)
BALL_RATIO = 0.0115        # first guess of ball radius / felt width
CORNER_POCKET = 2.4        # pot if a wall hit is this many radii from a corner
MIDDLE_POCKET = 1.6        # ...or this many radii from a middle pocket
V_PER_POWER = 146.0         # cue speed (ball radii / s) per unit of power-bar fill   } measured
DECEL = 18.9                # rolling slowdown (ball radii / s^2)                    } in the game
K_OBJ = 0.9                 # share of the head-on speed the object ball gets
K_FOLLOW = 0.28             # after hitting a ball the cue ball keeps this share of its forward speed
                            # (rolling ball; measured 0.20-0.31 in the game, physics says 2/7)
E_CUSHION = 0.64            # cushions keep 64% of the speed into them (measured: tan(out) = 0.64 tan(in))
THIN_CUT = 48.0             # cuts thinner than this (deg) are shown as POT?, not POT
MIDDLE_MIN_ANGLE = 48.0     # middle pocket: steepest-rejected approach angle to the rail (deg)
MIDDLE_SURE_ANGLE = 60.0    # below this a middle-pocket pot is shown as POT?
POT_SLACK = 0.35            # path must pass within pocket radius + this many r of the hole centre
POCKET_REACH = (2.7, 1.1)   # a wall hit this close (pocket radius + this many r) drops in: corner, middle
                            # (measured from real shots in the game)
MAX_BOUNCES = 8             # keep following the cue ball until it meets a ball / pocket
OBJ_BOUNCES = 4

# cushion calibration learnt from the game's ghost ball, as a fraction of felt
# height measured inward from each felt edge to the ball-centre limit
CAL = {}
STATE = {'r': None}


# ---------------------------------------------------------------- vision
def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def hue_dist(h, h0):
    d = np.abs(h.astype(np.int16) - h0)
    return np.minimum(d, 180 - d)


class Table:
    def __init__(self, bbox, hue, r):
        self.bx0, self.by0, self.bx1, self.by1 = bbox
        self.hue, self.r = hue, r

    @property
    def bh(self):
        return self.by1 - self.by0

    def edge(self, side):
        return {'l': self.bx0, 't': self.by0, 'r': self.bx1, 'b': self.by1}[side]

    def limit(self, side):
        """Coordinate the ball centre can reach on that side."""
        sign = 1 if side in 'lt' else -1
        if side in CAL:
            return self.edge(side) + sign * CAL[side] * self.bh
        return self.edge(side) + sign * (BAND_RATIO * self.bh + self.r)

    @property
    def inner(self):
        return self.limit('l'), self.limit('t'), self.limit('r'), self.limit('b')

    @property
    def nose(self):     # cushion faces
        ix0, iy0, ix1, iy1 = self.inner
        r = self.r
        return ix0 - r, iy0 - r, ix1 + r, iy1 + r

    def pocket_at(self, p, d=None):
        """Pocket position if a ball reaching the cushion line at p (moving along d)
        drops, else None. Near a pocket the cushion gives way to the pocket mouth: the
        ball carries straight on, and drops if that line passes through the hole
        (measured on real shots: steep approaches drop, shallow ones rattle out)."""
        ix0, iy0, ix1, iy1 = self.inner
        r = self.r
        pk = getattr(self, 'pockets', None)
        if pk is not None and len(pk) >= 4:          # real holes seen on screen
            for k, (cx, cy, pr) in enumerate(pk):
                reach = POCKET_REACH[0] if k < 4 else POCKET_REACH[1]     # pockets: 4 corners, then middles
                c = np.array([cx, cy]) - np.asarray(p, float)
                if np.linalg.norm(c) > pr + reach * r:
                    continue                        # not in this pocket's mouth
                if d is None:
                    return (cx, cy)
                ahead = c @ d
                miss = abs(cross2(d, c))
                if k >= 4:
                    # middle pockets: shallow approaches hit the jaw and bounce out
                    # (measured: ~44 deg to the rail rattled out on a dead-centre line)
                    steep = math.degrees(math.asin(min(1.0, abs(d[1]))))
                    if steep < MIDDLE_MIN_ANGLE:
                        continue
                    self.marginal = steep < MIDDLE_SURE_ANGLE
                if ahead > -0.3 * r and miss < pr + POT_SLACK * r:
                    return (cx, cy)
            return None
        for cx in (ix0, ix1):
            for cy in (iy0, iy1):
                if math.hypot(p[0] - cx, p[1] - cy) < CORNER_POCKET * r:
                    return (cx + (-1 if cx == ix0 else 1) * 1.2 * r,
                            cy + (-1 if cy == iy0 else 1) * 1.2 * r)
        mx = (ix0 + ix1) / 2
        for cy in (iy0, iy1):
            if abs(p[1] - cy) < 0.6 * r and abs(p[0] - mx) < MIDDLE_POCKET * r:
                return (mx, cy + (-1 if cy == iy0 else 1) * 1.6 * r)
        return None

    def side_near(self, p, tol):
        ix0, iy0, ix1, iy1 = self.inner
        for side, dist in (('l', p[0] - ix0), ('r', ix1 - p[0]),
                           ('t', p[1] - iy0), ('b', iy1 - p[1])):
            if abs(dist) < tol:
                return side
        return None


def find_pockets(gray, t):
    """The six pockets: dark round holes at the corners / middle of the long rails.
    Ones hidden (e.g. under a window) are filled in by the table's symmetry."""
    r = t.r
    nx0, ny0, nx1, ny1 = t.nose
    mx, my = (nx0 + nx1) / 2, (ny0 + ny1) / 2
    slots = {'tl': (nx0, ny0), 'tr': (nx1, ny0), 'bl': (nx0, ny1), 'br': (nx1, ny1),
             'tm': (mx, ny0), 'bm': (mx, ny1)}
    dark = (gray < 55).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    found = {}
    for c in cnts:
        area = cv2.contourArea(c)
        if not np.pi * (1.1 * r) ** 2 < area < np.pi * (4 * r) ** 2:
            continue
        (cx, cy), rad = cv2.minEnclosingCircle(c)
        if area < 0.6 * np.pi * rad * rad:            # must be round (or a round-ish cut)
            continue
        for k, (sx, sy) in slots.items():
            dist = math.hypot(cx - sx, cy - sy)
            if dist < 3.5 * r and (k not in found or dist < found[k][3]):
                found[k] = (float(cx), float(cy), float(rad), dist)
    # fill hidden ones by mirroring across the table centre lines
    mirror = {'tl': ['tr', 'bl', 'br'], 'tr': ['tl', 'br', 'bl'], 'bl': ['br', 'tl', 'tr'],
              'br': ['bl', 'tr', 'tl'], 'tm': ['bm'], 'bm': ['tm']}
    out = []
    for k in slots:
        if k in found:
            out.append(found[k][:3])
            continue
        for src in mirror[k]:
            if src in found:
                x, y, rad, _ = found[src]
                if src[1] != k[1] and k[1] != 'm':
                    x = 2 * mx - x
                if src[0] != k[0]:
                    y = 2 * my - y
                out.append((x, y, rad))
                break
    corners = [p for p in out[:4]]
    if 'tm' not in found and 'bm' not in found and len(out) >= 4:
        # middle pockets not visible: they sit level with the corner pockets
        ty = (out[0][1] + out[1][1]) / 2
        by = (out[2][1] + out[3][1]) / 2
        rad = 0.9 * np.mean([p[2] for p in corners])
        out += [(mx, ty, rad), (mx, by, rad)]
    STATE['pockets_seen'] = len(found)
    return out if len(found) >= 2 else []


def find_power_bar(img, bbox):
    """The power bar: a tall narrow bar (cue stick + coloured fill) beside the table.
    Returns its screen rect (x0, y0, x1, y1) or None."""
    bx0, by0, bx1, by1 = bbox
    H, W = img.shape[:2]
    best = None
    for x0, x1 in ((max(int(bx0) - 300, 0), max(int(bx0) - 5, 0)), (min(int(bx1) + 5, W), min(int(bx1) + 300, W))):
        if x1 - x0 < 10:
            continue
        sub = cv2.cvtColor(img[int(by0):int(by1), x0:x1], cv2.COLOR_BGR2HSV)
        m = ((sub[..., 0] >= 5) & (sub[..., 0] <= 35) & (sub[..., 1] > 22) & (sub[..., 2] > 150)).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 3), np.uint8))
        n, _, st, _ = cv2.connectedComponentsWithStats(m)
        for i in range(1, n):
            x, y, w, h, area = st[i]
            if h > 0.5 * (by1 - by0) and w < 0.15 * h and (best is None or h > best[0]):
                best = (h, (x0 + x, int(by0) + y, x0 + x + w, int(by0) + y + h))
    return None if best is None else best[1]


def read_power(img, rect):
    """Fraction of the power bar filled (0 when not pulling)."""
    x0, y0, x1, y1 = rect
    xc = (x0 + x1) // 2
    pad = max(30, (x1 - x0) * 4)
    sub = img[y0:y1, max(xc - pad, 0):xc + pad]
    if sub.size == 0:
        return None
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    vivid = (hsv[..., 0] >= 0) & (hsv[..., 0] <= 35) & (hsv[..., 1] > 150) & (hsv[..., 2] > 190)
    rows = vivid.sum(axis=1) >= 12            # the fill is a wide vivid band; the stick is pale and thin
    if rows.sum() < 3:
        return 0.0
    idx = np.flatnonzero(rows)
    return float(np.clip((idx[-1] - idx[0] + 1) / (y1 - y0), 0, 1))


def _table_rects(m):
    """Rectangle-shaped components of a felt mask: (area, x, y, w, h). If felt from
    something else touches the table (another window, a picture of a table...), the
    blob isn't rectangular; eroding harder separates them, then the box is grown back."""
    for k in (0, 5, 9, 15, 25):
        mk = m if k == 0 else cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        n, _, stats, _ = cv2.connectedComponentsWithStats(mk)
        out = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if w >= 60 and h >= 30 and 1.5 < w / h < 2.6 and area >= 0.8 * w * h:
                g = k // 2
                out.append((area, x - g, y - g, w + 2 * g, h + 2 * g))
        if out:
            return out
    # nothing cleanly rectangular (e.g. a window covers part of the table): fall back
    # to the largest roughly table-shaped blob
    n, _, stats, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return []
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = stats[i]
    if w >= 60 and h >= 30 and 1.5 < w / h < 2.6 and area >= 0.6 * w * h:
        return [(area, x, y, w, h)]
    return []


def find_table(hsv):
    """Largest table-shaped (rectangular) region of one saturated colour = the felt."""
    small = hsv[::4, ::4]
    sat = (small[..., 1] > 70) & (small[..., 2] > 45)
    if sat.sum() < 1000:
        return None
    hist = np.bincount(small[..., 0][sat], minlength=180).astype(float)
    hist = np.convolve(np.r_[hist[-6:], hist, hist[:6]], np.ones(7), 'same')[6:-6]
    best = None
    for _ in range(3):
        h0 = int(np.argmax(hist))
        if hist[h0] <= 0:
            break
        hist[hue_dist(np.arange(180), h0) <= 12] = 0
        m = ((hue_dist(small[..., 0], h0) <= 12) & sat).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        for area, x, y, w, h in _table_rects(m):
            if w >= 75 and h >= 40 and (best is None or area > best[0]):
                best = (area, h0, (x, y, w, h))
    if best is None:
        return None
    _, h0, (x, y, w, h) = best
    # refine at full resolution, but only inside that rectangle
    H, W = hsv.shape[:2]
    X0, Y0 = max(4 * x - 8, 0), max(4 * y - 8, 0)
    X1, Y1 = min(4 * (x + w) + 8, W), min(4 * (y + h) + 8, H)
    sub = hsv[Y0:Y1, X0:X1]
    m = ((hue_dist(sub[..., 0], h0) <= 12) & (sub[..., 1] > 70) & (sub[..., 2] > 45)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, _ = stats[i]
    x, y = x + X0, y + Y0
    if STATE.get('r') is None and STATE.get('r_ratio'):
        STATE['r'] = STATE['r_ratio'] * w           # remembered from an earlier session
    r = STATE['r'] or BALL_RATIO * w
    return Table((float(x), float(y), float(x + w), float(y + h)), h0, r)


def refine_centre(gray, cx, cy, r):
    """Sub-pixel ball centre + radius: edge along 48 rays, then a circle fit."""
    H, W = gray.shape
    rad = np.arange(0.6 * r, 1.45 * r, 0.25)
    ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    ca, sa = np.cos(ang)[:, None], np.sin(ang)[:, None]
    xs, ys = cx + ca * rad, cy + sa * rad
    if xs.min() < 1 or ys.min() < 1 or xs.max() >= W - 2 or ys.max() >= H - 2:
        return cx, cy, None
    vals = cv2.remap(gray, xs.astype(np.float32), ys.astype(np.float32), cv2.INTER_LINEAR)
    k = np.abs(np.diff(vals, axis=1)).argmax(axis=1)
    er = (rad[k] + rad[k + 1]) / 2
    ex, ey = cx + ca[:, 0] * er, cy + sa[:, 0] * er
    ok = np.abs(er - r) < 0.3 * r
    if ok.sum() < 16:
        return cx, cy, None
    ex, ey = ex[ok], ey[ok]
    for _ in range(2):   # algebraic circle fit, then drop the worst outliers
        A = np.c_[ex, ey, np.ones_like(ex)]
        sol = np.linalg.lstsq(A, ex ** 2 + ey ** 2, rcond=None)[0]
        ox, oy = sol[0] / 2, sol[1] / 2
        rr = np.hypot(ex - ox, ey - oy)
        keep = np.abs(rr - np.median(rr)) < 1.0
        if keep.sum() < 12:
            break
        ex, ey = ex[keep], ey[keep]
    if math.hypot(ox - cx, oy - cy) > 0.35 * r:
        return cx, cy, None
    return float(ox), float(oy), float(np.median(np.hypot(ex - ox, ey - oy)))


def felt_maps(img, hsv, t):
    """Per-pixel colour distance from the felt. Everything else (lines, stick,
    ghost ring, balls) is found as 'not felt', so the game's colours don't matter."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    y0, y1, x0, x1 = int(t.by0), int(t.by1), int(t.bx0), int(t.bx1)
    key = (t.off, lab.shape, t.hue)
    cached = STATE.get('bg')
    if cached is not None and cached[0] == key and STATE['n'] % 30:
        t.felt_ref, bg = cached[1], cached[2]          # the felt doesn't change while aiming
        diff = cv2.absdiff(lab, bg)
        t.con = cv2.transform(diff.astype(np.float32), np.ones((1, 3), np.float32))
        return thin_thick(t)
    feltlike = (hue_dist(hsv[..., 0], t.hue) <= 10) & (hsv[..., 1] > 60) & (hsv[..., 2] > 40)
    t.felt_ref = np.median(lab[y0:y1:3, x0:x1:3][feltlike[y0:y1:3, x0:x1:3]], axis=0)
    # local felt colour (handles spotlights / gradients / textures): a smooth
    # average of nearby felt-like pixels only, so balls and lines don't leak in
    f = 0.25
    sl = cv2.resize(lab.astype(np.float32), None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    sm = cv2.resize(feltlike.astype(np.float32), None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    sig = max(2.0, t.r * f * 3)
    num = cv2.GaussianBlur(sl * sm[..., None], (0, 0), sig)
    den = cv2.GaussianBlur(sm, (0, 0), sig)[..., None]
    bg = np.where(den > 0.05, num / np.maximum(den, 1e-3), t.felt_ref)
    bg = cv2.resize(bg.astype(np.float32), (lab.shape[1], lab.shape[0]), interpolation=cv2.INTER_LINEAR)
    bg = np.clip(bg, 0, 255).astype(np.uint8)
    STATE['bg'] = (key, t.felt_ref, bg)
    diff = cv2.absdiff(lab, bg)
    t.con = cv2.transform(diff.astype(np.float32), np.ones((1, 3), np.float32))
    thin_thick(t)


def thin_thick(t):
    """Split 'not felt' into thin marks (lines) and thick stuff (stick, balls)."""
    o = STATE.get('own_crop')
    if o is not None and o.shape == t.con.shape and o.any():
        t.con[o] = 0                       # our own drawings are not part of the game
    nonfelt = (t.con > 30).astype(np.uint8)
    faint = t.con > 18                     # thin lines are antialiased: accept fainter
    k = max(5, int(round(0.3 * t.r)) | 1)
    opened = cv2.morphologyEx(nonfelt, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    t.thin = faint & ~cv2.dilate(opened, np.ones((3, 3), np.uint8)).astype(bool)
    ke = max(3, int(round(0.33 * t.r)) | 1)          # scale with ball size (window size)
    t.thick = cv2.erode(nonfelt, np.ones((ke, ke), np.uint8)).astype(bool)


def find_balls(img, hsv, gray, t):
    r = t.r
    pad = int(2 * r)
    X0, Y0 = max(int(t.bx0) - pad, 0), max(int(t.by0) - pad, 0)
    crop = img[Y0:int(t.by1) + pad, X0:int(t.bx1) + pad]
    if crop.size == 0:
        return []
    g = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (5, 5), 1.2)
    lo, hi = (0.8, 1.25) if STATE['r'] else (0.7, 1.4)
    circles = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1, minDist=1.6 * r * lo,
                               param1=55, param2=11,
                               minRadius=int(r * lo), maxRadius=int(r * hi) + 1)
    if circles is None:
        return []
    ix0, iy0, ix1, iy1 = t.inner
    H, W = gray.shape
    # message banners on the table ("You break", "Computer goes again"...): wide, dark,
    # solid boxes. Round letters in them look like balls - ignore everything inside.
    dark = (hsv[..., 2] < 75).astype(np.uint8)
    n_, _, st_, _ = cv2.connectedComponentsWithStats(dark)
    banners = []
    for i in range(1, n_):
        x, y, w, h, area = st_[i]
        if w > 5 * r and 1.2 * r < h < 6 * r and area > 0.4 * w * h:
            banners.append((x - 0.5 * r, y - 0.5 * r, x + w + 0.5 * r, y + h + 0.5 * r))
    cand = []
    rings = []
    for cx, cy, crad in circles[0]:
        gx, gy = cx + X0, cy + Y0
        if any(bx0 <= gx <= bx1 and by0 <= gy <= by1 for bx0, by0, bx1, by1 in banners):
            continue
        if not (ix0 - 0.5 * r <= gx <= ix1 + 0.5 * r and iy0 - 0.5 * r <= gy <= iy1 + 0.5 * r):
            continue
        a, b = int(round(gy)), int(round(gx))
        R = int(1.6 * r) + 1
        if a - R < 0 or b - R < 0 or a + R >= H or b + R >= W:
            continue
        yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
        rr = np.hypot(yy, xx)
        diff = t.con[a - R:a + R + 1, b - R:b + R + 1]
        if (diff[rr <= 0.75 * r] > 30).mean() < 0.6:     # hollow ring / felt
            ring_px = diff[(rr > 0.8 * r) & (rr < 1.2 * r)]
            if (ring_px > 30).mean() > 0.6:               # a drawn ring: maybe the ghost ball
                rings.append((gx, gy))
            continue
        ann = diff[(rr > 1.15 * r) & (rr < 1.6 * r)]
        feltfrac = (ann < 30).mean()
        hp = hsv[a - R:a + R + 1, b - R:b + R + 1][rr <= 0.75 * r]
        if (hp[:, 2] < 45).mean() > 0.85:                 # pocket hole
            continue
        white = ((hp[:, 1] < 60) & (hp[:, 2] > 185)).mean() \
            - 3 * ((hp[:, 1] > 90) & (hp[:, 2] > 60)).mean() - 2 * (hp[:, 2] < 70).mean()
        cand.append([gx, gy, white, feltfrac, crad])
    # A real ball has felt around it. Lone balls need plenty; balls in a group
    # (the rack) need some, or two neighbours that do. Text on banners or windows
    # on top of the table fails both.
    def near(c, pool):
        return sum(0 < math.hypot(c[0] - q[0], c[1] - q[1]) < 2.4 * r for q in pool)
    good = [c for c in cand if c[3] >= 0.15]
    keep = [c for c in cand if
            (c[3] >= 0.4 if near(c, cand) == 0 else (c[3] >= 0.15 or near(c, good) >= 2))]
    STATE['rings'] = rings
    if len(keep) >= 3 and STATE['r'] is None:
        t.r = float(np.median([c[4] for c in keep]))
    balls = []
    for c in keep:
        x, y, _ = refine_centre(gray, c[0], c[1], r)
        balls.append([x, y, c[2]])
    return balls


def line_mask(t):
    """Thin marks on the felt, any colour: the aim line and the game's hint lines."""
    return t.thin


def fit_line_subpix(v, m, start, d, s0, s1):
    """Contrast-weighted centre of a thin line between distances s0..s1 (vectorised)."""
    H, W = v.shape
    nrm = np.array([-d[1], d[0]])
    off = np.arange(-3, 4)
    ss = np.arange(s0, s1, 2.0)
    if len(ss) < 6:
        return None
    q = start[None] + d[None] * ss[:, None]                                   # N,2
    xs = (q[:, 0:1] + nrm[0] * off[None]).round().astype(int)               # N,7
    ys = (q[:, 1:2] + nrm[1] * off[None]).round().astype(int)
    inside = (xs.min(1) >= 0) & (ys.min(1) >= 0) & (xs.max(1) < W) & (ys.max(1) < H)
    if not inside.all():
        n = int(np.argmin(inside))
        q, xs, ys = q[:n], xs[:n], ys[:n]
    if len(q) < 6:
        return None
    vals = v[ys, xs]
    w = vals - vals.min(axis=1, keepdims=True)
    ok = m[ys[:, 2:5], xs[:, 2:5]].any(axis=1) & (w.sum(axis=1) >= 60)
    if ok.sum() < 6:
        return None
    cen = (w[ok] * off).sum(axis=1) / w[ok].sum(axis=1)
    cs = (q[ok] + nrm[None] * cen[:, None]).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(cs, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    fd = np.array([vx, vy], float)
    return (fd if fd @ d > 0 else -fd), np.array([x0, y0], float)


def walk(m, start, d, s, r, refit=None):
    """Follow a line in mask m from distance s; return (last distance with a hit,
    hit points relative to start). Vectorised; with refit it runs twice so a
    slightly-off starting direction can't make it drift off a long line."""
    H, W = m.shape
    for _ in range(2 if refit else 1):
        ss = np.arange(s, 4000.0)
        nrm = np.array([-d[1], d[0]])
        offs = np.arange(-2, 3)
        xs = (start[0] + d[0] * ss[:, None] + nrm[0] * offs[None]).round().astype(int)
        ys = (start[1] + d[1] * ss[:, None] + nrm[1] * offs[None]).round().astype(int)
        inside = ((xs >= 0) & (ys >= 0) & (xs < W) & (ys < H)).all(axis=1)
        n = int(np.argmin(inside)) if not inside.all() else len(ss)
        hits = m[ys[:n].clip(0, H - 1), xs[:n].clip(0, W - 1)]
        row = hits.any(axis=1)
        idx = np.flatnonzero(row)
        if len(idx) == 0:
            return s, []
        gaps = np.diff(np.r_[idx, n + 10 * r])
        stop = int(np.argmax(gaps > 1.5 * r))
        idx = idx[:stop + 1]
        last = ss[idx[-1]]
        sel = np.zeros(n, bool)
        sel[idx[0]:idx[-1] + 1] = True
        k = hits & sel[:, None]
        pts = np.c_[xs[:n][k] - start[0], ys[:n][k] - start[1]]
        if refit and len(pts) > 10:
            d = refit(pts, d)
    return last, pts


def fit_through_origin(pts, d0):
    p = np.array(pts, float)
    sxx, syy, sxy = (p[:, 0] ** 2).sum(), (p[:, 1] ** 2).sum(), (p[:, 0] * p[:, 1]).sum()
    th = 0.5 * math.atan2(2 * sxy, sxx - syy)
    fd = np.array([math.cos(th), math.sin(th)])
    return fd if fd @ d0 > 0 else -fd


def find_aim(hsv, m, t, balls):
    """Cue ball + aim. The stick says which ball is the cue ball; the game's own
    thin aim line, when it's long, gives the most exact direction."""
    res = _find_aim(hsv, m, t, balls)
    if res is None:
        return None
    cue_i, d, origin, _ = res
    return cue_i, d, origin, line_extent(t, origin, d)


def find_cue_by_ring(t, m, balls):
    """The game's ghost ring sits at the end of the aim line; the cue ball is the ball
    joined to it by a straight line (not the ball it touches). Works for any style of
    stick or line. Returns (cue index, rough direction) or None."""
    r = t.r
    H, W = m.shape
    line = m | (t.con > 30)
    best = None
    for gx, gy in STATE.get('rings', []):
        g = np.array([gx, gy])
        for i, b in enumerate(balls):
            c = np.array(b[:2], float)
            L = np.linalg.norm(g - c)
            if L < 2.6 * r:
                continue                          # touching the ring: that's the object ball
            u = (g - c) / L
            ss = np.arange(1.3 * r, L - 1.1 * r, 1.0)
            if len(ss) < 4:
                continue
            xs = (c[0] + u[0] * ss).round().astype(int).clip(0, W - 1)
            ys = (c[1] + u[1] * ss).round().astype(int).clip(0, H - 1)
            frac = line[ys, xs].mean()
            if frac < 0.8:
                continue
            # nothing may sit between the cue ball and the ghost (that path is clear by
            # definition - the ghost marks the first thing the cue ball reaches)
            blocked = False
            for j, q in enumerate(balls):
                if j == i:
                    continue
                w = np.array(q[:2]) - c
                along = w @ u
                if 0.5 * r < along < L - 1.2 * r and abs(cross2(u, w)) < 1.6 * r:
                    blocked = True
                    break
            if blocked:
                continue
            # the line must not run THROUGH the ball (then it's the object ball's hint line)
            bs = np.arange(1.3 * r, 2.6 * r, 1.0)
            bx = (c[0] - u[0] * bs).round().astype(int).clip(0, W - 1)
            by = (c[1] - u[1] * bs).round().astype(int).clip(0, H - 1)
            if m[by, bx].mean() > 0.6 and t.thick[by, bx].mean() < 0.3:
                continue
            score = frac * L * (0.3 + max(b[2], 0))
            if best is None or score > best[0]:
                best = (score, i, u, (gx, gy))
    if best is None:
        STATE['ring_hit'] = None
        return None
    STATE['ring_hit'] = best[3]
    return best[1], best[2]


def _find_aim(hsv, m, t, balls):
    # which ball is the cue ball: from the ghost ring if we can see it, else the stick
    ring = find_cue_by_ring(t, m, balls)
    if ring is not None:
        cue_i, u = ring
        c = np.array(balls[cue_i][:2], float)
        # exact direction: the long aim line if there is one, else the stick (if it agrees)
        near = line_near(t, m, c, u, spread=4.0)
        if near is not None:
            fd, o2, last = follow_line(t, m, near[1], near[0])
            if last > 6 * t.r and np.linalg.norm(o2 - c) < 0.8 * t.r and fd @ u > math.cos(math.radians(4)):
                STATE['aim_src'] = 'line'
                return cue_i, fd, o2, last
        # cue centre -> ring centre is exact (the stick can be drawn off-centre)
        STATE['aim_src'] = 'ring'
        rh = STATE.get('ring_hit')
        last = float(np.linalg.norm(np.array(rh) - c)) if rh else walk(m, c, u, 1.2 * t.r, t.r)[0]
        return cue_i, u, c, last
    # direction source, best first: long aim line > cue stick > (later) ghost ring
    st = find_aim_stick(hsv, m, t, balls)
    if st is None:
        STATE['aim_src'] = 'line'
        return find_aim_line(hsv, m, t, balls)
    cue_i, d, origin, _ = st
    near = line_near(t, m, origin, d)
    if near is not None:
        fd, o2, last = follow_line(t, m, near[1], near[0])
        if last > 6 * t.r and np.linalg.norm(o2 - origin) < 0.8 * t.r:
            STATE['aim_src'] = 'line'
            return cue_i, fd, o2, last
    STATE['aim_src'] = 'stick'
    return st


def find_aim_stick(hsv, m, t, balls):
    """The cue ball is the ball with the (thick) cue stick lined up behind it.
    The stick is long, so its axis gives a very precise aim even when the
    game's white aim line is only a few pixels long."""
    r = t.r
    H, W = m.shape
    # a row of touching balls looks like a thick bar too: blank out every ball first
    nb = np.zeros(t.thick.shape, np.uint8)
    for b in balls:
        cv2.circle(nb, (int(round(b[0])), int(round(b[1]))), int(1.2 * r), 1, -1)
    thick = t.thick & (nb == 0)
    prev = STATE.get('stick_prev')
    ang = np.radians(np.arange(0, 360, 1.0))
    dist = np.arange(1.8 * r, 14 * r, 2.0)
    ca, sa = np.cos(ang)[:, None], np.sin(ang)[:, None]

    nx0, ny0, nx1, ny1 = t.nose

    def sample(x, y):
        xs, ys = x.round().astype(int), y.round().astype(int)
        return thick[ys.clip(0, H - 1), xs.clip(0, W - 1)]

    # only balls with something thick touching them can have the stick behind
    ring = np.arange(1.8 * r, 10 * r, 3.0)
    rx_ = lambda b: (b[0] + ca * ring).round().astype(int).clip(0, W - 1)
    ry_ = lambda b: (b[1] + sa * ring).round().astype(int).clip(0, H - 1)
    cands = [i for i, b in enumerate(balls) if thick[ry_(b), rx_(b)].mean(axis=1).max() > 0.12]
    second = 0.0
    last_cue = STATE.get('cue_xy')
    if last_cue is not None:
        cands.sort(key=lambda i: math.hypot(balls[i][0] - last_cue[0], balls[i][1] - last_cue[1]))
    best = None
    for i in cands:
        b = balls[i]
        bx, by = b[0] + ca * dist, b[1] + sa * dist
        lx, ly, rx, ry = bx - sa * r, by + ca * r, bx + sa * r, by - ca * r
        on = ((np.minimum(lx, rx) > nx0) & (np.maximum(lx, rx) < nx1) &
              (np.minimum(ly, ry) > ny0) & (np.maximum(ly, ry) < ny1))   # over the felt only
        n = on.sum(axis=1)
        n = np.where(n >= r, n, np.inf)                # need at least ~2r of it
        mid = (sample(bx, by) & on).sum(axis=1) / n
        side = ((sample(lx, ly) & on).sum(axis=1) + (sample(rx, ry) & on).sum(axis=1)) / (2 * n)
        # a bar: solid in the middle, felt beside it - and only on ONE side of the ball
        # (the cue ball sits at the stick's tip; a ball the stick lies across has it both sides)
        opposite = np.roll(mid, len(mid) // 2)
        sc = mid - 1.5 * side - 1.2 * opposite
        k = int(np.argmax(sc))
        # how wide is the bar right behind the ball? The stick is fat (about half a ball
        # radius or more); the game's aim / hint lines are thin, even when drawn boldly
        u0 = np.array([math.cos(ang[k]), math.sin(ang[k])])
        pn = np.array([-u0[1], u0[0]])
        ws = []
        for dd in np.arange(2.0 * r, 5.0 * r, 1.0 * r):
            q = np.array(b[:2]) + u0 * dd
            if not (nx0 < q[0] < nx1 and ny0 < q[1] < ny1):
                continue
            offs = np.arange(-1.0 * r, 1.0 * r + 0.01, 0.5)
            px = (q[0] + pn[0] * offs).round().astype(int).clip(0, W - 1)
            py = (q[1] + pn[1] * offs).round().astype(int).clip(0, H - 1)
            prof = t.con[py, px] > 30
            if prof[len(offs) // 2]:
                lo = hi = len(offs) // 2          # contiguous run around the centre
                while lo > 0 and prof[lo - 1]:
                    lo -= 1
                while hi < len(offs) - 1 and prof[hi + 1]:
                    hi += 1
                ws.append((hi - lo + 1) * 0.5)
        if ws and np.median(ws) < 0.38 * r:
            sc = sc * 0.1                         # a line, not the cue stick
        # if the bar runs into another ball and stops there, that ball is at the stick's
        # tip (the cue ball), not this one. If the bar carries on past it, the stick is
        # just lying over that ball.
        u = np.array([math.cos(ang[k]), math.sin(ang[k])])
        for j, q in enumerate(balls):
            if j == i:
                continue
            w = np.array(q[:2]) - np.array(b[:2])
            along = w @ u
            if 1.8 * r < along < 60 * r and abs(cross2(u, w)) < 0.8 * r:
                ss = along + np.arange(1.8 * r, 6 * r, 2.0)
                px = (b[0] + u[0] * ss).round().astype(int)
                py = (b[1] + u[1] * ss).round().astype(int)
                inside = (px >= 0) & (py >= 0) & (px < W) & (py < H)
                carries_on = inside.any() and thick[py[inside], px[inside]].mean() > 0.4
                if not carries_on:
                    sc = sc * 0.2
                break
        if best is None or sc[k] > best[0]:
            second = best[0] if best is not None else 0.0
            best = (sc[k], i, ang[k])
        else:
            second = max(second, sc[k])
        if last_cue is not None and i == cands[0] and sc[k] > 0.7 and \
                math.hypot(b[0] - last_cue[0], b[1] - last_cue[1]) < r:
            break                       # same cue ball as last frame, clearly: done
    if best is None:
        return None
    # a strong bar, or a shorter one (stick mostly over the cushion) that clearly wins
    if best[0] < 0.45 and not (best[0] >= 0.12 and best[0] > 3 * max(second, 0.0)) and \
            not (best[0] >= 0.3 and best[0] > 1.25 * max(second, 0.0)):
        return None
    _, cue_i, a = best
    c = np.array(balls[cue_i][:2], float)
    u = np.array([math.cos(a), math.sin(a)])
    nrm = np.array([-u[1], u[0]])
    ss = np.arange(1.8 * r, 60 * r, 1.0)
    ls = np.arange(-8, 9)
    xs = (c[0] + u[0] * ss[:, None] + nrm[0] * ls[None]).round().astype(int)
    ys = (c[1] + u[1] * ss[:, None] + nrm[1] * ls[None]).round().astype(int)
    ok = (xs > nx0) & (ys > ny0) & (xs < nx1) & (ys < ny1)      # over the felt only
    hit = thick[ys.clip(0, H - 1), xs.clip(0, W - 1)] & ok
    if hit.sum() < 30:
        return None
    pts = np.c_[xs[hit], ys[hit]].astype(np.float32)
    if len(pts) > 600:
        pts = pts[::len(pts) // 600 + 1]                # plenty for a line fit, much faster
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    ud = np.array([vx, vy], float)
    if ud @ u < 0:
        ud = -ud
    p0 = np.array([x0, y0], float)
    origin = p0 + ud * ((c - p0) @ ud)
    if np.linalg.norm(origin - c) > 0.7 * r:
        return None
    d = -ud
    last, _ = walk(m, origin, d, 1.2 * r, r)
    return cue_i, d, origin, last


def find_aim_line(hsv, m, t, balls):
    """Fallback: find the game's aim line and the ball it starts from.
    Returns (cue_index, direction, origin, line_end_distance) or None."""
    r = t.r
    X0, Y0 = max(int(t.bx0), 0), max(int(t.by0), 0)
    sub = (m[Y0:int(t.by1), X0:int(t.bx1)] * 255).astype(np.uint8)
    segs = cv2.HoughLinesP(sub, 1, np.pi / 720, threshold=int(2 * r),
                           minLineLength=int(2.5 * r), maxLineGap=int(r))
    if segs is None:
        return None
    stick = t.thick | m
    H, W = m.shape

    def width(c, d, sign):
        """Median width (px) of bright stuff 3r..9r from c along sign*d."""
        ss = np.arange(3 * r, 9 * r, 2.0)
        nrm = np.array([-d[1], d[0]])
        offs = np.arange(-7, 8)
        xs = (c[0] + sign * d[0] * ss[:, None] + nrm[0] * offs[None]).round().astype(int)
        ys = (c[1] + sign * d[1] * ss[:, None] + nrm[1] * offs[None]).round().astype(int)
        ok = ((xs >= 0) & (ys >= 0) & (xs < W) & (ys < H)).all(axis=1)
        if not ok.any():
            return 0
        return float(np.median(stick[ys[ok], xs[ok]].sum(axis=1)))

    best = None
    for x1, y1, x2, y2 in segs:
        p1, p2 = np.array([x1 + X0, y1 + Y0], float), np.array([x2 + X0, y2 + Y0], float)
        L = np.linalg.norm(p2 - p1)
        u = (p2 - p1) / L
        for i, bl in enumerate(balls):
            c = np.array(bl[:2], float)
            proj = (c - p1) @ u
            if not -3 * r < proj < L + 3 * r or abs(cross2(u, c - p1)) > 0.7 * r:
                continue
            d, far = (u, L - proj) if proj < L / 2 else (-u, proj)
            if far < 2.5 * r:
                continue
            ahead, behind = width(c, d, 1), width(c, d, -1)
            if ahead > 4:            # that's the cue stick, not the thin aim line
                continue
            # the game's object-ball hint runs THROUGH its ball (from the ghost ring);
            # the aim line only starts at the cue ball. Thin line behind the ball = hint.
            bs = np.arange(1.2 * r, 2.6 * r, 1.0)
            bx = (c[0] - d[0] * bs).round().astype(int).clip(0, W - 1)
            by = (c[1] - d[1] * bs).round().astype(int).clip(0, H - 1)
            if m[by, bx].mean() > 0.4 and t.thick[by, bx].mean() < 0.3:
                continue                         # (a fat cue stick behind it is fine)
            score = far * (1 + (behind > 4)) * (0.3 + max(bl[2], 0))
            if best is None or score > best[0]:
                best = (score, i, d, p1)
    if best is None:
        return None
    _, cue_i, d, p1 = best
    c = np.array(balls[cue_i][:2], float)
    c = p1 + d * ((c - p1) @ d)             # put the centre on the segment first
    return (cue_i,) + follow_line(t, m, c, d)


def follow_line(t, m, c, d):
    """Follow the aim line from cue centre c roughly along d, fit it sub-pixel.
    Returns (direction, origin on the line, length)."""
    r = t.r
    last, pts = walk(m, c, d, 1.9 * r, r, refit=fit_through_origin)
    if len(pts) >= 10:
        d = fit_through_origin(pts, d)
    fit = fit_line_subpix(t.con, m, c, d, 2.5 * r, last - 1.5 * r)
    if fit is None:
        return d, c, last
    fd, p0 = fit
    return fd, p0 + fd * ((c - p0) @ fd), last      # snap the cue centre onto the line


def line_extent(t, origin, d):
    """How far the game's aim line visibly runs from `origin` along the (already
    exact) direction d. Uses a ridge test (brighter or darker than both sides),
    which survives faint stretches on bright felt better than a mask does."""
    r, g = t.r, t.gray
    H, W = g.shape
    ss = np.arange(1.9 * r, 4000.0, 1.0)
    x, y = origin[0] + d[0] * ss, origin[1] + d[1] * ss
    ok = (x > 3) & (y > 3) & (x < W - 3) & (y < H - 3)
    if not ok.any():
        return 1.9 * r
    n = int(np.argmin(ok)) if not ok.all() else len(ss)
    ss, x, y = ss[:n], x[:n], y[:n]
    px, py = -d[1] * 2.5, d[0] * 2.5
    at = lambda a, b: cv2.remap(g, a.astype(np.float32)[None], b.astype(np.float32)[None],
                                cv2.INTER_LINEAR)[0]
    c, l, rr = at(x, y), at(x + px, y + py), at(x - px, y - py)
    ridge = np.maximum(c - np.maximum(l, rr), np.minimum(l, rr) - c)
    hit = np.flatnonzero(ridge > 6)
    if len(hit) == 0:
        return 1.9 * r
    gaps = np.diff(np.r_[hit, n + 10 * r])
    stop = int(np.argmax(gaps > 1.5 * r))
    return float(ss[hit[stop]])


def line_near(t, m, c, d0, spread=3.0):
    """Best thin line leaving (near) c within +-spread degrees of d0. The stick
    can sit a few px off the true centre (shadows), so also slide sideways.
    Returns (direction, start point) or None."""
    r = t.r
    H, W = m.shape
    base = math.atan2(d0[1], d0[0])
    angs = base + np.radians(np.arange(-spread, spread + 0.01, 0.1))
    lats = np.arange(-0.6 * r, 0.6 * r + 0.01, 1.0)
    dist = np.arange(1.9 * r, 8 * r, 1.0)
    A, Lt = np.meshgrid(angs, lats)
    A, Lt = A.ravel(), Lt.ravel()
    ca, sa = np.cos(A)[:, None], np.sin(A)[:, None]
    nx, ny = -math.sin(base), math.cos(base)
    xs = (c[0] + nx * Lt[:, None] + ca * dist).round().astype(int).clip(0, W - 1)
    ys = (c[1] + ny * Lt[:, None] + sa * dist).round().astype(int).clip(0, H - 1)
    sc = m[ys, xs].mean(axis=1)
    k = int(np.argmax(sc))
    if sc[k] < 0.5:
        return None
    return np.array([math.cos(A[k]), math.sin(A[k])]), c + np.array([nx, ny]) * Lt[k]


def find_ghost(gray, origin, d, last, r):
    """The game's ghost-ball ring at the end of its aim line -> (centre, radius)."""
    H, W = gray.shape
    unit = np.stack([np.cos(np.linspace(0, 2 * np.pi, 72, endpoint=False)),
                     np.sin(np.linspace(0, 2 * np.pi, 72, endpoint=False))], -1)

    nrm = np.array([-d[1], d[0]])

    def scores(ss, rads, ls):
        """Ring-ness of every (distance along line, radius, sideways) triple, in one pass."""
        c = origin[None, :] + d[None, :] * ss[:, None] + nrm[None, :] * ls[:, None]   # N,2
        on = c[:, None, :] + unit[None] * rads[:, None, None]                         # N,72,2
        ins = c[:, None, :] + unit[None] * (0.6 * rads)[:, None, None]
        def sample(p):
            return cv2.remap(gray, p[..., 0].astype(np.float32), p[..., 1].astype(np.float32),
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        ring = sample(on).reshape(len(ss), 8, 9).mean(axis=2)     # 8 sectors of the ring
        sc = ring.min(axis=1) - np.median(sample(ins), axis=1)   # whole ring there; marks inside ok
        bad = (on.min(axis=(1, 2)) < 1) | (on[..., 0].max(axis=1) >= W - 1) | (on[..., 1].max(axis=1) >= H - 1)
        sc[bad] = -1
        return sc, c

    radii = np.arange(r - 1, r + 1.01, 0.5) if STATE.get('r') else np.arange(0.7 * r, 1.4 * r, 0.5)
    def coarse(lo, hi):
        S, R = np.meshgrid(np.arange(lo, hi, 2.0), radii)
        S, R = S.ravel(), R.ravel()
        sc, _ = scores(S, R, np.zeros_like(S))
        return S, R, sc

    def attempt(lo, hi):
        S, R, sc = coarse(lo, hi)
        if len(sc) == 0:
            return None
        k = int(np.argmax(sc))
        base = float(np.percentile(np.maximum(sc, 0), 25))   # how ring-like plain line/felt looks
        s0, r0 = S[k], R[k]
        # fine search in two cheap passes: along the line + size, then sideways
        S, R = np.meshgrid(np.arange(s0 - 2.5, s0 + 2.51, 0.25), np.arange(r0 - 1, r0 + 1.01, 0.25))
        S, R = S.ravel(), R.ravel()
        sc2, _ = scores(S, R, np.zeros_like(S))
        k = int(np.argmax(sc2))
        s1, r1 = S[k], R[k]
        S, L = np.meshgrid(np.arange(s1 - 0.5, s1 + 0.51, 0.25), np.arange(-2, 2.01, 0.25))
        S, L = S.ravel(), L.ravel()
        sc2, cs = scores(S, np.full_like(S, r1), L)
        k = int(np.argmax(sc2))
        if sc2[k] < max(15.0, 2.5 * base + 8):        # the ring must stand out
            return None
        return cs[k], float(r1)

    # the ring is where the visible aim line ends; the whole line is the fallback
    found = attempt(max(1.8 * r, last - 2.5 * r), last + 2.5 * r) or attempt(1.8 * r, last + 8 * r)
    return found if found else (None, None)


def read_hint(t, start, n, spread=8):
    """Read the game's object-ball hint line leaving `start` near direction n.
    It's found as a bright ridge (brighter than both sides), which still works
    where it is drawn across other balls. The line goes through the true ball
    centre, so we also let the start point slide sideways a little.
    Returns (direction, corrected start) or None."""
    r, g = t.r, t.gray
    H, W = g.shape
    base = math.atan2(n[1], n[0])
    angs = base + np.radians(np.arange(-spread, spread + 0.01, 0.2))
    lats = np.arange(-0.3 * r, 0.3 * r + 0.01, 1.0)
    dist = np.arange(1.3 * r, 8 * r, 1.5)
    A, Lt = np.meshgrid(angs, lats)
    A, Lt = A.ravel(), Lt.ravel()
    nx, ny = -math.sin(base), math.cos(base)
    cx = start[0] + nx * Lt[:, None] + np.cos(A)[:, None] * dist
    cy = start[1] + ny * Lt[:, None] + np.sin(A)[:, None] * dist
    if cx.min() < 3 or cy.min() < 3 or cx.max() >= W - 3 or cy.max() >= H - 3:
        return None
    px, py = -np.sin(A)[:, None] * 2.5, np.cos(A)[:, None] * 2.5

    def at(x, y):
        return cv2.remap(g, x.astype(np.float32), y.astype(np.float32), cv2.INTER_LINEAR)

    c = at(cx, cy)
    ridge = c - np.maximum(at(cx + px, cy + py), at(cx - px, cy - py))
    sc = (ridge > 10).mean(axis=1)
    k = int(np.argmax(sc))
    if sc[k] < 0.45 or sc[k] < 3 * np.median(sc):
        return None
    d = np.array([math.cos(A[k]), math.sin(A[k])])
    s0 = start + np.array([nx, ny]) * Lt[k]
    # sub-pixel: brightness centroid across the ridge at each step, then a line fit
    pn = np.array([-d[1], d[0]])
    offs = np.arange(-3, 4, dtype=np.float32)
    good = ridge[k] > 10
    q = s0[None] + d[None] * dist[good][:, None]
    xs = q[:, 0:1] + pn[0] * offs[None]
    ys = q[:, 1:2] + pn[1] * offs[None]
    prof = at(xs, ys)
    w = np.maximum(prof - prof.min(axis=1, keepdims=True) - 0.5 * np.ptp(prof, axis=1, keepdims=True), 0)
    ok = w.sum(axis=1) > 1e-3
    if ok.sum() >= 6:
        cen = (w[ok] * offs).sum(axis=1) / w[ok].sum(axis=1)
        pts = (q[ok] + pn[None] * cen[:, None]).astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        fd = np.array([vx, vy], float)
        fd = fd if fd @ d > 0 else -fd
        if fd @ d > math.cos(math.radians(2)):
            p0 = np.array([x0, y0], float)
            return fd, p0 + fd * ((start - p0) @ fd)     # ball centre snapped onto the line
    return d, s0


# ---------------------------------------------------------------- physics
def ray_circle(p, d, c, R):
    f = p - c
    b = f @ d
    disc = b * b - (f @ f - R * R)
    if disc < 0:
        return None
    t = -b - math.sqrt(disc)
    return t if t > 1e-6 else None


def ray_wall(p, d, t):
    ix0, iy0, ix1, iy1 = t.inner
    best, wall = 1e18, None
    for comp, lim, w in ((0, ix1, 'x'), (0, ix0, 'x'), (1, iy1, 'y'), (1, iy0, 'y')):
        if abs(d[comp]) > 1e-9:
            tt = (lim - p[comp]) / d[comp]
            if 1e-6 < tt < best:
                best, wall = tt, w
    return max(best, 0), wall


def bounce(d, wall, with_speed=False):
    """Cushion rebound: the part of the motion into the cushion comes back weaker."""
    d = d.copy()
    d[0 if wall == 'x' else 1] *= -E_CUSHION
    k = np.linalg.norm(d)
    return (d / k, k) if with_speed else d / k


def trace(p, d, balls, skip, t, bounces=MAX_BOUNCES, max_len=1e9, speed=None):
    """Follow a ball. Returns (points, end): ('ball', idx, contact, v) / ('pocket', xy, v) /
    ('stop', v, why). With a starting speed (px/s) the ball slows down and stops."""
    ix0, iy0, ix1, iy1 = t.inner
    p = np.array([min(max(p[0], ix0), ix1), min(max(p[1], iy0), iy1)], float)   # never start outside
    pts = [p.copy()]
    used = 0.0
    a = DECEL * t.r
    v2 = None if speed is None else speed * speed

    def go(dist):                       # roll `dist`; return False if it stops first
        nonlocal v2
        if v2 is None:
            return True
        if v2 <= 2 * a * dist:
            return False
        v2 -= 2 * a * dist
        return True

    def vel():
        return None if v2 is None else math.sqrt(max(v2, 0.0))

    for _ in range(bounces + 1):
        tb, bi = 1e18, None
        for i, b in enumerate(balls):
            if i in skip:
                continue
            tt = ray_circle(p, d, np.array(b[:2]), 2 * t.r)
            if tt is not None and tt < tb:
                tb, bi = tt, i
        tw, wall = ray_wall(p, d, t)
        step = min(tb, tw)
        if v2 is not None and v2 <= 2 * a * step:          # runs out of speed first
            pts.append(p + d * (v2 / (2 * a)))
            return pts, ('stop', 0.0, 'slow')
        if used + step > max_len:
            pts.append(p + d * (max_len - used))
            return pts, ('stop', vel(), 'far')
        if tb < tw:
            go(tb)
            p = p + d * tb
            pts.append(p)
            return pts, ('ball', bi, p, vel())
        go(tw)
        p = p + d * tw
        used += tw
        pts.append(p)
        pk = t.pocket_at(p, d)
        if pk is not None:
            pts[-1] = np.array(pk)
            return pts, ('pocket', pk, vel())
        d, k = bounce(d, wall, with_speed=True)
        if v2 is not None:
            v2 *= k * k
        skip = set()      # after a bounce the ball may come back to any ball
    return pts, ('stop', vel(), 'far')


def predict(t, balls, cue_i, aim, origin, ghost, m, v):
    out = {'cue_path': [], 'obj_path': [], 'ghost': None, 'pot': None, 'chain': [], 'chain_pot': None,
           'chains': [], 'chain_pots': [],
           'obj_ball': None, 'cue_after': [], 'cue_pot': None, 'locked': False}
    r = t.r
    a = DECEL * r
    power = getattr(t, 'power', None)
    v0 = V_PER_POWER * power * r if power else None     # known only while the bar is pulled
    out['power'] = power
    max_len = 1.5 * (t.bx1 - t.bx0)        # enough to see where it ends up, not clutter
    pts, end = trace(origin, aim, balls, {cue_i}, t, max_len=max_len, speed=v0)
    # The game's ghost ball is the ground truth for the first contact: it alone
    # decides what gets hit (a ball it really touches, else the cushion/pocket).
    if ghost is not None:
        near = [(abs(math.hypot(b[0] - ghost[0], b[1] - ghost[1]) - 2 * r), i)
                for i, b in enumerate(balls) if i != cue_i]
        # Away from the cushions the ghost can only have stopped because it touches a
        # ball, so accept small centre errors. Against a cushion it must clearly touch.
        at_rail = t.side_near(ghost, 0.35 * r) is not None
        near = [x for x in near if x[0] < (0.12 * r + 1.0 if at_rail else 0.45 * r)]
        L0 = np.linalg.norm(ghost - origin)
        vg = None if v0 is None else math.sqrt(max(v0 * v0 - 2 * a * L0, 0.0))
        if v0 is not None and vg <= 0:
            pts, end = [origin, origin + aim * (v0 * v0 / (2 * a))], ('stop', 0.0, 'slow')
        elif near:
            bi = min(near)[1]
            pts, end = [origin, ghost], ('ball', bi, ghost, vg)
        else:
            side = t.side_near(ghost, 0.6 * r)
            pk = t.pocket_at(ghost, aim) if side else None
            if pk is not None:
                pts, end = [origin, np.array(pk)], ('pocket', pk, vg)
            elif side:
                d2 = aim.copy()
                k = 0 if side in 'lr' else 1
                d2[k] = abs(d2[k]) * E_CUSHION * (1 if side in 'lt' else -1)   # away from that cushion
                kk = np.linalg.norm(d2)
                d2 /= kk
                rest, end = trace(ghost, d2, balls, set(), t, bounces=MAX_BOUNCES - 1,
                                  max_len=max_len - L0, speed=None if vg is None else vg * kk)
                pts = [origin] + rest
            else:
                pts, end = [origin, ghost], ('stop', vg, 'far')   # touches nothing we can see
    if ghost is not None:
        out['locked'] = True
        out['game_ghost'] = ghost
    out['cue_path'] = pts
    out['cue_stops'] = end[0] == 'stop' and len(end) > 2 and end[2] == 'slow'
    if end[0] == 'pocket':
        out['cue_pot'] = end[1]
        return out
    if end[0] != 'ball':
        return out
    bi, g = end[1], end[2]
    vc = end[3] if len(end) > 3 else None          # cue speed at contact
    out['ghost'] = g
    ob = np.array(balls[bi][:2], float)
    out['obj_ball'] = ob
    n = (ob - g) / np.linalg.norm(ob - g)
    vin = pts[-1] - pts[-2]
    vin = vin / (np.linalg.norm(vin) or 1)
    if len(pts) == 2:
        # first contact straight off the cue: the game's own object-ball hint is exact
        hint = read_hint(t, ob, n)
        if hint is not None:
            n, ob = hint
            balls[bi][0], balls[bi][1] = ob
            out['obj_ball'] = ob
            out['hint'] = True
    head_on = max(float(vin @ n), 0.0)
    vo = None if vc is None else K_OBJ * vc * head_on
    t.marginal = False
    opts, oend = trace(ob, n, balls, {bi, cue_i}, t, bounces=OBJ_BOUNCES,
                       max_len=1.5 * (t.bx1 - t.bx0), speed=vo)
    out['obj_path'] = opts
    out['obj_stops'] = oend[0] == 'stop' and len(oend) > 2 and oend[2] == 'slow'
    if oend[0] == 'pocket':
        out['pot'] = oend[1]
        # with the power known the speed is accounted for; without it, direct pots
        # and one-cushion banks are reliable but longer trips depend on power
        cut = math.degrees(math.acos(max(-1.0, min(1.0, float(vin @ n)))))
        out['cut'] = cut
        # very thin cuts magnify any error (and real balls 'throw'): be honest about it
        out['pot_sure'] = ((vo is not None or len(opts) - 2 <= 1) and not getattr(t, 'marginal', False)
                           and cut < THIN_CUT)
    # combination shots: each ball that gets hit pushes the next, up to 4 deep
    out['chains'], out['chain_pots'] = [], []
    end, moved, prev = oend, {bi, cue_i}, opts
    while end[0] == 'ball' and len(out['chains']) < 4:
        bj, c2 = end[1], end[2]
        b2 = np.array(balls[bj][:2], float)
        n2 = (b2 - c2) / np.linalg.norm(b2 - c2)
        din = prev[-1] - prev[-2]
        din = din / (np.linalg.norm(din) or 1)
        vb = None if end[3] is None else K_OBJ * end[3] * max(float(din @ n2), 0.0)
        moved.add(bj)
        cpts, end = trace(b2, n2, balls, set(moved), t, bounces=1,
                          max_len=0.8 * (t.bx1 - t.bx0), speed=vb)   # knocked-on balls carry less pace
        prev = cpts
        out['chains'].append(cpts)
        out['chain_pots'].append(end[1] if end[0] == 'pocket' else None)
    out['chain'] = out['chains'][0] if out['chains'] else []
    out['chain_pot'] = next((p for p in out['chain_pots'] if p is not None), None)
    tang = vin - (vin @ n) * n + K_FOLLOW * (vin @ n) * n     # sideways part + rolled-through part
    if np.linalg.norm(tang) > 0.05:
        va = None if vc is None else vc * float(np.linalg.norm(tang))
        tang = tang / np.linalg.norm(tang)
        cpts, cend = trace(g, tang, balls, {bi, cue_i}, t, bounces=2,
                           max_len=0.7 * (t.bx1 - t.bx0), speed=va)
        out['cue_after'] = cpts
        out['cue_after_stops'] = cend[0] == 'stop' and len(cend) > 2 and cend[2] == 'slow'
        if cend[0] == 'pocket':
            out['cue_pot'] = cend[1]
    return out


def calibrate(t, origin, aim, ghost, balls, cue_i):
    """When the game's ghost sits against a cushion, learn where that cushion is."""
    if ghost is None:
        return
    r = t.r
    if any(abs(math.hypot(b[0] - ghost[0], b[1] - ghost[1]) - 2 * r) < 0.2 * r
           for i, b in enumerate(balls) if i != cue_i):
        return
    _, wall = ray_wall(origin, aim, t)
    if wall == 'x':
        side = 'r' if aim[0] > 0 else 'l'
        val = ghost[0]
    else:
        side = 'b' if aim[1] > 0 else 't'
        val = ghost[1]
    sign = 1 if side in 'lt' else -1
    frac = sign * (val - t.edge(side)) / t.bh
    expected = BAND_RATIO + r / t.bh             # cushion width + one ball radius
    if not 0.5 * expected < frac < 1.8 * expected:
        return                          # impossible for a cushion: a misread, ignore it
    if side in CAL and abs(frac - CAL[side]) * t.bh > 0.5 * r:
        # far from what we learned: usually a misread, but if it keeps saying the same
        # thing the stored value is the wrong one - replace it
        pend = STATE.setdefault('cal_pending', {})
        prev = pend.get(side)
        if prev is not None and abs(prev[0] - frac) * t.bh < 0.3 * r:
            pend[side] = (prev[0], prev[1] + 1)
            if prev[1] + 1 >= 3:
                CAL[side] = frac
                pend.pop(side, None)
        else:
            pend[side] = (frac, 1)
        return
    CAL[side] = frac if side not in CAL else 0.6 * CAL[side] + 0.4 * frac
    STATE.setdefault('cal_measured', set()).add(side)
    opp = {'l': 'r', 'r': 'l', 't': 'b', 'b': 't'}[side]   # tables are symmetric
    CAL.setdefault(opp, CAL[side])


def needs_full_frame():
    """True when the next frame should be the whole screen (to (re)find the table)."""
    return STATE.get('table') is None or (STATE.get('n', 0) + 1) % 15 == 0


def table_region(W, H):
    """Screen rectangle analyse() will look at, for grabbing only that."""
    (bx0, by0, bx1, by1), _ = STATE['table']
    pad = int(4 * (STATE['r'] or BALL_RATIO * (bx1 - bx0)))
    x0, y0 = max(int(bx0) - pad, 0), max(int(by0) - pad, 0)
    return x0, y0, min(int(bx1) + pad, W), min(int(by1) + pad, H)


WORK_R = 13.0      # ball radius (px) the analysis works at


def _to_screen(res):
    """Scale an analysis result from the shrunken working image back to screen pixels
    (still relative to t.off)."""
    t, balls, ci, pr = res
    k = 1.0 / getattr(t, 'scale', 1.0)
    if k == 1.0:
        return res
    t2 = Table((t.bx0 * k, t.by0 * k, t.bx1 * k, t.by1 * k), t.hue, t.r * k)
    t2.off, t2.scale, t2.power = t.off, 1.0, getattr(t, 'power', None)
    t2.pockets = [(x * k, y * k, rr * k) for x, y, rr in getattr(t, 'pockets', [])]
    balls2 = [[b[0] * k, b[1] * k] + list(b[2:]) for b in balls]
    if pr:
        pr = dict(pr)
        for key, v in list(pr.items()):
            if key in ('aim', 'origin', 'steady', 'locked', 'hint', 'power') or v is None:
                continue
            if isinstance(v, (list, tuple)) and v and isinstance(v[0], (list, tuple, np.ndarray)):
                if key in ('chains',):
                    pr[key] = [[np.asarray(q, float) * k for q in ch] for ch in v]
                else:
                    pr[key] = [np.asarray(q, float) * k for q in v]
            elif key == 'chain_pots':
                pr[key] = [None if q is None else tuple(np.asarray(q, float) * k) for q in v]
            elif isinstance(v, np.ndarray) and v.shape == (2,):
                pr[key] = v * k
            elif isinstance(v, tuple) and len(v) == 2 and all(isinstance(x, (int, float, np.floating)) for x in v):
                pr[key] = (v[0] * k, v[1] * k)
    return (t2, balls2, ci, pr)


def analyse(img, base=(0, 0), full=None):
    """Full pipeline on one BGR screen frame (or the sub-region starting at `base`).
    Coordinates in the result are relative to t.off (screen position of the
    table crop), so drawers must translate by it."""
    STATE['n'] = STATE.get('n', 0) + 1
    if full is None:
        full = base == (0, 0) and img.shape[0] > 0
    if full and (STATE.get('table') is None or STATE['n'] % 15 == 0):   # re-find the table now and then
        t = find_table(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        if t is not None:
            pb = find_power_bar(img, (t.bx0, t.by0, t.bx1, t.by1))
            t.bx0, t.bx1, t.by0, t.by1 = t.bx0 + base[0], t.bx1 + base[0], t.by0 + base[1], t.by1 + base[1]
            if pb is not None:
                pb = (pb[0] + base[0], pb[1] + base[1], pb[2] + base[0], pb[3] + base[1])
                old_pb = STATE.get('pbar')
                if old_pb is not None and abs(old_pb[0] - pb[0]) < 60 and abs(old_pb[3] - pb[3]) < 30:
                    pb = (min(pb[0], old_pb[0]), min(pb[1], old_pb[1]), max(pb[2], old_pb[2]), max(pb[3], old_pb[3]))
                STATE['pbar'] = pb
        old = STATE.get('table')
        STATE['table'] = None if t is None else ((t.bx0, t.by0, t.bx1, t.by1), t.hue)
        if t is not None and old is not None and \
                abs((t.bx1 - t.bx0) / (old[0][2] - old[0][0]) - 1) > 0.03:
            # window resized: ball size scales with the table; cushion calibration is
            # stored relative to the table so it still holds
            if STATE['r']:
                STATE['r'] *= (t.bx1 - t.bx0) / (old[0][2] - old[0][0])
            STATE['smooth'] = None
            STATE['balls'] = None
            STATE['bg'] = None
    if STATE.get('table') is None:
        return None, 'table not found'
    (bx0, by0, bx1, by1), hue = STATE['table']
    r0 = STATE['r'] or BALL_RATIO * (bx1 - bx0)
    pad = int(4 * r0)
    ox, oy = max(int(bx0) - pad, 0), max(int(by0) - pad, 0)
    img = np.ascontiguousarray(img[oy - base[1]:int(by1) + pad - base[1], ox - base[0]:int(bx1) + pad - base[0]])
    if img.size == 0:
        STATE['table'] = None
        return None, 'table moved'
    own = STATE.get('own')
    own_crop = None if own is None else \
        own[oy - base[1]:oy - base[1] + img.shape[0], ox - base[0]:ox - base[0] + img.shape[1]]
    # Work at a standard size (balls ~24 px across) whatever the window size: big
    # tables are shrunk (much faster, same accuracy); results are scaled back at the end.
    sc = min(1.0, WORK_R / r0)
    prev = STATE.get('work_scale')
    if prev is not None and abs(sc / prev - 1) < 0.08:
        sc = prev                          # keep the same zoom: tiny size wobble must not matter
    else:
        sc = round(sc * 20) / 20 if sc < 0.97 else 1.0
        STATE['work_scale'] = sc
    if sc < 0.97:
        img = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
        if own_crop is not None and own_crop.size:
            own_crop = cv2.resize(own_crop.astype(np.uint8), (img.shape[1], img.shape[0]),
                                  interpolation=cv2.INTER_NEAREST) > 0
    else:
        sc = 1.0
    STATE['own_crop'] = own_crop
    t = Table(((bx0 - ox) * sc, (by0 - oy) * sc, (bx1 - ox) * sc, (by1 - oy) * sc), hue, r0 * sc)
    t.off = (ox, oy)
    t.scale = sc
    pw = STATE.get('power')
    t.power = pw if pw is not None and pw > 0.02 else None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    t.gray = gray
    pc = STATE.get('pockets')
    if pc is None or pc[0] != t.off or STATE['n'] % 30 == 0:
        STATE['pockets'] = pc = (t.off, find_pockets(gray, t))
    t.pockets = pc[1]
    felt_maps(img, hsv, t)
    thumb = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    bc = STATE.get('balls')
    balls = None
    if bc is not None and bc[0] == t.off and bc[1].shape == thumb.shape:
        changed = np.abs(thumb - bc[1]) > 25
        oc = STATE.get('own_crop')
        if oc is not None and oc.shape == gray.shape and oc.any():
            changed &= cv2.resize(oc.astype(np.uint8), (thumb.shape[1], thumb.shape[0]),
                                  interpolation=cv2.INTER_AREA) == 0
        rq = max(1, int(round(t.r * 0.25 * 0.5)))
        moved = any(changed[max(int(b[1] / 4) - rq, 0):int(b[1] / 4) + rq + 1,
                            max(int(b[0] / 4) - rq, 0):int(b[0] / 4) + rq + 1].any()
                    for b in bc[2])
        if not moved and changed.mean() < 0.04 and STATE['n'] - bc[3] < 6:
            balls = [list(b) for b in bc[2]]
    if balls is None:
        balls = find_balls(img, hsv, gray, t)
        STATE['balls'] = (t.off, thumb, [list(b) for b in balls], STATE['n'])
    if not balls:
        STATE['table'] = None
        return _to_screen((t, [], None, None)), 'no balls'
    m = line_mask(t)
    aim = find_aim(hsv, m, t, balls)
    if aim is None:
        return _to_screen((t, balls, None, None)), 'waiting for aim line'
    cue_i, d, origin, last = aim
    STATE['cue_xy'] = tuple(balls[cue_i][:2])
    balls[cue_i][0], balls[cue_i][1] = origin
    ghost, grad = find_ghost(t.con, origin, d, last, t.r)
    if ghost is not None and STATE['r'] is not None and abs(grad / t.scale / STATE['r'] - 1) > 0.15:
        ghost = None                    # wrong size for a ghost ball: not the game's ring
    if ghost is not None and (STATE.get('aim_src') == 'ring' or
                              (last < 8 * t.r and STATE.get('aim_src') != 'stick')):
        g = (ghost - origin) / np.linalg.norm(ghost - origin)
        if g @ d > math.cos(math.radians(3)):
            d = g                       # short line and no stick: the ghost ring pins the aim
    if ghost is not None:
        ghost = origin + d * ((ghost - origin) @ d)    # ghost sits on the aim line, by definition
    # temporal smoothing: tiny frame-to-frame differences are noise, average them;
    # a real change (the player moving the cue) resets immediately
    sm = STATE.get('smooth')
    if (sm is not None and sm['d'] @ d > math.cos(math.radians(0.3)) and
            np.linalg.norm(sm['o'] - origin) < 1.5 and (ghost is None) == (sm['g'] is None) and
            (ghost is None or np.linalg.norm(sm['g'] - ghost) < 3)):
        a = 0.25
        d = (1 - a) * sm['d'] + a * d
        d /= np.linalg.norm(d)
        origin = (1 - a) * sm['o'] + a * origin
        if ghost is not None:
            ghost = (1 - a) * sm['g'] + a * ghost
            ghost = origin + d * ((ghost - origin) @ d)
        STATE['steady'] = True
    else:
        STATE['steady'] = sm is None
    STATE['smooth'] = {'d': d, 'o': origin, 'g': ghost}
    balls[cue_i][0], balls[cue_i][1] = origin
    # learn ball size / cushion positions only from calm, repeated frames
    # (frames with balls rolling would teach it nonsense)
    if ghost is not None and (STATE['steady'] or STATE['r'] is None):
        gs = grad / t.scale                               # back to screen pixels
        STATE['r'] = gs if STATE['r'] is None else 0.85 * STATE['r'] + 0.15 * gs
        t.r = STATE['r'] * t.scale
        STATE['r_learned'] = True
        STATE['r_ratio'] = STATE['r'] / ((t.bx1 - t.bx0) / t.scale)
    if STATE['steady']:
        calibrate(t, origin, d, ghost, balls, cue_i)
    pr = predict(t, balls, cue_i, d, origin, ghost, m, t.con)
    # Confidence gate: either the game's ghost ring is locked, or the game's own
    # visible aim line ends where we predict the first contact. Otherwise we may
    # be looking at the wrong thing (cue ball covered, balls moving...): draw nothing.
    if not pr['locked']:
        first = np.linalg.norm(pr['cue_path'][1] - origin) if len(pr['cue_path']) > 1 else 0
        if abs(first - last) > 2.5 * t.r:
            return _to_screen((t, balls, cue_i, None)), 'unsure'
    pr['aim'] = d
    pr['origin'] = origin / t.scale + np.array(t.off, float)
    pr['steady'] = STATE['steady']
    return _to_screen((t, balls, cue_i, pr)), 'ok' + (' (locked)' if pr['locked'] else '')


# ---------------------------------------------------------------- debug drawing (test mode)
def draw_cv(img, res):
    t, balls, cue_i, pr = res
    o = img.copy()
    P = lambda q: (int(round(q[0] + t.off[0])), int(round(q[1] + t.off[1])))
    nx0, ny0, nx1, ny1 = t.nose
    cv2.rectangle(o, P((nx0, ny0)), P((nx1, ny1)), (255, 0, 255), 1)
    for i, b in enumerate(balls):
        cv2.circle(o, P(b), int(t.r), (0, 0, 255) if i != cue_i else (255, 255, 0), 1)
    if pr:
        for a, b in zip(pr['cue_path'], pr['cue_path'][1:]): cv2.line(o, P(a), P(b), (255, 255, 0), 1)
        for a, b in zip(pr['obj_path'], pr['obj_path'][1:]): cv2.line(o, P(a), P(b), (0, 255, 255), 2)
        for a, b in zip(pr['cue_after'], pr['cue_after'][1:]): cv2.line(o, P(a), P(b), (255, 128, 0), 1)
        if pr['ghost'] is not None: cv2.circle(o, P(pr['ghost']), int(t.r), (255, 0, 255), 1)
        if pr['pot'] is not None: cv2.circle(o, P(pr['pot']), int(2 * t.r), (0, 255, 0), 3)
    return o


# ---------------------------------------------------------------- overlay app
# Exact colours the overlay draws with (RGB). Drawn without antialiasing and
# fully opaque, so the analyser can recognise and ignore its own drawings.
COL = {
    'maybe':  (254, 221, 1),
    'cue':    (0, 229, 254),
    'obj':    (254, 169, 1),
    'pot':    (41, 254, 81),
    'chain':  (254, 121, 201),
    'after':  (1, 199, 254),
    'scratch': (254, 41, 41),
}


_OWN_LUT = np.zeros((3, 256), np.uint8)       # per channel: bit k set if colour k has that value
for _k, _rgb in enumerate(COL.values()):
    for _c, _v in enumerate(_rgb[::-1]):          # BGR order
        _OWN_LUT[_c, _v] |= 1 << _k


def own_mask(img):
    """Pixels the overlay itself drew (they show up in screen captures)."""
    b, g, r = cv2.split(img)
    m = cv2.bitwise_and(cv2.bitwise_and(cv2.LUT(b, _OWN_LUT[0]), cv2.LUT(g, _OWN_LUT[1])),
                        cv2.LUT(r, _OWN_LUT[2]))
    return cv2.dilate((m > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == '--test':
        for f in sys.argv[2:]:
            STATE.clear()
            STATE['r'] = None
            CAL.clear()
            img = cv2.imread(f)
            res, st = analyse(img)
            print(f, st, '' if not res else f"r={res[0].r:.2f} balls={len(res[1])} cue={res[2]}")
            if res and res[3]:
                pr = res[3]
                print('   ghost:', None if pr['ghost'] is None else np.round(pr['ghost'], 1),
                      ' pot:', pr['pot'], ' scratch:', pr['cue_pot'], ' cal:', {k: round(v, 4) for k, v in CAL.items()})
            if res:
                cv2.imwrite(f + '.out.png', draw_cv(img, res))
    else:
        import app
        app.main()
