"""PoolSight Guide - a simple manual overlay for your pool game.

Set up once: mark the felt (cushion edge to cushion edge) and the ball size.
    1        put the point where the mouse is (on a ball): lines from every pocket meet there
    2        start / freeze the line from that point (follows the mouse while moving)
    Up/Down  turn the line 0.1 deg   (Shift + Up/Down: 1 deg)
    F8       show / hide        F9  quit
Keys 1, 2 and the arrows are only taken while the guide is switched on.
"""
import json
import logging
import math
import os
import signal
import sys
import threading
import time

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, 'guide.json')
log = logging.getLogger('guide')

# ---- measured from staged test shots in the game (fractions of the marked felt box) ----
BALL_FRAC = 14.5 / 980          # ball radius / felt width (best ghost-ball size in cut tests)
CORNER_AIM = (14.15 / 980, 12.15 / 490)   # where shots really drop: inset from each corner
MID_AIM = 5.95 / 490            # middle pocket aim point, in from the cushion line
CORNER_MOUTH = 18.8 / 980       # half-width of the line that still drops (median)
MID_MOUTH = 12.6 / 980
CORNER_MAX_ANGLE = 55.0         # steepest approach (deg from the pocket's axis) seen to drop
MID_MAX_ANGLE = 65.0
# speed (game units/s; 980 units = felt width) - measured with straight test shots
POWER_V0 = [(0.0, 0), (0.1, 264), (0.2, 446), (0.3, 644), (0.4, 858), (0.5, 1090), (0.6, 1338),
            (0.7, 1604), (0.8, 1886), (0.9, 2184), (1.0, 2500)]
SLIDE_DECEL = 1260              # right after a hit / bounce, until the ball is rolling
ROLL_DECEL = 190                # once rolling
ROLL_FRAC = 5 / 7               # it starts rolling at ~5/7 of the speed it was sliding with
# off a cushion: tan(out) = k * tan(in), k depends on the speed at the cushion (measured)
BOUNCE_K = [(0, 0.643), (299, 0.643), (488, 0.663), (877, 0.733), (1271, 0.755), (1690, 0.765), (2142, 0.771),
            (3000, 0.775)]
# measured cushion grid: rows = speed into the cushion, columns = angle to the cushion (deg)
CUSH_V = [299, 488, 877, 1271, 1690, 2142]
CUSH_A = [5, 10, 15, 20, 30, 40, 50, 60, 70, 75, 80, 84, 87]
CUSH_ROLL = [   # rolling speed once the ball has settled after the cushion
    [298, 296, 293, 288, 276, 260, 242, 224, 207, 201, 196, 194, 193],
    [459, 456, 451, 444, 426, 403, 377, 350, 327, 318, 311, 307, 306],
    [693, 689, 684, 675, 653, 624, 593, 561, 534, 523, 515, 511, 510],
    [955, 950, 942, 932, 903, 867, 827, 787, 753, 740, 730, 725, 723],
    [1243, 1237, 1227, 1214, 1179, 1133, 1083, 1034, 992, 975, 963, 957, 954],
    [1561, 1554, 1541, 1525, 1481, 1423, 1362, 1299, 1246, 1225, 1210, 1202, 1198]]
CUSH_SLIDE = [  # distance it skids before it settles
    [1, 2, 2, 3, 5, 6, 6, 7, 7, 7, 7, 7, 7],
    [11, 11, 12, 13, 16, 18, 19, 20, 21, 21, 21, 20, 20],
    [113, 113, 112, 110, 107, 102, 96, 91, 87, 85, 84, 83, 83],
    [277, 275, 271, 267, 254, 239, 222, 207, 194, 189, 186, 183, 183],
    [517, 513, 506, 497, 471, 441, 407, 377, 351, 342, 335, 331, 330],
    [848, 841, 830, 815, 772, 723, 667, 618, 576, 561, 549, 543, 541]]
CUSH_SHIFT = [  # the settled path is shifted back along the cushion by this much
    [.1, .1, .2, .3, .4, .5, .5, .4, .3, .2, .2, .1, .1],
    [.9, 1, 1, 1.1, 1.3, 1.3, 1.3, 1.1, .8, .6, .4, .3, .1],
    [4, 4, 3.9, 3.8, 3.5, 3.1, 2.5, 2, 1.3, 1, .7, .4, .2],
    [5.1, 5, 4.9, 4.7, 4.3, 3.7, 3, 2.3, 1.5, 1.1, .8, .5, .2],
    [5.6, 5.5, 5.4, 5.2, 4.7, 4, 3.2, 2.4, 1.6, 1.2, .8, .5, .2],
    [5.9, 5.8, 5.7, 5.5, 4.9, 4.2, 3.4, 2.5, 1.7, 1.3, .9, .5, .2]]
# ball arriving ROLLING (the usual case in a bank) - measured separately
RIN_V = [490, 780, 1200, 1500, 2200]
RIN_A = [10, 20, 30, 45, 60, 75, 87]
RIN_K = [[.643, .643, .643, .643, .643, .643, .643],
         [.643, .643, .643, .643, .643, .643, .643],
         [.711, .707, .700, .680, .643, .643, .643],
         [.710, .707, .700, .690, .696, .672, .672],
         [.730, .730, .725, .720, .720, .722, .722]]
RIN_KEEP = [[.99, .965, .925, .84, .75, .67, .643],     # share of speed kept (rolling after)
            [.99, .965, .925, .84, .75, .67, .644],
            [.83, .82, .80, .77, .75, .67, .644],
            [.80, .79, .77, .73, .68, .64, .62],
            [.75, .74, .72, .68, .62, .60, .59]]
RIN_SLIDE = [[5, 9, 12, 14, 12, 9, 9],
             [12, 22, 31, 38, 39, 37, 37],
             [189, 178, 160, 122, 82, 81, 81],
             [300, 290, 270, 230, 244, 192, 192],
             [700, 680, 650, 600, 560, 520, 509]]
CUSHION_INSET = 0.038     # cushion strip width / table height (measured from your manual mark)
E_CUSHION = 0.64        # measured in your game: cushions return 64% of the speed into them

TUNE_DEFAULT = {'bounce': 1.04, 'slow': 1.59, 'mouth': 0.5, 'corner': 53.0}   # player's saved defaults
KEYS_DEFAULT = {'point': '1', 'line': '2', 'learn': '3', 'up': 'Up', 'down': 'Down',
                'toggle': 'F8', 'quit': 'F9'}
KEY_NAMES = {'point': 'Lock ball', 'line': 'Line / pocket', 'learn': 'Learn shot',
             'up': 'Turn up', 'down': 'Turn down', 'toggle': 'Show/hide', 'quit': 'Quit'}
# mouse binds are stored as 'Mouse<X11 button>' (left click is never taken - the game needs it)
MOUSE_LABELS = {2: 'Middle', 3: 'Right', 4: 'Wheel↑', 5: 'Wheel↓', 6: 'Wheel←', 7: 'Wheel→',
                8: 'Back', 9: 'Fwd'}


def key_label(name):
    """How a bound key/button is shown to the player."""
    if name.startswith('Mouse') and name[5:].isdigit():
        return MOUSE_LABELS.get(int(name[5:]), 'M' + name[5:])
    return {'0x1008ff26': 'Back', '0x1008ff27': 'Fwd'}.get(name, name)


CFG = {'keys': dict(KEYS_DEFAULT), 'tune': dict(TUNE_DEFAULT), 'power': 0.5, 'felt': None, 'r': 15.0, 'bounces': 3, 'real': True, 'thick': 4, 'point': None,
       'on': True, 'panel_pos': None}
LIVE = {'moving': False, 'dir': None, 'shot': None, 'balls': []}

COL_POCKET = QtGui.QColor(190, 120, 255, 220)
COL_LINE = QtGui.QColor(0, 229, 254)
COL_POT = QtGui.QColor(41, 254, 81)


def load():
    try:
        CFG.update(json.load(open(CONFIG)))
    except Exception:
        # first run: take the table marked earlier in the old app, if any
        try:
            old = json.load(open(os.path.join(HERE, 'calibration.json'))).get('manual', {})
            for k in ('felt', 'r', 'bounces', 'real'):
                if old.get(k) is not None:
                    CFG[k] = old[k]
        except Exception:
            pass


def save():
    tmp = CONFIG + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(CFG, f, indent=1)
    os.replace(tmp, CONFIG)


_TL = threading.local()


def grabber():
    """One screen grabber per thread, reused. (Opening a new one per grab leaks a
    display connection each time until the X server refuses new clients.)"""
    g = getattr(_TL, 'sct', None)
    if g is None:
        import mss
        g = mss.MSS() if hasattr(mss, 'MSS') else mss.mss()
        _TL.sct = g
    return g


def pockets():
    """Where each pocket really takes the ball (measured), in screen pixels:
    top-left, top-middle, top-right, bottom-left, bottom-middle, bottom-right."""
    x0, y0, x1, y1 = CFG['felt']
    w, h = x1 - x0, y1 - y0
    cx, cy = CORNER_AIM[0] * w, CORNER_AIM[1] * h
    mx, my = (x0 + x1) / 2, MID_AIM * h
    return [(x0 + cx, y0 + cy), (mx, y0 + my), (x1 - cx, y0 + cy),
            (x0 + cx, y1 - cy), (mx, y1 - my), (x1 - cx, y1 - cy)]


def tune(k):
    return CFG.get('tune', {}).get(k, TUNE_DEFAULT[k])


def pocket_mouth(i):
    w = CFG['felt'][2] - CFG['felt'][0]
    return (MID_MOUTH if i in (1, 4) else CORNER_MOUTH) * w * tune('mouth')


POCKET_AXIS = [(-1, -1), (0, -1), (1, -1), (-1, 1), (0, 1), (1, 1)]   # outward direction


def pocket_hit(pt, d, limit):
    """Does a ball at pt rolling along d drop in a pocket before travelling `limit`?"""
    best = None
    for i, A in enumerate(pockets()):
        w = np.array(A) - pt
        t = float(w @ d)
        if t <= 0 or t > limit + 3 * CFG['r']:
            continue
        if abs(d[0] * w[1] - d[1] * w[0]) > pocket_mouth(i):
            continue
        ax = np.array(POCKET_AXIS[i], float)
        ax /= np.linalg.norm(ax)
        ang = math.degrees(math.acos(max(-1.0, min(1.0, float(d @ ax)))))
        if ang > (MID_MAX_ANGLE if i in (1, 4) else tune('corner')):
            continue                                  # too flat: hits the jaw
        if best is None or t < best[0]:
            best = (t, i, np.array(A))
    return best


def measure_ball_size(felt):
    """Measure the real ball radius on screen: find the round balls on the felt
    (circle detection + edge fit) and take the typical size. None if not enough balls."""
    import cv2
    import mss
    x0, y0, x1, y1 = [int(v) for v in felt]
    sct = grabber()
    img = np.ascontiguousarray(np.asarray(sct.grab({'left': x0, 'top': y0, 'width': x1 - x0,
                                                    'height': y1 - y0}))[..., :3])
    guess = (x1 - x0) / 76
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 1.2)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=1.5 * guess,
                               param1=60, param2=14, minRadius=int(guess * 0.6),
                               maxRadius=int(guess * 1.5) + 1)
    if circles is None:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    felt_h = np.median(hsv[..., 0])
    gf = gray.astype(np.float32)
    radii = []
    ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    for cx, cy, rr in circles[0]:
        # skip empty rings / felt (a ball is mostly not felt-coloured inside)
        yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
        inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= (0.6 * rr) ** 2
        hh = hsv[..., 0][inside]
        if len(hh) == 0 or (np.abs(hh.astype(int) - felt_h) < 8).mean() > 0.5:
            continue
        # edge-fit radius: strongest brightness change along 48 rays
        rad = np.arange(0.6 * rr, 1.5 * rr, 0.25)
        xs = (cx + np.cos(ang)[:, None] * rad).astype(np.float32)
        ys = (cy + np.sin(ang)[:, None] * rad).astype(np.float32)
        if xs.min() < 1 or ys.min() < 1 or xs.max() > img.shape[1] - 2 or ys.max() > img.shape[0] - 2:
            continue
        v = cv2.remap(gf, xs, ys, cv2.INTER_LINEAR)
        k = np.abs(np.diff(v, axis=1)).argmax(axis=1)
        radii.append(float(np.median((rad[k] + rad[k + 1]) / 2)))
    if len(radii) < 2:
        # late in a game: measure the cue ball alone (solid white disc)
        white = ((hsv[..., 1] < 45) & (hsv[..., 2] > 205)).astype(np.uint8)
        n, _, st, _ = cv2.connectedComponentsWithStats(white)
        areas = [st[i, 4] for i in range(1, n)
                 if 0.5 * guess * guess < st[i, 4] < 6 * guess * guess
                 and 0.7 < st[i, 2] / max(st[i, 3], 1) < 1.4]
        if not areas:
            return None
        return round(float(math.sqrt(max(areas) / math.pi)), 1)
    return round(float(np.median(radii)), 1)


def find_cue(felt, r):
    """The cue ball: the one solid white, ball-sized blob on the felt."""
    import cv2
    import mss
    x0, y0, x1, y1 = [int(v) for v in felt]
    sct = grabber()
    img = np.ascontiguousarray(np.asarray(sct.grab({'left': x0, 'top': y0, 'width': x1 - x0,
                                                    'height': y1 - y0}))[..., :3])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 1] < 45) & (hsv[..., 2] > 205)).astype(np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, st, ce = cv2.connectedComponentsWithStats(white)
    best = None
    for i in range(1, n):
        w, h, area = st[i, 2], st[i, 3], st[i, 4]
        if not (1.2 * r * r < area < 4.5 * r * r) or not (0.6 < w / max(h, 1) < 1.6) or area < 0.55 * w * h:
            continue
        if best is None or area > best[0]:
            best = (area, ce[i])
    if best is None:
        return None
    # exact centre from the ball's edge against the felt (the stick tip doesn't count)
    c = snap_in_image(img, best[1][0], best[1][1], r)
    cx, cy = (c if c is not None and math.hypot(c[0] - best[1][0], c[1] - best[1][1]) < r
              else best[1])
    return [x0 + float(cx), y0 + float(cy)]


def solid_mask(img, r, felt_h=None):
    """Pixels that belong to balls: colour clearly different from the felt (compared in
    full Lab colour, not just hue - so blue balls on blue felt still count; shadows on the
    felt stay felt), with everything thinner than about 0.7 ball radii removed - that
    strips out the cue stick, the game's aim line and our own lines. Returns (opened, raw)."""
    import cv2
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, s_ = hsv[..., 0].astype(int), hsv[..., 1]
    if felt_h is None:
        sat = s_ > 70
        felt_h = int(np.median(h[sat])) if sat.any() else int(np.median(h))
    dh = np.abs(h - felt_h)
    dh = np.minimum(dh, 180 - dh)
    cand = (dh <= 10) & (s_ > 60)
    ref = np.median(lab[cand], axis=0) if cand.sum() > 50 else np.median(lab.reshape(-1, 3), axis=0)
    # chroma (a, b) decides; lightness only loosely, so shadows/lighting stay felt
    dc = np.hypot(lab[..., 1] - ref[1], lab[..., 2] - ref[2])
    dl = np.abs(lab[..., 0] - ref[0])
    feltlike = (dc < 9) & (dl < 70)
    raw = (~feltlike).astype(np.uint8)
    k = max(3, int(0.7 * r) | 1)
    opened = cv2.morphologyEx(raw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return opened, raw


def snap_in_image(img, mx, my, r):
    """Centre (in img coords) of the ball nearest (mx, my), from where the ball meets
    the felt - not from stripes or numbers inside it. None if no ball is close."""
    import cv2
    opened, raw = solid_mask(img, r)
    dist = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
    peak = dist == cv2.dilate(dist, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(r) | 1, int(r) | 1)))
    ys, xs = np.nonzero(peak & (dist > 0.55 * r))
    if len(xs) == 0:
        return None
    d = np.hypot(xs - mx, ys - my)
    k = int(np.argmin(d))
    if d[k] > 2.8 * r:
        return None
    cx, cy = float(xs[k]), float(ys[k])
    # refine: along 72 rays, where does the ball end and the felt begin?
    fm = cv2.GaussianBlur(raw.astype(np.float32), (3, 3), 0.7)
    ang = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    rad = np.arange(0.5 * r, 1.6 * r, 0.25)
    for _ in range(2):
        rx = (cx + np.cos(ang)[:, None] * rad).astype(np.float32)
        ry = (cy + np.sin(ang)[:, None] * rad).astype(np.float32)
        vals = cv2.remap(fm, rx, ry, cv2.INTER_LINEAR, borderValue=0)
        edge = []
        for i in range(len(ang)):
            below = np.nonzero(vals[i] < 0.5)[0]
            if len(below) == 0 or below[0] == 0:
                continue
            j = below[0]                                    # first felt sample
            a, b = vals[i][j - 1], vals[i][j]               # sub-sample crossing
            er = rad[j - 1] + (a - 0.5) / max(a - b, 1e-6) * (rad[j] - rad[j - 1])
            if 0.75 * r < er < 1.25 * r:                    # touching balls / stick: skip that ray
                edge.append((cx + math.cos(ang[i]) * er, cy + math.sin(ang[i]) * er))
        if len(edge) < 20:
            break
        ex, ey = np.array(edge).T
        A = np.c_[ex, ey, np.ones_like(ex)]
        sol = np.linalg.lstsq(A, ex ** 2 + ey ** 2, rcond=None)[0]
        fx, fy = sol[0] / 2, sol[1] / 2
        if math.hypot(fx - cx, fy - cy) > 0.4 * r:
            break
        cx, cy = fx, fy
    return cx, cy


def snap_to_ball(x, y, r):
    """Exact centre of the ball nearest (x, y) on screen, or None."""
    R = int(4 * r)
    x0, y0 = int(x) - R, int(y) - R
    img = np.ascontiguousarray(np.asarray(grabber().grab(
        {'left': x0, 'top': y0, 'width': 2 * R, 'height': 2 * R}))[..., :3])
    c = snap_in_image(img, x - x0, y - y0, r)
    if c is not None:
        return [x0 + c[0], y0 + c[1]]
    return snap_to_ball_old(x, y, r)


def snap_to_ball_old(x, y, r):
    """Exact centre of the ball nearest (x, y): circle detection near the mouse, then
    a sub-pixel circle fit to the ball's edge. None if no ball is close."""
    import cv2
    import mss
    R = int(4 * r)
    sct = grabber()
    x0, y0 = int(x) - R, int(y) - R
    img = np.ascontiguousarray(np.asarray(sct.grab({'left': x0, 'top': y0, 'width': 2 * R,
                                                    'height': 2 * R}))[..., :3])
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 1.0)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=1.4 * r, param1=60,
                               param2=12, minRadius=int(0.75 * r), maxRadius=int(1.3 * r) + 1)
    if circles is None:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    felt_h = int(np.median(hsv[..., 0]))
    best = None
    for cx, cy, _ in circles[0]:
        d = math.hypot(cx - R, cy - R)
        if d > 2.5 * r:
            continue
        # a ball, not the game's hollow ghost ring: mostly non-felt inside
        yy, xx = np.ogrid[:2 * R, :2 * R]
        inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= (0.6 * r) ** 2
        hh = hsv[..., 0][inside].astype(int)
        if len(hh) == 0 or (np.abs(hh - felt_h) < 8).mean() > 0.5:
            continue
        if best is None or d < best[0]:
            best = (d, cx, cy)
    if best is None:
        return None
    _, cx, cy = best
    # refine: strongest edge along 48 rays, then an algebraic circle fit
    gf = gray.astype(np.float32)
    ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    rad = np.arange(0.6 * r, 1.45 * r, 0.25)
    xs = (cx + np.cos(ang)[:, None] * rad).astype(np.float32)
    ys = (cy + np.sin(ang)[:, None] * rad).astype(np.float32)
    v = cv2.remap(gf, xs, ys, cv2.INTER_LINEAR)
    k = np.abs(np.diff(v, axis=1)).argmax(axis=1)
    er = (rad[k] + rad[k + 1]) / 2
    ok = np.abs(er - r) < 0.3 * r
    if ok.sum() >= 16:
        ex = cx + np.cos(ang)[ok] * er[ok]
        ey = cy + np.sin(ang)[ok] * er[ok]
        A = np.c_[ex, ey, np.ones_like(ex)]
        sol = np.linalg.lstsq(A, ex ** 2 + ey ** 2, rcond=None)[0]
        fx, fy = sol[0] / 2, sol[1] / 2
        if math.hypot(fx - cx, fy - cy) < 0.35 * r:
            cx, cy = fx, fy
    return [x0 + float(cx), y0 + float(cy)]


def detect_balls(felt, r):
    """Centres of all balls on the felt (round, not felt-coloured inside)."""
    import cv2
    import mss
    pad = int(1.5 * r)                    # balls touching a cushion stick out past the felt box
    x0, y0, x1, y1 = [int(v) for v in felt]
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    sct = grabber()
    img = np.ascontiguousarray(np.asarray(sct.grab({'left': x0, 'top': y0, 'width': x1 - x0,
                                                    'height': y1 - y0}))[..., :3])
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 1.0)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=1.5 * r, param1=60,
                               param2=11, minRadius=int(0.75 * r), maxRadius=int(1.3 * r) + 1)
    if circles is None:
        return []
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    felt_h = int(np.median(hsv[..., 0]))
    H, W = gray.shape
    out = []
    R = int(r)
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    disc = xx ** 2 + yy ** 2 <= (0.6 * r) ** 2
    for cx, cy, _ in circles[0]:
        a, b = int(round(cy)), int(round(cx))
        if a - R < 0 or b - R < 0 or a + R >= H or b + R >= W:
            continue
        hh = hsv[a - R:a + R + 1, b - R:b + R + 1, 0][disc].astype(int)
        if (np.abs(hh - felt_h) < 8).mean() > 0.35:       # felt, or the game's hollow aim ring
            continue                                      # (even with a red X drawn in it)
        out.append(np.array([x0 + cx, y0 + cy], float))
    return out


def first_block(pts, balls, skip, r):
    """Where a ball travelling along pts first touches another ball. The moving ball
    is 2 radii wide, so anything whose centre comes within 2r of the line is hit,
    even if the thin line misses it. Returns (segment index, touch point, ball) or None."""
    for si, (a, b) in enumerate(zip(pts, pts[1:])):
        seg = b - a
        L = np.linalg.norm(seg)
        if L < 1e-6:
            continue
        u = seg / L
        best = None
        for q in balls:
            if any(np.linalg.norm(q - sk) < 0.8 * r for sk in skip):
                continue
            w = q - a
            along = w @ u
            side = abs(u[0] * w[1] - u[1] * w[0])
            if side >= 2 * r:
                continue
            t = along - math.sqrt(4 * r * r - side * side)       # first touch along the line
            if 0 < t < L and (best is None or t < best[0]):
                best = (t, q)
        if best:
            return si, a + u * best[0], best[1]
    return None


def nearest_pocket(pt):
    pk = pockets()
    return min(range(len(pk)), key=lambda i: math.hypot(pt[0] - pk[i][0], pt[1] - pk[i][1]))


def obj_speed(c, d, cue=None):
    """Speed (game units/s) the target ball starts with when sent along d at the set
    power: the cue ball slows on its way to the ghost spot, and a thinner hit passes on
    less speed (measured). Without a known cue ball, assume a medium-length shot."""
    unit = (CFG['felt'][2] - CFG['felt'][0]) / 980.0
    v0 = start_speed(CFG.get('power', 0.5))
    g = np.array(c, float) - d * 2 * CFG['r']
    if cue is not None and np.linalg.norm(np.array(cue) - g) > 1:
        u = g - np.array(cue, float)
        L = np.linalg.norm(u)
        cosc = max(0.05, float((u / L) @ d))
        dist = L / unit
    else:
        cosc, dist = 1.0, 300.0
    ball = Roll(v0)
    if ball.go(dist) < dist - 1e-6:
        return 0.0
    return ball.v * cosc * 0.975


def path_blocked(c, d, pts, balls, cue=None):
    """first_block for the target ball's path. From the first cushion on, the cue ball
    is in the way too: after the hit it rolls on along the shot line ('follow'), so it
    is treated as a row of balls from the contact spot to where it stops."""
    blk = first_block(pts[:2], balls, [c], CFG['r'])
    if blk is not None or len(pts) <= 2:
        return blk
    r = CFG['r']
    unit = (CFG['felt'][2] - CFG['felt'][0]) / 980.0
    ghost = np.array(c, float) - d * 2 * r
    follow = 0.0
    if cue is not None and np.linalg.norm(np.array(cue) - ghost) > 1:
        u = ghost - np.array(cue, float)
        L = np.linalg.norm(u)
        cosc = max(0.0, float((u / L) @ d))
        ball = Roll(start_speed(CFG.get('power', 0.5)))
        ball.go(L / unit)
        v_follow = ball.v * cosc * (2 / 7)        # the hit takes its pace, the spin brings it back
        follow = 0.0 * v_follow    # follow-through model was too cautious in tests: contact spot only
    x0, y0, x1, y1 = CFG['felt']
    along = []
    for t in np.arange(0, follow + 1e-6, r):
        q = ghost + d * t
        if not (x0 + r <= q[0] <= x1 - r and y0 + r <= q[1] <= y1 - r):
            break
        along.append(q)
    rest = first_block(pts[1:], list(balls) + (along or [ghost]), [c], r)
    return None if rest is None else (rest[0] + 1, rest[1], rest[2])


def solve_to_pocket(c, target, balls, cue=None):
    """All directions from c that roll into pocket `target` (direct and off cushions).
    Each window of working angles is reduced to its middle (most room for error).
    Returns the best: clear corridor first, then fewest bounces, then widest window."""
    angs = np.radians(np.arange(0, 360, 0.25))
    hits = []
    for a in angs:
        d = np.array([math.cos(a), math.sin(a)])
        pts, pot, _ = rebound_path(c, d, max(CFG['bounces'], 1), obj_speed(c, d, cue))
        ok = pot is not None and nearest_pocket(pot) == target
        hits.append((ok, len(pts) - 2 if ok else -1))
    runs, i, n = [], 0, len(hits)
    while i < n:
        if hits[i][0]:
            j = i
            while j + 1 < n and hits[j + 1][0] and hits[j + 1][1] == hits[i][1]:
                j += 1
            runs.append((i, j, hits[i][1]))
            i = j + 1
        else:
            i += 1
    best = None
    for i, j, nb in runs:
        mid = angs[(i + j) // 2]
        d = np.array([math.cos(mid), math.sin(mid)])
        pts, _, _ = rebound_path(c, d, max(CFG['bounces'], 1), obj_speed(c, d, cue))
        blocked = path_blocked(c, d, pts, balls, cue) is not None
        width = math.degrees(angs[j] - angs[i]) + 0.25
        key = (blocked, nb, -width)
        if best is None or key < best[0]:
            best = (key, d, nb, width, blocked)
    return None if best is None else best[1:]


def start_speed(power):
    """Measured speed for 0-100% power; above 100% it keeps growing at the same rate
    (for when the app's power has to be set higher than the game's to match it)."""
    if power <= 1.0:
        return interp(POWER_V0, power)
    return POWER_V0[-1][1] + (power - 1.0) * (POWER_V0[-1][1] - POWER_V0[-2][1]) / 0.1


def interp(table, x):
    xs, ys = zip(*table)
    return float(np.interp(x, xs, ys))


def grid_rolling(table, v, a):
    rows = [float(np.interp(a, RIN_A, row)) for row in table]
    return float(np.interp(v, RIN_V, rows))


def grid(table, v, a):
    """Bilinear lookup in a measured cushion table (speed rows x angle columns);
    speeds above the last row scale with speed."""
    scale = 1.0
    if v > CUSH_V[-1]:
        scale, v = v / CUSH_V[-1], CUSH_V[-1]
    rows = [float(np.interp(a, CUSH_A, row)) for row in table]
    return float(np.interp(v, CUSH_V, rows)) * (scale if table is not CUSH_SHIFT else 1.0)


class Roll:
    """Speed of a ball, in game units: slides then rolls (measured decelerations)."""

    def __init__(self, v):
        self.v, self.rolling_at, self.free = v, ROLL_FRAC * v, 0.0

    def go(self, dist):
        """Roll `dist` game units; returns how far it actually gets."""
        done = 0.0
        if self.free > 0:                            # still skidding off the last cushion
            f = min(self.free, dist)
            self.free -= f
            done, dist = f, dist - f
            if dist <= 0:
                return done
        if self.v > self.rolling_at:                 # sliding part
            ds = (self.v ** 2 - self.rolling_at ** 2) / (2 * SLIDE_DECEL * tune('slow'))
            if dist <= ds:
                self.v = math.sqrt(max(self.v ** 2 - 2 * SLIDE_DECEL * tune('slow') * dist, 0))
                return dist
            self.v, done, dist = self.rolling_at, ds, dist - ds
        reach = self.v ** 2 / (2 * ROLL_DECEL * tune('slow'))
        if dist >= reach:
            self.v = 0.0
            return done + reach
        self.v = math.sqrt(self.v ** 2 - 2 * ROLL_DECEL * tune('slow') * dist)
        return done + dist

    def rolling(self):
        return self.free <= 0 and self.v <= self.rolling_at + 1e-6

    def bounced(self, ang_deg):
        """After a cushion (measured): it skids a while, then rolls at a known speed.
        Returns the bounce ratio k for tan(out) = k tan(in)."""
        v = self.v
        if self.rolling():
            k = grid_rolling(RIN_K, v, ang_deg)
            self.v = self.rolling_at = v * grid_rolling(RIN_KEEP, v, ang_deg)
            self.free = grid_rolling(RIN_SLIDE, v, ang_deg)
        else:
            k = interp(BOUNCE_K, v)
            self.v = self.rolling_at = grid(CUSH_ROLL, v, ang_deg)
            self.free = grid(CUSH_SLIDE, v, ang_deg)
        return k


def ball_colour(pt, r):
    """Typical colour (Lab) of the ball at pt - used to follow it during a shot."""
    import cv2
    R = int(r)
    img = np.asarray(grabber().grab({'left': int(pt[0]) - R, 'top': int(pt[1]) - R,
                                     'width': 2 * R + 1, 'height': 2 * R + 1}))[..., :3]
    lab = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    disc = (xx ** 2 + yy ** 2 <= (0.6 * r) ** 2).ravel()
    pix = lab[disc]
    # the ball's own colour: the most saturated half of the disc (skips number circle / shine)
    chroma = np.hypot(pix[:, 1] - 128, pix[:, 2] - 128)
    keep = pix[chroma >= np.median(chroma)]
    return np.median(keep if len(keep) else pix, axis=0)


def track_ball(start, colour, r, wait=20.0, run=9.0):
    """Follow one ball by its colour while it moves. Returns its positions (screen px)."""
    import cv2
    x0, y0, x1, y1 = [int(v) for v in CFG['felt']]
    pad = int(2 * r)
    box = {'left': x0 - pad, 'top': y0 - pad, 'width': x1 - x0 + 2 * pad, 'height': y1 - y0 + 2 * pad}
    last = np.array(start, float)
    pts, moving_since, still, t0 = [last.copy()], None, 0, time.time()
    LIVE['track_end'] = 'none'
    area_lo, area_hi = 0.15 * math.pi * r * r, 2.2 * math.pi * r * r
    while time.time() - t0 < wait + run:
        if not LIVE.get('learning'):
            return pts
        img = np.ascontiguousarray(np.asarray(grabber().grab(box))[..., :3])
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.int16)
        m = (np.abs(lab - colour.astype(np.int16)).sum(axis=2) < 32).astype(np.uint8)
        solid, _ = solid_mask(img, r)
        m &= cv2.dilate(solid, np.ones((3, 3), np.uint8))   # balls only: never the cue stick
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, _, st, ce = cv2.connectedComponentsWithStats(m)
        best = None
        for i in range(1, n):
            if not area_lo < st[i, 4] < area_hi:
                continue
            p = np.array([ce[i][0] + box['left'], ce[i][1] + box['top']])
            dist = np.linalg.norm(p - last)
            if dist < 12 * r and (best is None or dist < best[0]):
                best = (dist, p)
        if best is None:
            if moving_since is not None:
                LIVE['track_end'] = 'lost'             # gone: potted (or hidden)
                break
            continue
        p = best[1]
        if moving_since is None:
            if np.linalg.norm(p - pts[0]) > 0.6 * r:
                moving_since = time.time()
                LIVE['learn_msg'] = 'recording the shot...'
            elif time.time() - t0 > wait:
                break
            else:
                continue
        pts.append(p)
        still = still + 1 if np.linalg.norm(p - last) < 0.6 else 0
        last = p
        if still > 8:
            LIVE['track_end'] = 'still'
            break
        if time.time() - moving_since > run:
            LIVE['track_end'] = 'timeout'
            break
    return pts


def learn_from_track(track, predicted):
    """Compare real bounces with the ones the line predicted; returns (new bounce
    tuning, message)."""
    r = CFG['r']
    x0, y0, x1, y1 = CFG['felt']
    ix0, iy0, ix1, iy1 = x0 + r, y0 + r, x1 - r, y1 - r
    P = np.array(track, float)
    if len(P) < 6:
        return None, "didn't see the ball move - try again"
    bounces = []
    for k in range(2, len(P) - 2):
        for axis, lo, hi in ((0, ix0, ix1), (1, iy0, iy1)):
            near = min(abs(P[k][axis] - lo), abs(P[k][axis] - hi)) < 1.3 * r
            before, after = P[k][axis] - P[k - 2][axis], P[k + 2][axis] - P[k][axis]
            if near and before * after < 0 and (not bounces or k - bounces[-1][0] > 3):
                bounces.append((k, axis))
    if not bounces:
        return None, 'no cushion hit seen - learning needs a bank shot'
    ratios = []
    for (k, axis), pred in zip(bounces, predicted):
        if pred['axis'] != axis:
            break                                      # the shot went a different way: stop
        inn = P[k] - P[max(0, k - 4)]
        j = k + 1                                       # settled outgoing direction
        while j < len(P) - 1 and np.linalg.norm(P[j] - P[k]) < 5 * r:
            j += 1
        out = P[j] - P[k]
        if np.linalg.norm(inn) < r or np.linalg.norm(out) < 2 * r:
            continue
        def tan_to_cushion(v):
            return abs(v[axis]) / max(abs(v[1 - axis]), 1e-6)
        k_obs = tan_to_cushion(out) / max(tan_to_cushion(inn), 1e-6)
        if 0.2 < k_obs < 1.5:
            ratios.append(k_obs / pred['k'])
    if not ratios:
        return None, "couldn't read the bounces clearly - try a cleaner bank"
    f = float(np.median(ratios))
    old = tune('bounce')
    new = float(np.clip(old * (1 + 0.6 * (f - 1)), 0.7, 1.3))   # move most of the way, not all
    word = 'steeper' if f > 1 else 'flatter'
    return new, (f"real rebounds came off {abs(f - 1) * 100:.0f}% {word} than the line "
                 f"({len(ratios)} cushion{'s' if len(ratios) > 1 else ''}) - rebound angle "
                 f"{old * 100:.0f}% -> {new * 100:.0f}%")


def learn_corner(track, potted):
    """Corner angle from a real shot: a ball that dropped in a corner at angle A means the
    limit is at least A; a ball that reached a corner mouth and stayed out means A is too
    flat. Returns (new limit or None, message)."""
    P = np.array(track, float)
    if len(P) < 4:
        return None, ''
    r = CFG['r']
    pk = pockets()
    best = None                                   # closest approach to a corner aim point
    for i in (0, 2, 3, 5):
        A = np.array(pk[i])
        dists = np.linalg.norm(P - A, axis=1)
        k = int(np.argmin(dists))
        if best is None or dists[k] < best[0]:
            best = (dists[k], i, k)
    dist, i, k = best
    if dist > max(pocket_mouth(i), 2.5 * r) or k < 2:
        return None, ''                           # never got to a corner
    j = k
    while j > 0 and np.linalg.norm(P[k] - P[j]) < 3 * r:
        j -= 1
    v = P[k] - P[j]
    if np.linalg.norm(v) < r:
        return None, ''
    ax = np.array(POCKET_AXIS[i], float)
    ax /= np.linalg.norm(ax)
    ang = math.degrees(math.acos(max(-1.0, min(1.0, float(v @ ax) / np.linalg.norm(v)))))
    old = tune('corner')
    ended_there = k >= len(P) - 3
    if potted and ended_there and ang > old:
        new = min(70.0, ang + 2)
        return new, f'dropped in a corner at {ang:.0f}° -> corner angle {old:.0f}°->{new:.0f}°'
    if not potted and not ended_there and ang < old and ang >= 20:
        new = max(20.0, ang - 2)
        return new, f'rattled out of a corner at {ang:.0f}° -> corner angle {old:.0f}°->{new:.0f}°'
    return None, ''


# ---------------------------------------------------------------- auto-tune
class Mouse:
    """Moves the real mouse (X11 XTest) to aim and shoot in the game."""

    def __init__(self):
        from Xlib import X, display
        from Xlib.ext import xtest
        self.X, self.xtest = X, xtest
        self.d = display.Display()
        self.root = self.d.screen().root

    def pos(self):
        q = self.root.query_pointer()
        return np.array([q.root_x, q.root_y], float)

    def move(self, x, y):
        self.xtest.fake_input(self.d, self.X.MotionNotify, x=int(round(x)), y=int(round(y)))
        self.d.sync()

    def button(self, down):
        self.xtest.fake_input(self.d, self.X.ButtonPress if down else self.X.ButtonRelease, 1)
        self.d.sync()

    def drag(self, a, b, steps=12, hold=0.08):
        self.move(*a)
        time.sleep(0.05)
        self.button(True)
        for k in range(1, steps + 1):
            self.move(a[0] + (b[0] - a[0]) * k / steps, a[1] + (b[1] - a[1]) * k / steps)
            time.sleep(0.02)
        time.sleep(hold)
        self.button(False)


def slide_aim(mouse, cue, g, swing=25.0, steps=40, secs=0.8):
    """Aim like a person: press on the table a little off the shot line and slide the
    mouse round in an arc onto it, ending as far out along the line as the table allows."""
    cue = np.array(cue, float)
    u = (np.array(g, float) - cue)
    u /= np.linalg.norm(u)
    x0, y0, x1, y1 = CFG['felt']
    m = 2 * CFG['r']

    def reach(dirv):                     # how far from the cue ball until near the felt edge
        ts = []
        for k, lo, hi in ((0, x0 + m, x1 - m), (1, y0 + m, y1 - m)):
            if dirv[k] > 1e-9:
                ts.append((hi - cue[k]) / dirv[k])
            elif dirv[k] < -1e-9:
                ts.append((lo - cue[k]) / dirv[k])
        return max(min(ts), 0) if ts else 0

    end_d = max(reach(u), np.linalg.norm(np.array(g) - cue))
    a_end = math.atan2(u[1], u[0])
    a0 = a_end - math.radians(swing)
    start_d = min(end_d, reach(np.array([math.cos(a0), math.sin(a0)]))) * 0.8
    start = cue + np.array([math.cos(a0), math.sin(a0)]) * max(start_d, 3 * CFG['r'])
    mouse.move(*start)
    time.sleep(0.08)
    mouse.button(True)
    for k in range(1, steps + 1):
        t = k / steps
        t = t * t * (3 - 2 * t)                          # ease in / out
        a = a0 + (a_end - a0) * t
        dd = start_d + (end_d - start_d) * t
        mouse.move(*(cue + np.array([math.cos(a), math.sin(a)]) * max(dd, 3 * CFG['r'])))
        time.sleep(secs / steps)
    time.sleep(0.15)
    mouse.button(False)


def find_power_bar_on_screen():
    """Locate the game's power bar (tall bar beside the table)."""
    import cv2
    sys.path.insert(0, HERE)
    import poolsight as PS
    g = grabber()
    mon = g.monitors[0]
    img = np.ascontiguousarray(np.asarray(g.grab(mon))[..., :3])
    t = PS.find_table(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    if t is None:
        return None
    pb = PS.find_power_bar(img, (t.bx0, t.by0, t.bx1, t.by1))
    return None if pb is None else [int(v) for v in pb]


def table_still(timeout=12.0):
    """Wait until nothing on the felt moves any more."""
    import cv2
    x0, y0, x1, y1 = [int(v) for v in CFG['felt']]
    box = {'left': x0, 'top': y0, 'width': x1 - x0, 'height': y1 - y0}
    prev, calm, t0 = None, 0, time.time()
    while time.time() - t0 < timeout:
        img = cv2.cvtColor(np.ascontiguousarray(np.asarray(grabber().grab(box))[..., :3]),
                           cv2.COLOR_BGR2GRAY)[::3, ::3].astype(np.int16)
        if prev is not None:
            calm = calm + 1 if np.abs(img - prev).mean() < 0.6 else 0
            if calm >= 3:
                return True
        prev = img
        time.sleep(0.25)
    return False


def plan_test_shot(cue, balls):
    """A clean calibration bank: nearly full hit (no throw), target ball into a cushion at
    25-70 deg, a long run after the cushion, nothing in the way, away from pockets."""
    r = CFG['r']
    x0, y0, x1, y1 = CFG['felt']
    cue = np.array(cue, float)
    targets = sorted([b for b in balls if np.linalg.norm(b - cue) > 4 * r],
                     key=lambda b: np.linalg.norm(b - cue))[:6]
    best = None
    for T in targets:
        others = [b for b in balls if np.linalg.norm(b - T) > 0.8 * r and np.linalg.norm(b - cue) > 0.8 * r]
        for a in np.radians(np.arange(0, 360, 2.0)):
            d = np.array([math.cos(a), math.sin(a)])
            g = T - d * 2 * r
            if not (x0 + r < g[0] < x1 - r and y0 + r < g[1] < y1 - r):
                continue
            u = g - cue
            L = np.linalg.norm(u)
            if L < 4 * r:
                continue
            cut = math.degrees(math.acos(max(-1.0, min(1.0, float((u / L) @ d)))))
            if cut > 20:
                continue
            if first_block([cue, g], others, [cue, T], r) is not None:
                continue
            rec = []
            pts, pot, stop = rebound_path(T, d, 2, obj_speed(T, d, cue), rec)
            if not rec or len(pts) < 3 or pot is not None:
                continue
            ang = rec[0]['ang']
            if not 25 <= ang <= 70:
                continue
            out_len = np.linalg.norm(pts[2] - pts[1])
            if out_len < 8 * r:
                continue
            if min(np.linalg.norm(pts[1] - np.array(q)) for q in pockets()) < 5 * r:
                continue
            if path_blocked(T, d, pts[:3], others, cue) is not None:
                continue
            score = abs(ang - 45) + cut * 1.5 - out_len / r * 0.5
            if best is None or score < best[0]:
                best = (score, T, d, g)
    return None if best is None else best[1:]


def rebound_path(start, d, bounces, speed=None, record=None):
    """Line + cushion rebounds inside the felt; stops in a pocket, or where it runs out
    of speed when `speed` (game units/s) is given. Returns (points, pot, stopped)."""
    r = CFG['r']
    unit = (CFG['felt'][2] - CFG['felt'][0]) / 980.0        # screen px per game unit
    ball = Roll(speed) if speed else None
    x0, y0, x1, y1 = CFG['felt']
    ix0, iy0, ix1, iy1 = x0 + r, y0 + r, x1 - r, y1 - r      # where a ball centre can go
    pt = np.array(start, float)
    pts, pot = [pt], None
    for i in range(bounces + 1):
        ts = []
        if d[0] > 1e-9: ts.append(((ix1 - pt[0]) / d[0], 0))
        if d[0] < -1e-9: ts.append(((ix0 - pt[0]) / d[0], 0))
        if d[1] > 1e-9: ts.append(((iy1 - pt[1]) / d[1], 1))
        if d[1] < -1e-9: ts.append(((iy0 - pt[1]) / d[1], 1))
        if not ts:
            break
        tt, axis = min(ts)
        hit = pocket_hit(pt, d, max(tt, 0))
        if ball is not None:
            need = (hit[0] if hit is not None else max(tt, 0)) / unit
            got = ball.go(need)
            if got < need - 1e-6:                    # runs out of speed first
                pts.append(pt + d * got * unit)
                return pts, None, True
        if hit is not None:
            pot = hit[2]
            pts.append(pot)
            break
        pt = pt + d * max(tt, 0)
        pts.append(pt)
        d = d.copy()
        sin_in = abs(d[axis])
        ang = math.degrees(math.asin(min(1.0, sin_in)))      # angle to the cushion
        if not CFG['real']:
            k = 1.0
        elif ball is not None:
            shift = 0.0 if ball.rolling() else grid(CUSH_SHIFT, ball.v, ang) * unit
            k = ball.bounced(ang)
            tang = 1 - axis                                    # the along-the-cushion axis
            pt = pt.copy()
            pt[tang] -= math.copysign(shift, d[tang]) if abs(d[tang]) > 1e-9 else 0
        else:
            k = E_CUSHION
        if CFG['real']:
            k = min(0.98, k * tune('bounce'))
        if record is not None:
            record.append({'axis': axis, 'ang': ang, 'k': k, 'at': pts[-1].copy()})
        d[axis] *= -k
        d /= np.linalg.norm(d)
    return pts, pot, False


# ---------------------------------------------------------------- overlay
class Overlay(QtWidgets.QWidget):
    key = QtCore.pyqtSignal(int)

    def __init__(self, geo):
        super().__init__(None, QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint |
                         QtCore.Qt.Tool | QtCore.Qt.X11BypassWindowManagerHint |
                         QtCore.Qt.WindowTransparentForInput)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating)
        self.setGeometry(geo)
        self.key.connect(self.on_key)
        self.timer = QtCore.QTimer(self, timeout=self.tick)
        self.timer.start(16)                    # ~60 fps while the line follows the mouse

    def mouse(self):
        pos = QtGui.QCursor.pos()
        dpr = self.devicePixelRatioF()
        return np.array([pos.x() * dpr, pos.y() * dpr])

    def tick(self):
        now = time.time()
        if not getattr(self, 'scanner', None):     # one long-lived background scanner
            self.scanner = threading.Thread(target=self._scan_loop, daemon=True)
            self.scanner.start()
        if CFG['on'] and CFG['point']:
            self.update()                         # follows the mouse for the highlight
        if CFG['on'] and LIVE['moving'] and CFG['point']:
            v = self.mouse() - np.array(CFG['point'])
            if np.linalg.norm(v) > 3:
                LIVE['dir'] = v / np.linalg.norm(v)
            self.update()

    def _scan_loop(self):
        while True:
            if CFG['on'] and CFG['point'] and CFG['felt'] and not LIVE['moving']:
                self._cue_bg()
                time.sleep(0.5)
            else:
                time.sleep(0.2)

    def _cue_bg(self):
        LIVE['busy'] = True
        try:
            c = find_cue(CFG['felt'], CFG['r'])
            if c is not None:
                LIVE['cue'] = c
            balls = detect_balls(CFG['felt'], CFG['r'])
            LIVE['balls'] = balls
            # the locked point is never moved here: it only changes when the player presses 1
        except Exception:
            log.exception('find cue/balls failed')
        finally:
            LIVE['busy'] = False

    def autotune(self, shots=6):
        if LIVE.get('auto'):
            LIVE['auto'] = False
            LIVE['learn_msg'] = 'auto-tune: stopping...'
            return
        if not CFG['felt']:
            LIVE['learn_msg'] = 'auto-tune: mark the table first'
            return
        LIVE['auto'] = True
        threading.Thread(target=self._autotune, args=(shots,), daemon=True).start()

    def _autotune(self, shots):
        was_on = CFG['on']
        done, results = 0, []
        try:
            mouse = Mouse()
            pb = CFG.get('pbar') or find_power_bar_on_screen()
            if not pb:
                LIVE['learn_msg'] = "auto-tune: can't find the game's power bar"
                return
            CFG['pbar'] = pb
            CFG['on'] = False                       # keep our lines out of the camera
            QtCore.QMetaObject.invokeMethod(self, 'update', QtCore.Qt.QueuedConnection)
            powers = [0.55, 0.75, 0.45, 0.85, 0.6, 0.7, 0.5, 0.8]
            while LIVE.get('auto') and done < shots:
                LIVE['learn_msg'] = f'auto-tune: waiting for the table to settle ({done}/{shots})'
                table_still()
                cue = find_cue(CFG['felt'], CFG['r'])
                balls = detect_balls(CFG['felt'], CFG['r'])
                if cue is None:
                    LIVE['learn_msg'] = "auto-tune: can't see the cue ball - stopping"
                    break
                CFG['power'] = powers[done % len(powers)]
                plan = plan_test_shot(cue, balls)
                if plan is None:
                    LIVE['learn_msg'] = 'auto-tune: no clean test shot here - move some balls and run again'
                    break
                T, d, g = plan
                rec = []
                rebound_path(T, d, 2, obj_speed(T, d, cue), rec)
                full, _, stops = rebound_path(T, d, 6, obj_speed(T, d, cue))
                pred_len = sum(np.linalg.norm(b - a) for a, b in zip(full, full[1:]))
                colour = ball_colour(T, CFG['r'])
                LIVE['learn_msg'] = f'auto-tune: shot {done + 1}/{shots} at {int(CFG["power"] * 100)}% power'
                # aim by sliding: press beside the shot line, sweep round to it, and finish far
                # out along the line (the game aims at the mouse; far = finer angle)
                slide_aim(mouse, cue, g)
                time.sleep(0.35)
                # follow the target ball while the shot happens
                box = {}
                LIVE['learning'] = True
                th = threading.Thread(target=lambda: box.setdefault('track', track_ball(T, colour, CFG['r'], wait=6, run=10)))
                th.start()
                bx = (pb[0] + pb[2]) / 2
                top, bot = pb[1] + 15, pb[3] - 5
                mouse.drag((bx, top + 5), (bx, top + 5 + CFG['power'] * (bot - top - 5)), steps=10, hold=0.12)
                rest = mouse.pos()
                while th.is_alive():
                    th.join(0.2)
                    if np.linalg.norm(mouse.pos() - rest) > 40:      # the player took over
                        LIVE['auto'] = False
                        LIVE['learning'] = False
                LIVE['learning'] = False
                if not LIVE.get('auto'):
                    break
                track = box.get('track') or []
                done += 1
                # 1) cushions
                new, msg = learn_from_track(track, rec)
                if new is not None:
                    CFG['tune']['bounce'] = new
                cnew, cmsg = learn_corner(track, LIVE.get('track_end') == 'lost')
                if cnew is not None:
                    CFG['tune']['corner'] = cnew
                    msg += '; ' + cmsg
                # 2) how far it rolls (only when it stopped on the table and we expected a stop)
                smsg = ''
                if LIVE.get('track_end') == 'still' and stops and len(track) > 5:
                    real_len = sum(np.linalg.norm(np.array(b) - np.array(a)) for a, b in zip(track, track[1:]))
                    if real_len > 6 * CFG['r'] and pred_len > 6 * CFG['r']:
                        f = pred_len / real_len                 # >1: it rolled less far -> more slowdown
                        old = tune('slow')
                        CFG['tune']['slow'] = float(np.clip(old * (1 + 0.6 * (f - 1)), 0.2, 1.6))
                        smsg = f"; rolled {real_len / pred_len * 100:.0f}% of predicted -> slowdown {old * 100:.0f}%->{CFG['tune']['slow'] * 100:.0f}%"
                save()
                results.append(msg + smsg)
                log.info('autotune shot %d: %s%s', done, msg, smsg)
                LIVE['learn_new'] = True
                LIVE['learn_msg'] = f'auto-tune {done}/{shots}: ' + msg + smsg
            if done:
                LIVE['learn_msg'] = (f"auto-tune done ({done} shots): rebound angle {tune('bounce') * 100:.0f}%, "
                                     f"slowdown {tune('slow') * 100:.0f}%, corner angle {tune('corner'):.0f}°")
        except Exception:
            log.exception('autotune failed')
            LIVE['learn_msg'] = 'auto-tune failed (see guide.log)'
        finally:
            LIVE['auto'] = False
            LIVE['learning'] = False
            CFG['on'] = was_on
            LIVE['learn_new'] = True
            QtCore.QMetaObject.invokeMethod(self, 'update', QtCore.Qt.QueuedConnection)

    def learn(self):
        """Key 3: watch the next shot of the locked ball and tune the cushions from it."""
        if LIVE.get('learning'):
            LIVE['learning'] = False
            LIVE['learn_msg'] = 'learning cancelled'
            return
        if not CFG['point'] or LIVE.get('dir') is None:
            LIVE['learn_msg'] = 'learning: lock a ball (1) and set the line (2) first'
            return
        c = np.array(CFG['point'], float)
        rec = []
        rebound_path(c, LIVE['dir'], max(CFG['bounces'], 1), obj_speed(c, LIVE['dir'], LIVE.get('cue')), rec)
        try:
            colour = ball_colour(c, CFG['r'])
        except Exception:
            log.exception('colour failed')
            LIVE['learn_msg'] = "learning: couldn't read the ball colour"
            return
        LIVE['learning'] = True
        LIVE['learn_msg'] = 'learning: take the shot now (3 again = cancel)'

        def work():
            try:
                track = track_ball(c, colour, CFG['r'])
                if not LIVE.get('learning'):
                    return
                new, msg = learn_from_track(track, rec)
                if new is not None:
                    CFG.setdefault('tune', dict(TUNE_DEFAULT))['bounce'] = new
                cnew, cmsg = learn_corner(track, LIVE.get('track_end') == 'lost')
                if cnew is not None:
                    CFG.setdefault('tune', dict(TUNE_DEFAULT))['corner'] = cnew
                save()
                if new is None and cnew is not None:
                    msg, new = cmsg, cnew
                elif cmsg:
                    msg += '; ' + cmsg
                LIVE['learn_msg'] = 'learned: ' + msg if new is not None else 'learning: ' + msg
                LIVE['learn_new'] = True
                log.info('learn: %s', msg)
            except Exception:
                log.exception('learning failed')
                LIVE['learn_msg'] = 'learning failed (see guide.log)'
            finally:
                LIVE['learning'] = False

        threading.Thread(target=work, daemon=True).start()

    def on_key(self, k):
        if k == 30:
            if LIVE.get('auto'):
                self.autotune()                       # 3 also stops auto-tune
            else:
                self.learn()
            self.update()
            return
        if k == 1:                                   # set the meeting point (snapped to the ball)
            m = self.mouse()
            snap = None
            try:
                # nearest ball from the latest scan (forgiving), then its exact centre
                balls = LIVE.get('balls') or detect_balls(CFG['felt'], CFG['r'])
                cand = [b for b in balls if np.linalg.norm(b - m) < 4 * CFG['r']]
                start = min(cand, key=lambda b: np.linalg.norm(b - m)) if cand else m
                snap = snap_to_ball(start[0], start[1], CFG['r'])
                if snap is None and cand:
                    snap = [float(start[0]), float(start[1])]
            except Exception:
                log.exception('snap failed')
            CFG['point'] = snap if snap else m.tolist()
            LIVE['locked'] = bool(snap)
            LIVE['lost'] = 0
            LIVE['moving'], LIVE['dir'], LIVE['tuned'] = False, None, False
            save()
        elif k == 2 and CFG['point'] and LIVE.pop('tuned', False):
            LIVE['moving'] = False                   # after Up/Down, 2 locks the tuned line
            LIVE['shot'] = 'locked'                  # (press 2 again to follow the mouse)
        elif k == 2 and CFG['point']:
            m = self.mouse()
            pk = pockets()
            near = [i for i in range(6) if math.hypot(m[0] - pk[i][0], m[1] - pk[i][1]) < 3.5 * CFG['r']]
            if near and not LIVE['moving']:
                # pocket way: calculate the line from the ball into that pocket
                tgt = min(near, key=lambda i: math.hypot(m[0] - pk[i][0], m[1] - pk[i][1]))
                c = np.array(CFG['point'], float)
                if not LIVE.get('balls'):
                    try:
                        LIVE['balls'] = detect_balls(CFG['felt'], CFG['r'])
                    except Exception:
                        log.exception('balls failed')
                sol = solve_to_pocket(c, tgt, LIVE.get('balls') or [], LIVE.get('cue'))
                if sol:
                    d, nb, width, blocked = sol
                    LIVE['dir'] = d
                    LIVE['shot'] = (('direct' if nb == 0 else f'{nb} bank' + ('s' if nb > 1 else '')) +
                                    f'  ±{width / 2:.1f}°' + ('  BLOCKED' if blocked else ''))
                else:
                    LIVE['shot'] = 'no way into that pocket'
                LIVE['moving'] = False
            else:
                # original way: the line follows the mouse; press 2 again to freeze
                LIVE['moving'] = not LIVE['moving']
                LIVE['shot'] = None
        elif k in (3, 4, 13, 14) and LIVE['dir'] is not None:
            LIVE['moving'] = False                   # fine turn: 0.1 deg (Shift: 1 deg)
            LIVE['tuned'] = True
            LIVE['shot'] = None
            step = math.radians(0.1 if k < 10 else 1.0) * (1 if k in (3, 13) else -1)
            c, s = math.cos(step), math.sin(step)
            dx, dy = LIVE['dir']
            LIVE['dir'] = np.array([c * dx - s * dy, s * dx + c * dy])
        self.update()

    def paintEvent(self, _):
        if not CFG['on'] or not CFG['felt']:
            return
        p = QtGui.QPainter(self)
        try:
            sc = 1 / self.devicePixelRatioF()
            p.scale(sc, sc)
            p.setRenderHint(QtGui.QPainter.Antialiasing, True)
            Q = lambda q: QtCore.QPointF(q[0], q[1])
            r, k = CFG['r'], CFG['thick']
            x0, y0, x1, y1 = CFG['felt']
            # faint table outline + pockets, to check the setup lines up
            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 70), 1, QtCore.Qt.DotLine))
            p.drawRect(QtCore.QRectF(x0, y0, x1 - x0, y1 - y0))
            if LIVE.get('learn_msg'):
                p.setPen(QtGui.QColor(255, 225, 0))
                p.setFont(QtGui.QFont('Sans', 12, QtGui.QFont.Bold))
                p.drawText(Q((x0 + 10, y0 - 12)), LIVE['learn_msg'])
            if not CFG['point']:
                return
            c = np.array(CFG['point'], float)
            p.setPen(QtGui.QPen(COL_POCKET, k))
            ghosts = []
            for pk in pockets():
                p.drawLine(Q(pk), Q(c))
                u = c - np.array(pk)
                if np.linalg.norm(u) > 1:          # ghost ball: where to hit for this pocket
                    ghosts.append(c + u / np.linalg.norm(u) * 2 * r)
            # alignment helper: the ghost nearest the mouse, with a line from the cue ball.
            # Line the game's white aim line up on it and its ring inside the yellow circle.
            cue = LIVE.get('cue')
            balls = LIVE.get('balls') or []
            red = QtGui.QColor(255, 50, 50)
            if ghosts:
                m = self.mouse()
                if LIVE['dir'] is not None:
                    g = c - LIVE['dir'] * 2 * r     # hit here to send the ball along the line
                else:
                    g = min(ghosts, key=lambda q: np.linalg.norm(q - m))
                yel = QtGui.QColor(255, 225, 0)
                if cue is not None and np.linalg.norm(np.array(cue) - g) > 2 * r:
                    cu = np.array(cue)
                    v = (g - cu) / np.linalg.norm(g - cu)
                    # is the cue ball's corridor to the ghost clear?
                    blk = first_block([cu, g], balls, [cu, c], r)
                    if blk:
                        _, touch, q = blk
                        p.setPen(QtGui.QPen(yel, max(2, k // 2), QtCore.Qt.DashLine))
                        p.drawLine(Q(cu + v * r), Q(touch))
                        p.setPen(QtGui.QPen(red, max(2, k // 2), QtCore.Qt.DashLine))
                        p.drawLine(Q(touch), Q(g - v * r))
                        p.setPen(QtGui.QPen(red, 3))
                        p.drawEllipse(Q(q), 1.3 * r, 1.3 * r)
                    else:
                        p.setPen(QtGui.QPen(yel, max(2, k // 2), QtCore.Qt.DashLine))
                        p.drawLine(Q(cu + v * r), Q(g - v * r))
                p.setPen(QtGui.QPen(yel, max(3, k)))
                p.drawEllipse(Q(g), r, r)
                p.setPen(QtGui.QPen(yel, 2))
                p.drawLine(Q(g + np.array([-4, 0])), Q(g + np.array([4, 0])))
                p.drawLine(Q(g + np.array([0, -4])), Q(g + np.array([0, 4])))
            lock_col = QtGui.QColor(41, 254, 81) if LIVE.get('locked') else QtGui.QColor(255, 255, 255)
            p.setPen(QtGui.QPen(lock_col, max(2, k // 2)))
            p.drawEllipse(Q(c), r, r)
            if LIVE['dir'] is not None:
                pts, pot, stopped = rebound_path(c, LIVE['dir'], max(CFG['bounces'], 1),
                                                 obj_speed(c, LIVE['dir'], LIVE.get('cue')))
                blk = path_blocked(c, LIVE['dir'], pts, balls, LIVE.get('cue'))
                cut = len(pts) - 1 if blk is None else blk[0]
                p.setPen(QtGui.QPen(COL_LINE, k + 1))
                for i, (a, b) in enumerate(zip(pts, pts[1:])):
                    if i < cut:
                        p.drawLine(Q(a), Q(b))
                    elif i == cut and blk:
                        p.drawLine(Q(a), Q(blk[1]))
                        p.setPen(QtGui.QPen(red, k + 1))     # past the blocking ball: won't happen
                        p.drawLine(Q(blk[1]), Q(b))
                    else:
                        p.drawLine(Q(a), Q(b))
                if stopped and not blk:
                    p.setPen(QtGui.QPen(QtGui.QColor(255, 170, 0), 2, QtCore.Qt.DashLine))
                    p.drawEllipse(Q(pts[-1]), r, r)          # where it stops at this power
                    p.setFont(QtGui.QFont('Sans', 11, QtGui.QFont.Bold))
                    p.drawText(Q(pts[-1] + np.array([1.4 * r, 0])), 'too soft')
                if blk:
                    p.setPen(QtGui.QPen(red, 3))
                    p.drawEllipse(Q(blk[2]), 1.3 * r, 1.3 * r)
                    p.drawEllipse(Q(blk[1]), r, r)            # where it touches
                elif pot is not None:
                    p.setPen(QtGui.QPen(COL_POT, k + 2))
                    p.drawEllipse(Q(pot), 1.6 * r, 1.6 * r)
                if LIVE.get('shot'):
                    p.setPen(red if 'BLOCK' in LIVE['shot'] or 'no way' in LIVE['shot'] else COL_POT)
                    p.setFont(QtGui.QFont('Sans', 12, QtGui.QFont.Bold))
                    p.drawText(Q(c + np.array([1.6 * r, -1.6 * r])), LIVE['shot'])
        except Exception:
            log.exception('paint failed')
        finally:
            p.end()


# ---------------------------------------------------------------- mark the table
class FeltSelector(QtWidgets.QWidget):
    chosen = QtCore.pyqtSignal(object)

    def __init__(self, geo):
        super().__init__(None, QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint |
                         QtCore.Qt.X11BypassWindowManagerHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setGeometry(geo)
        self.setCursor(QtCore.Qt.CrossCursor)
        self.a = self.b = None

    def showEvent(self, e):
        self.activateWindow()
        self.setFocus()

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Escape:
            self.chosen.emit(None)
            self.close()

    def mousePressEvent(self, e):
        self.a = self.b = e.pos()
        self.update()

    def mouseMoveEvent(self, e):
        self.b = e.pos()
        self.update()

    def mouseReleaseEvent(self, e):
        r = QtCore.QRect(self.a, e.pos()).normalized()
        dpr = self.devicePixelRatioF()
        if r.width() > 100 and r.height() > 60:
            self.chosen.emit([r.left() * dpr, r.top() * dpr, r.right() * dpr, r.bottom() * dpr])
            self.close()
        else:
            self.a = self.b = None
            self.update()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 110))
        p.setPen(QtGui.QColor(255, 255, 255))
        p.setFont(QtGui.QFont('Sans', 16, QtGui.QFont.Bold))
        p.drawText(self.rect().adjusted(0, 40, 0, 0), QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop,
                   'Drag a box over the FELT: from cushion edge to cushion edge.   Esc = cancel')
        if self.a is not None:
            r = QtCore.QRect(self.a, self.b).normalized()
            p.setCompositionMode(QtGui.QPainter.CompositionMode_Clear)
            p.fillRect(r, QtCore.Qt.transparent)
            p.setCompositionMode(QtGui.QPainter.CompositionMode_SourceOver)
            p.setPen(QtGui.QPen(QtGui.QColor(255, 220, 0), 2))
            p.drawRect(r)
        p.end()


# ---------------------------------------------------------------- panel
class Panel(QtWidgets.QWidget):
    @QtCore.pyqtSlot()
    def toggle_slot(self):
        self.toggle()

    def __init__(self, ov, geo, app):
        super().__init__(None, QtCore.Qt.Window | QtCore.Qt.WindowStaysOnTopHint)
        self.ov, self.geo, self.app = ov, geo, app
        self.setWindowTitle('PoolSight Guide')
        self.setWindowIcon(QtGui.QIcon(os.path.join(HERE, 'icon.png')))
        self.setStyleSheet('''
            QWidget { background: #15181e; color: #e6e6e6; font-size: 12px; }
            QPushButton { background: #263040; border: 1px solid #3b4a60; border-radius: 5px; padding: 6px; }
            QPushButton:hover { background: #31405a; }
            QPushButton#big { font-size: 14px; font-weight: bold; padding: 8px; }
        ''')
        lay = QtWidgets.QVBoxLayout(self)
        self.onoff = QtWidgets.QPushButton()
        self.onoff.setObjectName('big')
        self.onoff.clicked.connect(self.toggle)
        lay.addWidget(self.onoff)

        row = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton('Auto mark')
        b.setToolTip('Find the table by itself (game must be visible, nothing covering the felt)')
        b.clicked.connect(self.auto_mark)
        row.addWidget(b)
        b = QtWidgets.QPushButton('Manual mark…')
        b.setToolTip('Drag a box over the felt, from cushion edge to cushion edge')
        b.clicked.connect(self.mark)
        row.addWidget(b)
        lay.addLayout(row)
        self.msg = QtWidgets.QLabel('')
        self.msg.setWordWrap(True)
        self.msg.setStyleSheet('color: #ffd84d;')
        lay.addWidget(self.msg)

        def slider(text, lo, hi, val, cb):
            row = QtWidgets.QHBoxLayout()
            lab = QtWidgets.QLabel(text)
            lab.setFixedWidth(80)
            row.addWidget(lab)
            s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            s.setRange(lo, hi)
            s.setValue(int(val))
            s.valueChanged.connect(cb)
            row.addWidget(s)
            lay.addLayout(row)
            return s

        self.r_sl = slider('Ball size', 5, 40, CFG['r'], lambda v: self.set('r', float(v)))
        b = QtWidgets.QPushButton('Auto')
        b.setFixedWidth(46)
        b.setToolTip('Measure the balls on the table')
        b.clicked.connect(self.auto_ball)
        lay.itemAt(lay.count() - 1).layout().addWidget(b)
        slider('Thickness', 1, 10, CFG['thick'], lambda v: self.set('thick', v))
        slider('Bounces', 1, 6, CFG['bounces'], lambda v: self.set('bounces', v))
        self.pw_lab = QtWidgets.QLabel()
        slider('Power', 5, 200, int(CFG.get('power', 0.5) * 100),
               lambda v: (self.set('power', v / 100), self.pw_lab.setText(f'planned power {v}%')))
        self.pw_lab.setText(f"planned power {int(CFG.get('power', 0.5) * 100)}%  (for banks / 'too soft')")
        self.pw_lab.setStyleSheet('color: #8a93a3;')
        lay.addWidget(self.pw_lab)
        cb = QtWidgets.QCheckBox('Real cushions (as measured in your game)')
        cb.setChecked(CFG['real'])
        cb.toggled.connect(lambda v: self.set('real', v))
        lay.addWidget(cb)

        # ---- tuning: adjust by hand, or let key 3 learn it from a real shot
        tbox = QtWidgets.QGroupBox('Tuning')
        tbox.setStyleSheet('QGroupBox { border: 1px solid #333a45; border-radius: 6px; margin-top: 10px;'
                           ' padding: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 8px;'
                           ' color: #ffd84d; }')
        tl = QtWidgets.QVBoxLayout(tbox)
        self.tune_sl = {}

        def tslider(key, text, lo, hi, scale, tip):
            row = QtWidgets.QHBoxLayout()
            lab = QtWidgets.QLabel(text)
            lab.setFixedWidth(78)
            lab.setToolTip(tip)
            row.addWidget(lab)
            sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            sl.setRange(lo, hi)
            sl.setToolTip(tip)
            val = QtWidgets.QLabel()
            val.setFixedWidth(38)
            row.addWidget(sl)
            row.addWidget(val)

            def changed(v):
                CFG.setdefault('tune', dict(TUNE_DEFAULT))[key] = v / scale
                val.setText(f'{v}°' if scale == 1 else f'{v}%')
                save()
                self.ov.update()
            sl.valueChanged.connect(changed)
            sl.setValue(int(round(tune(key) * scale)))
            val.setText(f'{sl.value()}°' if scale == 1 else f'{sl.value()}%')
            tl.addLayout(row)
            self.tune_sl[key] = (sl, scale)

        tslider('bounce', 'Rebound angle', 70, 130, 100,
                'tan(out) = k x tan(in), angles measured from the cushion. Higher = comes off '
                'steeper (more straight out); lower = flatter (closer to the cushion)')
        self.angle_lab = QtWidgets.QLabel()
        self.angle_lab.setStyleSheet('color: #8a93a3;')
        tl.addWidget(self.angle_lab)
        self.tune_sl['bounce'][0].valueChanged.connect(lambda _: self.show_angles())
        self.show_angles()
        tslider('slow', 'Slowdown', 20, 160, 100,
                'Higher = balls stop sooner ("too soft" earlier); lower = they roll further')
        tslider('mouth', 'Pocket size', 50, 160, 100,
                'Higher = more shots count as going in; lower = stricter')
        tslider('corner', 'Corner angle', 25, 70, 1,
                'How flat a ball can come into a corner pocket and still drop')
        row = QtWidgets.QHBoxLayout()
        ab = QtWidgets.QPushButton('Auto-tune: take 6 test shots')
        ab.setToolTip('Moves your mouse and plays test shots in the game, then tunes Bounce and '
                      'Slowdown. Use a 2-player game. Stop: press 3 or F8, or move the mouse.')
        ab.clicked.connect(lambda: self.ov.autotune(6))
        tl.addWidget(ab)
        b = QtWidgets.QPushButton('Learn from shot (3)')
        b.setToolTip('Lock a ball (1), set a bank line (2), click this or press 3, then shoot')
        b.clicked.connect(lambda: self.ov.key.emit(30))
        row.addWidget(b)
        b = QtWidgets.QPushButton('Reset')
        b.setToolTip('Back to the values measured from test shots')
        b.clicked.connect(self.reset_tune)
        row.addWidget(b)
        tl.addLayout(row)
        lay.addWidget(tbox)

        # ---- key bindings: click a button, then press the key you want
        kbox = QtWidgets.QGroupBox('Keys  (click, then press a key)')
        kbox.setStyleSheet(tbox.styleSheet())
        kg = QtWidgets.QGridLayout(kbox)
        self.key_btn = {}
        for i, (name, label) in enumerate(KEY_NAMES.items()):
            kg.addWidget(QtWidgets.QLabel(label), i // 2, (i % 2) * 2)
            kb = QtWidgets.QPushButton()
            kb.setFixedWidth(52)
            kb.clicked.connect(lambda _, n=name: self.start_bind(n))
            kg.addWidget(kb, i // 2, (i % 2) * 2 + 1)
            self.key_btn[name] = kb
        rb = QtWidgets.QPushButton('Default keys')
        rb.clicked.connect(self.default_keys)
        kg.addWidget(rb, 4, 0, 1, 4)
        lay.addWidget(kbox)
        self.binding = None
        app.installEventFilter(self)        # catches mouse binds (see eventFilter)
        self.show_keys()

        help_ = QtWidgets.QLabel('Shift + turn keys = bigger steps')
        help_.setStyleSheet('color: #8a93a3;')
        lay.addWidget(help_)
        b = QtWidgets.QPushButton('Quit')
        b.clicked.connect(app.quit)
        lay.addWidget(b)
        self.setFixedWidth(260)
        self.ltimer = QtCore.QTimer(self, timeout=self.poll_learning)
        self.ltimer.start(300)
        self.refresh()
        self.adjustSize()
        pos = CFG.get('panel_pos')
        self.move(*(pos or (geo.right() - self.width() - 8, geo.top() + 32)))

    def moveEvent(self, e):
        CFG['panel_pos'] = [self.x(), self.y()]

    def closeEvent(self, e):
        save()
        self.app.quit()

    def show_angles(self):
        """Show what the rebound setting means in degrees (normal rolling ball)."""
        k = min(0.98, 0.643 * tune('bounce'))
        parts = []
        for a in (30, 45, 60):
            out = math.degrees(math.atan(k * math.tan(math.radians(a))))
            parts.append(f'{a}°→{out:.0f}°')
        self.angle_lab.setText('in → out:  ' + '   '.join(parts))

    def show_keys(self):
        keys = dict(KEYS_DEFAULT, **CFG.get('keys', {}))
        for n, b in self.key_btn.items():
            b.setText('…' if n == self.binding else key_label(keys[n]))

    def start_bind(self, name):
        self.binding = name
        LIVE['binding'] = True              # the global keys pause while you choose
        self.show_keys()
        self.msg.setText(f'press a key or mouse button / wheel for "{KEY_NAMES[name]}" (Esc = cancel)')
        self.activateWindow()
        self.setFocus()

    def keyPressEvent(self, e):
        if not self.binding:
            return super().keyPressEvent(e)
        name, self.binding = self.binding, None
        LIVE['binding'] = False
        if e.key() == QtCore.Qt.Key_Escape:
            self.msg.setText('key unchanged')
            self.show_keys()
            return
        key = qt_key_to_x(e)
        log.info('bind key: qt %#x text %r native sym %#x -> %r', e.key(), e.text(),
                 e.nativeVirtualKey(), key)
        self.set_bind(name, key)

    def eventFilter(self, obj, e):
        """While choosing a bind, mouse buttons / the wheel anywhere on our windows count too."""
        if self.binding and e.type() in (QtCore.QEvent.MouseButtonPress, QtCore.QEvent.Wheel):
            name, self.binding = self.binding, None
            LIVE['binding'] = False
            if e.type() == QtCore.QEvent.Wheel:
                d = e.angleDelta()
                btn = (4 if d.y() > 0 else 5) if abs(d.y()) >= abs(d.x()) else (6 if d.x() > 0 else 7)
                self.set_bind(name, f'Mouse{btn}')
            elif e.button() == QtCore.Qt.LeftButton:
                self.msg.setText("left click can't be used (the game needs it) - key unchanged")
                self.show_keys()
            else:
                # Qt numbers extra buttons as bit flags: Back=bit 3 is X button 8, Forward 9, ...
                b = int(e.button())
                btn = {2: 3, 4: 2}.get(b) or (b.bit_length() + 4 if b >= 8 and b & (b - 1) == 0 else 0)
                log.info('bind mouse: qt button %#x -> X button %s', b, btn)
                self.set_bind(name, f'Mouse{btn}' if btn else '')
            return True                       # swallow it so it doesn't also click/scroll
        return super().eventFilter(obj, e)

    def set_bind(self, name, key):
        if not key:
            self.msg.setText("that key can't be used - pick another")
        else:
            keys = dict(KEYS_DEFAULT, **CFG.get('keys', {}))
            clash = [n for n, k in keys.items() if k.lower() == key.lower() and n != name]
            for n in clash:                   # swap so no two actions share a key
                keys[n] = keys[name]
            keys[name] = key
            CFG['keys'] = keys
            save()
            self.msg.setText(f'"{KEY_NAMES[name]}" is now {key_label(key)}' +
                             (f' (swapped with "{KEY_NAMES[clash[0]]}")' if clash else ''))
        self.show_keys()

    def default_keys(self):
        CFG['keys'] = dict(KEYS_DEFAULT)
        save()
        self.show_keys()
        self.msg.setText('keys back to 1 / 2 / 3 / ↑ ↓ / F8 / F9')

    def poll_learning(self):
        msg = LIVE.get('learn_msg')
        if msg and msg != getattr(self, 'shown_msg', None):
            self.shown_msg = msg
            self.msg.setText(msg)
        if LIVE.pop('learn_new', False):                 # learning changed a value: show it
            for key, (sl, scale) in self.tune_sl.items():
                sl.blockSignals(True)
                sl.setValue(int(round(tune(key) * scale)))
                sl.blockSignals(False)
            self.show_angles()
            self.ov.update()

    def reset_tune(self):
        CFG['tune'] = dict(TUNE_DEFAULT)
        try:
            d = json.load(open(os.path.join(HERE, 'guide.defaults.json')))
            for k in ('power', 'bounces', 'thick', 'real'):
                if k in d:
                    CFG[k] = d[k]
        except Exception:
            pass
        for key, (sl, scale) in self.tune_sl.items():
            sl.setValue(int(round(TUNE_DEFAULT[key] * scale)))
        save()
        self.msg.setText('tuning reset to the measured values')

    def set(self, k, v):
        CFG[k] = v
        save()
        self.ov.update()

    def toggle(self):
        if LIVE.get('auto'):                         # F8 during auto-tune = stop it
            LIVE['auto'] = False
            LIVE['learning'] = False
            LIVE['learn_msg'] = 'auto-tune stopped'
            return
        CFG['on'] = not CFG['on']
        save()
        self.refresh()
        self.ov.update()

    def refresh(self):
        self.onoff.setText('Guide: ON  (F8)' if CFG['on'] else 'Guide: OFF  (F8)')

    def auto_mark(self):
        """Hide our own windows, grab the screen, find the felt rectangle."""
        self.hide()
        was_on = CFG['on']
        CFG['on'] = False
        self.ov.hide()
        QtCore.QTimer.singleShot(300, lambda: self._auto(was_on))

    def _auto(self, was_on):
        rect, r = None, None
        try:
            import cv2
            import mss
            sys.path.insert(0, HERE)
            import poolsight as P
            sct = grabber()
            mon = sct.monitors[0]
            img = np.ascontiguousarray(np.asarray(sct.grab(mon))[..., :3])
            t = P.find_table(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            if t is not None:
                # the blue cushion strips count as felt colour; the balls bounce off their
                # inner edge, which in your game sits ~3.8% of the table height further in
                inset = CUSHION_INSET * (t.by1 - t.by0)
                rect = [t.bx0 + inset + mon['left'], t.by0 + inset + mon['top'],
                        t.bx1 - inset + mon['left'], t.by1 - inset + mon['top']]
        except Exception:
            log.exception('auto mark failed')
        CFG['on'] = was_on
        self.ov.show()
        if rect:
            CFG['felt'] = rect
            CFG['r'] = round((rect[2] - rect[0]) * BALL_FRAC, 1)
            m = None
            self.r_sl.blockSignals(True)
            self.r_sl.setValue(int(round(CFG['r'])))
            self.r_sl.blockSignals(False)
            save()
            self.msg.setText(f"Table found ✔  ball size {CFG['r']:.1f}" + ('' if m else ' (estimated)') +
                             '  - check the dotted box sits on the cushions')
        else:
            self.msg.setText("Couldn't find the table - make sure the game is visible, or use Manual mark.")
        self.show()
        self.ov.update()

    def auto_ball(self):
        if not CFG['felt']:
            self.msg.setText('Mark the table first.')
            return
        was_on = CFG['on']
        CFG['on'] = False                     # hide our lines while measuring
        self.ov.hide()
        self.hide()
        QtCore.QTimer.singleShot(300, lambda: self._ball(was_on))

    def _ball(self, was_on):
        # the size that matters is the one the game's shots behave with: measured with
        # test shots as a fixed share of the table width (the drawn ball looks smaller)
        m = round((CFG['felt'][2] - CFG['felt'][0]) * BALL_FRAC, 1)
        CFG['on'] = was_on
        self.ov.show()
        self.show()
        if m:
            CFG['r'] = m
            self.r_sl.blockSignals(True)
            self.r_sl.setValue(int(round(m)))
            self.r_sl.blockSignals(False)
            save()
            self.msg.setText(f'Ball size measured: {m:.1f} px ✔')
        else:
            self.msg.setText("Couldn't find any balls on the table (is something covering it?) - "
                             "set the size with the slider.")
        self.ov.update()

    def mark(self):
        self.hide()
        QtCore.QTimer.singleShot(250, self._open)

    def _open(self):
        self.sel = FeltSelector(self.geo)
        self.sel.chosen.connect(self._done)
        self.sel.show()

    def _done(self, rect):
        if rect:
            CFG['felt'] = rect
            CFG['r'] = round((rect[2] - rect[0]) * BALL_FRAC, 1)
            self.r_sl.setValue(int(round(CFG['r'])))
            save()
        self.show()
        self.ov.update()


# ---------------------------------------------------------------- keys
def qt_key_to_x(e):
    """Name of the pressed key as X11 understands it ('' if unusable)."""
    from Xlib import XK
    k = e.key()
    special = {QtCore.Qt.Key_Up: 'Up', QtCore.Qt.Key_Down: 'Down', QtCore.Qt.Key_Left: 'Left',
               QtCore.Qt.Key_Right: 'Right', QtCore.Qt.Key_Space: 'space', QtCore.Qt.Key_Tab: 'Tab',
               QtCore.Qt.Key_PageUp: 'Prior', QtCore.Qt.Key_PageDown: 'Next', QtCore.Qt.Key_Home: 'Home',
               QtCore.Qt.Key_End: 'End', QtCore.Qt.Key_Insert: 'Insert', QtCore.Qt.Key_Delete: 'Delete',
               QtCore.Qt.Key_Return: 'Return', QtCore.Qt.Key_Enter: 'KP_Enter', QtCore.Qt.Key_Backspace: 'BackSpace'}
    if k in special:
        name = special[k]
    elif QtCore.Qt.Key_F1 <= k <= QtCore.Qt.Key_F24:
        name = f'F{k - QtCore.Qt.Key_F1 + 1}'
    else:
        t = e.text()
        name = t.lower() if t and t.isprintable() and not t.isspace() else ''
        named = {'`': 'grave', '-': 'minus', '=': 'equal', '[': 'bracketleft', ']': 'bracketright',
                 ';': 'semicolon', "'": 'apostrophe', ',': 'comma', '.': 'period', '/': 'slash',
                 '\\': 'backslash'}
        name = named.get(name, name)
    if XK.string_to_keysym(name):
        return name
    sym = e.nativeVirtualKey()             # media / mouse-side keys etc: keep the raw keysym
    return f'{sym:#x}' if sym > 0xff and not 0xffe1 <= sym <= 0xffee else ''   # not Shift/Ctrl/Alt


def hotkeys(app, ov, panel):
    """Global keys (X11). Bindings come from CFG['keys'] and are re-grabbed when changed.
    Guide keys are only taken while the guide is on; show/hide and quit always."""
    try:
        from Xlib import X, XK, display
        d = display.Display()
        root = d.screen().root
        mods = (0, X.Mod2Mask, X.LockMask, X.Mod2Mask | X.LockMask, X.ShiftMask, X.ShiftMask | X.Mod2Mask)
        CODES = {'point': 1, 'line': 2, 'learn': 30, 'up': 3, 'down': 4}

        def kc(name):
            if not name:
                return 0
            if name.startswith('Mouse') and name[5:].isdigit():
                return ('btn', int(name[5:]))
            if name.startswith('0x'):
                return d.keysym_to_keycode(int(name, 16))
            return d.keysym_to_keycode(XK.string_to_keysym(name))

        d.set_error_handler(lambda err, *_: log.warning('X error (button already taken?): %s', err))
        grabbed = set()                       # keycodes / ('btn', n) currently grabbed
        cur_keys, cur_on, cur_bind = None, None, None
        always, guide = {}, {}
        while True:
            keys = dict(KEYS_DEFAULT, **CFG.get('keys', {}))
            on = bool(CFG['on'] and CFG['felt'])
            binding = bool(LIVE.get('binding'))   # let go of everything so the panel sees the press
            if keys != cur_keys or on != cur_on or binding != cur_bind:
                always = {kc(keys['toggle']): 'toggle', kc(keys['quit']): 'quit'}
                guide = {kc(keys[k]): v for k, v in CODES.items()}
                want = set() if binding else \
                    {k for k in always if k} | ({k for k in guide if k} if on else set())
                for k in grabbed - want:
                    for m in mods:
                        if isinstance(k, tuple):
                            root.ungrab_button(k[1], m)
                        else:
                            root.ungrab_key(k, m)
                for k in want - grabbed:
                    for m in mods:
                        if isinstance(k, tuple):
                            root.grab_button(k[1], m, True, X.ButtonPressMask, X.GrabModeAsync,
                                             X.GrabModeAsync, X.NONE, X.NONE)
                        else:
                            root.grab_key(k, m, True, X.GrabModeAsync, X.GrabModeAsync)
                d.flush()
                grabbed, cur_keys, cur_on, cur_bind = want, keys, on, binding
            if not d.pending_events():
                time.sleep(0.02)
                continue
            ev = d.next_event()
            if ev.type not in (X.KeyPress, X.ButtonPress) or LIVE.get('binding'):
                continue
            hit = ('btn', ev.detail) if ev.type == X.ButtonPress else ev.detail
            if hit in always:
                if always[hit] == 'quit':
                    QtCore.QMetaObject.invokeMethod(app, 'quit', QtCore.Qt.QueuedConnection)
                else:
                    QtCore.QMetaObject.invokeMethod(panel, 'toggle_slot', QtCore.Qt.QueuedConnection)
            elif hit in guide and on:
                code = guide[hit]
                if code in (3, 4) and ev.state & X.ShiftMask:
                    code += 10
                ov.key.emit(code)
    except Exception:
        log.exception('hotkeys failed')


def main():
    logging.basicConfig(filename=os.path.join(HERE, 'guide.log'), level=logging.INFO,
                        format='%(asctime)s %(message)s')
    log.info('started')
    load()
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName('PoolSight Guide')
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda n, _: app.quit())
    geo = app.primaryScreen().virtualGeometry()
    ov = Overlay(geo)
    ov.show()
    panel = Panel(ov, geo, app)
    panel.show()
    if not CFG['felt']:
        QtCore.QTimer.singleShot(500, panel.mark)        # first run: mark the table
    threading.Thread(target=hotkeys, args=(app, ov, panel), daemon=True).start()
    keep = QtCore.QTimer(timeout=lambda: None)
    keep.start(250)
    app.aboutToQuit.connect(save)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
