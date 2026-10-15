"""PoolSight app: click-through overlay + control panel + calibration.

The detection / physics engine is poolsight.py; this file is only the UI.
    F7 snapshot   F8 overlay on/off   F9 quit
"""
import atexit
import faulthandler
import json
import logging
import math
import os
import signal
import sys
import threading
import time

import cv2
import mss
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

import poolsight as P

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, 'calibration.json')
log = logging.getLogger('poolsight')

# what the overlay draws (ticked in the control panel)
SHOW = {'cue': True, 'obj': True, 'chain': True, 'after': True, 'rest': True, 'labels': True}
# manual mode: no detection at all - a line from the pinned cue ball toward the mouse,
# bouncing off the cushions of a table rectangle the player marked once
MANUAL = {'on': False, 'felt': None, 'r': 15.0, 'bounces': 3, 'cue': None, 'real': True,
          'guide': False, 'point': None}
GUIDE = {'moving': False, 'dir': None}     # live state of the key-2 line
SIDE_NAME = {'t': 'top', 'b': 'bottom', 'l': 'left', 'r': 'right'}


def new_grabber():
    return mss.MSS() if hasattr(mss, 'MSS') else mss.mss()


# ---------------------------------------------------------------- calibration file
def load_config():
    try:
        cfg = json.load(open(CONFIG))
    except Exception:
        return {}
    cal = cfg.get('cal', {})
    # drop impossible cushion values (cushions are about 4-9 % of the table height in)
    P.CAL.update({k: v for k, v in cal.items() if 0.02 < v < 0.11})
    P.STATE['cal_measured'] = set(cfg.get('cal_measured', []))
    if cfg.get('r_ratio'):
        P.STATE['r_ratio'] = cfg['r_ratio']
        P.STATE['r_learned'] = True
    if cfg.get('region'):
        P.STATE['region'] = tuple(cfg['region'])
    SHOW.update(cfg.get('show', {}))
    MANUAL.update(cfg.get('manual', {}))
    P.STATE['thick'] = cfg.get('thick', 1.0)
    P.STATE['panel'] = cfg.get('panel', {})
    return cfg


def save_config():
    cfg = {'cal': dict(P.CAL), 'cal_measured': sorted(P.STATE.get('cal_measured', set())),
           'r_ratio': P.STATE.get('r_ratio'), 'region': P.STATE.get('region'),
           'show': SHOW, 'manual': MANUAL, 'thick': P.STATE.get('thick', 1.0), 'panel': P.STATE.get('panel', {})}
    tmp = CONFIG + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(cfg, f, indent=1)
    os.replace(tmp, CONFIG)


def reset_calibration():
    P.CAL.clear()
    for k in ('cal_measured', 'r_ratio', 'r_learned', 'table', 'pbar', 'balls', 'bg', 'smooth', 'pockets'):
        P.STATE.pop(k, None)
    P.STATE['r'] = None
    save_config()


def checklist():
    """[(done, text)] for the calibration panel."""
    st = P.STATE
    tab = st.get('table') is not None
    meas = st.get('cal_measured', set())
    seen = st.get('pockets_seen', 0)
    items = [
        (tab, 'Table found' if tab else 'Table: not found - is the game on screen? Try "Select game area"'),
        (bool(st.get('r_learned')), 'Ball size learned' if st.get('r_learned')
         else 'Ball size: aim at any ball so the game shows its ghost ball'),
    ]
    marks = '  '.join(f"{SIDE_NAME[s]} {'✔' if s in meas else '·'}" for s in 'tblr')
    todo = [s for s in 'tblr' if s not in meas]
    items.append((not todo, f'Cushions: {marks}' + (
        '' if not todo else f'<br><i>optional, sharper bank shots: aim the cue ball straight '
                            f'into the {SIDE_NAME[todo[0]]} cushion (no ball in the way) and hold still</i>')))
    items.append((seen >= 4, f'Pockets: {seen}/6 seen' + ('' if seen >= 4 else ' - move windows off the table')))
    pb = st.get('pbar') is not None
    items.append((pb, 'Power bar found' if pb else 'Power bar: not found (stop-point rings need it)'))
    return items


# ---------------------------------------------------------------- overlay
class Overlay(QtWidgets.QWidget):
    toggle = QtCore.pyqtSignal()
    snapshot = QtCore.pyqtSignal()
    pin = QtCore.pyqtSignal()
    gkey = QtCore.pyqtSignal(int)
    result = QtCore.pyqtSignal(object, str)

    def __init__(self, mon):
        super().__init__(None, QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint |
                         QtCore.Qt.Tool | QtCore.Qt.X11BypassWindowManagerHint |
                         QtCore.Qt.WindowTransparentForInput)
        self.mon = mon
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating)
        self.setGeometry(mon['left'], mon['top'], mon['width'], mon['height'])
        self.res, self.status, self.on, self.good_at = None, 'starting', True, 0
        self.fps, self.note, self.note_until = 0.0, '', 0
        self.want_snap = False
        self.paused = False                    # e.g. while the area selector is open
        self.toggle.connect(self.flip)
        self.snapshot.connect(lambda: setattr(self, 'want_snap', True))
        self.result.connect(self.on_result)
        self.pin.connect(self.pin_cue)
        self.gkey.connect(self.guide_key)
        self.fast = QtCore.QTimer(self, timeout=self.manual_tick)
        self.fast.start(16)                   # manual line follows the mouse at ~60 fps
        threading.Thread(target=self.worker, daemon=True).start()

    def manual_tick(self):
        """Manual mode: the aim point follows the mouse only while the left button is
        held (that's how the game aims). Released = frozen, so you can use the power bar."""
        if MANUAL['guide'] and self.on:
            if GUIDE['moving'] and MANUAL['point']:
                pos = QtGui.QCursor.pos()
                dpr = self.devicePixelRatioF()
                v = np.array([pos.x() * dpr, pos.y() * dpr]) - np.array(MANUAL['point'])
                if np.linalg.norm(v) > 3:
                    GUIDE['dir'] = v / np.linalg.norm(v)
                self.update()
            return
        if not (MANUAL['on'] and self.on):
            return
        try:
            if not hasattr(self, 'xd'):
                from Xlib import display
                self.xd = display.Display()
                self.xroot = self.xd.screen().root
            q = self.xroot.query_pointer()
            held = bool(q.mask & (1 << 8))            # Button1Mask
            x, y = q.root_x, q.root_y
        except Exception:
            return
        fx0, fy0, fx1, fy1 = MANUAL['felt'] or (0, 0, 0, 0)
        on_table = fx0 - 40 < x < fx1 + 40 and fy0 - 40 < y < fy1 + 40
        if held and on_table:
            if not getattr(self, 'was_held', False):
                self.find_white()               # new aim drag: find where the cue ball is now
            self.aim_pt = (float(x), float(y))
            self.update()
        self.was_held = held

    def find_white(self):
        """Locate the cue ball: the one solid white, ball-sized blob on the felt."""
        if not MANUAL['felt']:
            return
        x0, y0, x1, y1 = [int(v) for v in MANUAL['felt']]
        r = MANUAL['r']
        try:
            img = np.asarray(new_grabber().grab({'left': x0, 'top': y0, 'width': x1 - x0,
                                                 'height': y1 - y0}))[..., :3]
            hsv = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2HSV)
            white = ((hsv[..., 1] < 45) & (hsv[..., 2] > 205)).astype(np.uint8)
            white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            n, _, st, ce = cv2.connectedComponentsWithStats(white)
            best = None
            for i in range(1, n):
                w, h, area = st[i, 2], st[i, 3], st[i, 4]
                # solid and round, about ball size (thin lines and rings fail this)
                if not (1.2 * r * r < area < 4.5 * r * r) or not (0.6 < w / max(h, 1) < 1.6):
                    continue
                if area < 0.55 * w * h:
                    continue
                if best is None or area > best[0]:
                    best = (area, ce[i])
            if best is not None:
                MANUAL['cue'] = [float(x0 + best[1][0]), float(y0 + best[1][1])]
        except Exception:
            log.exception('find_white failed')

    def guide_key(self, k):
        pos = QtGui.QCursor.pos()
        dpr = self.devicePixelRatioF()
        if k == 1:                               # set the meeting point
            MANUAL['point'] = [pos.x() * dpr, pos.y() * dpr]
            GUIDE['moving'], GUIDE['dir'] = False, None
            save_config()
        elif k == 2 and MANUAL['point']:         # start / freeze the movable line
            GUIDE['moving'] = not GUIDE['moving']
        elif k in (3, 4, 13, 14) and GUIDE['dir'] is not None:
            # Up / Down: turn the line a tiny bit (Shift = bigger steps)
            GUIDE['moving'] = False
            step = math.radians(0.1 if k < 10 else 1.0) * (1 if k in (3, 13) else -1)
            c, s_ = math.cos(step), math.sin(step)
            dx, dy = GUIDE['dir']
            GUIDE['dir'] = np.array([c * dx - s_ * dy, s_ * dx + c * dy])
        self.update()

    def pin_cue(self):
        """F2: pin the cue ball where the mouse is (snapped to the white ball under it)."""
        pos = QtGui.QCursor.pos()
        dpr = self.devicePixelRatioF()
        x, y = pos.x() * dpr, pos.y() * dpr
        r = MANUAL['r']
        try:
            R = int(3 * r)
            sct = new_grabber()
            img = np.asarray(sct.grab({'left': int(x) - R, 'top': int(y) - R, 'width': 2 * R, 'height': 2 * R}))[..., :3]
            hsv = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2HSV)
            white = (hsv[..., 1] < 50) & (hsv[..., 2] > 200)
            if white.sum() > 0.6 * r * r:
                ys, xs = np.nonzero(white)
                x, y = x - R + xs.mean(), y - R + ys.mean()
        except Exception:
            log.exception('pin snap failed')
        MANUAL['cue'] = [float(x), float(y)]
        self.aim_pt = None                      # wait for the next aim drag
        self.note, self.note_until = 'cue ball pinned (F2 again to move it)', time.time() + 2
        save_config()
        self.update()

    def flip(self):
        self.on = not self.on
        self.update()

    # ---- capture + analysis thread
    def worker(self):
        mon = self.mon
        sct = new_grabber()                    # mss objects are per-thread
        while True:
            if not self.on or self.paused or MANUAL['on'] or MANUAL['guide']:
                time.sleep(0.1)
                continue
            t0 = time.time()
            region = P.STATE.get('region')
            if P.STATE.get('table') is None:
                self.misses = getattr(self, 'misses', 0) + 1
            else:
                self.misses = 0
            if region and self.misses > 30 and self.misses % 2:
                region = None                  # game moved out of the chosen area: look everywhere
            if P.needs_full_frame():
                if region:                     # the player told us where the game is
                    x0, y0, x1, y1 = region
                    base, reg = (x0, y0), {'left': mon['left'] + x0, 'top': mon['top'] + y0,
                                          'width': x1 - x0, 'height': y1 - y0}
                else:
                    base, reg = (0, 0), mon
                full = True
            else:
                x0, y0, x1, y1 = P.table_region(mon['width'], mon['height'])
                base = (x0, y0)
                reg = {'left': mon['left'] + x0, 'top': mon['top'] + y0, 'width': x1 - x0, 'height': y1 - y0}
                full = False
            try:
                shot = np.ascontiguousarray(np.asarray(sct.grab(reg))[..., :3])
                pb = P.STATE.get('pbar')
                if pb is not None:              # how hard the player is pulling right now
                    px0 = max(pb[0] - 60, 0)
                    sub = np.ascontiguousarray(np.asarray(sct.grab(
                        {'left': mon['left'] + px0, 'top': mon['top'] + pb[1],
                         'width': pb[2] + 60 - px0, 'height': pb[3] - pb[1]}))[..., :3])
                    P.STATE['power'] = P.read_power(sub, (pb[0] - px0, 0, pb[2] - px0, pb[3] - pb[1]))
                P.STATE['own'] = P.own_mask(shot)
                res, status = P.analyse(shot, base, full=full)
            except Exception as e:              # never let one bad frame kill the app
                log.exception('analyse failed')
                res, status, shot = None, f'error: {e}', None
            if self.want_snap and full and shot is not None:
                self.want_snap = False
                self.save_snapshot(shot, res, status)
            dt = time.time() - t0
            self.fps = 0.8 * self.fps + 0.2 / max(dt, 1e-3)
            self.result.emit(res, status)
            time.sleep(max(0.0, 0.03 - dt))

    def save_snapshot(self, shot, res, status):
        d = os.path.join(HERE, 'snapshots')
        os.makedirs(d, exist_ok=True)
        name = time.strftime('%Y%m%d-%H%M%S')
        cv2.imwrite(os.path.join(d, name + '.png'), shot)
        if res:
            cv2.imwrite(os.path.join(d, name + '.debug.png'), P.draw_cv(shot, res))
        log.info(f'snapshot {name}: {status}')
        self.note, self.note_until = f'snapshot saved: snapshots/{name}.png', time.time() + 3

    def on_result(self, res, status):
        now = time.time()
        pr = res[3] if res else None
        if pr:
            # draw only once the aim has stopped moving (analysis is ~0.1 s behind)
            prev = getattr(self, 'prev_pr', None)
            steady = pr['steady'] and prev is not None
            self.prev_pr = pr
            self.unsteady = 0 if steady else getattr(self, 'unsteady', 0) + 1
            if steady:
                self.res, self.status, self.good_at = res, status, now
            elif self.unsteady >= 2:           # really moving (one odd frame isn't)
                self.res, self.status = None, 'aiming...'
        else:
            self.prev_pr = None
            if now - self.good_at > 0.4:      # hold the last good picture briefly
                self.res, self.status = res, status
        self.update()

    # ---- drawing
    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        try:
            self.paint(p)
        except Exception:
            log.exception('paint failed')
        finally:
            p.end()

    def rebound_path(self, start, d, bounces):
        """Straight line + cushion rebounds inside the marked felt; stops in a pocket."""
        m = MANUAL
        r = m['r']
        x0, y0, x1, y1 = m['felt']
        ix0, iy0, ix1, iy1 = x0 + r, y0 + r, x1 - r, y1 - r
        corners = [(ix0, iy0), (ix1, iy0), (ix0, iy1), (ix1, iy1)]
        mids = [((ix0 + ix1) / 2, iy0), ((ix0 + ix1) / 2, iy1)]
        pts, pt, pot = [np.array(start, float)], np.array(start, float), None
        for i in range(bounces + 1):
            ts = []
            if d[0] > 1e-9: ts.append(((ix1 - pt[0]) / d[0], 0))
            if d[0] < -1e-9: ts.append(((ix0 - pt[0]) / d[0], 0))
            if d[1] > 1e-9: ts.append(((iy1 - pt[1]) / d[1], 1))
            if d[1] < -1e-9: ts.append(((iy0 - pt[1]) / d[1], 1))
            if not ts:
                break
            tt, axis = min(ts)
            pt = pt + d * max(tt, 0)
            pts.append(pt)
            if any(np.hypot(*(pt - np.array(q))) < 2.2 * r for q in corners) or \
                    any(np.hypot(*(pt - np.array(q))) < 1.3 * r for q in mids):
                pot = pt
                break
            d = d.copy()
            d[axis] *= -(P.E_CUSHION if m['real'] else 1.0)
            d /= np.linalg.norm(d)
        return pts, pot

    def pockets(self):
        x0, y0, x1, y1 = MANUAL['felt']
        r = MANUAL['r']
        mx = (x0 + x1) / 2
        return [(x0 - 0.3 * r, y0 - 0.3 * r), (mx, y0 - 0.9 * r), (x1 + 0.3 * r, y0 - 0.3 * r),
                (x0 - 0.3 * r, y1 + 0.3 * r), (mx, y1 + 0.9 * r), (x1 + 0.3 * r, y1 + 0.3 * r)]

    def paint_guide(self, p):
        m = MANUAL
        if not m['felt']:
            return
        Q = lambda q: QtCore.QPointF(q[0], q[1])
        r = m['r']
        k = max(2, round(P.STATE.get('thick', 1.0) * 4))
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        # table outline + pockets so you can see the setup is right
        x0, y0, x1, y1 = m['felt']
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 70), 1, QtCore.Qt.DotLine))
        p.drawRect(QtCore.QRectF(x0, y0, x1 - x0, y1 - y0))
        for pk in self.pockets():
            p.drawEllipse(Q(pk), 1.4 * r, 1.4 * r)
        if not m['point']:
            return
        c = np.array(m['point'], float)
        # every pocket -> the point
        p.setPen(QtGui.QPen(QtGui.QColor(190, 120, 255, 200), k))
        for pk in self.pockets():
            p.drawLine(Q(pk), Q(c))
            # where the cue ball must touch to send the ball into this pocket (ghost ball)
            u = c - np.array(pk)
            if np.linalg.norm(u) > 1:
                g = c + u / np.linalg.norm(u) * 2 * r
                p.drawEllipse(Q(g), r, r)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 2))
        p.drawEllipse(Q(c), r, r)
        # the movable line (key 2) with rebounds
        if GUIDE['dir'] is not None:
            pts, pot = self.rebound_path(c, GUIDE['dir'], max(m['bounces'], 1))
            p.setPen(QtGui.QPen(QtGui.QColor(*P.COL['cue']), k + 1))
            for a_, b_ in zip(pts, pts[1:]):
                p.drawLine(Q(a_), Q(b_))
            if pot is not None:
                p.setPen(QtGui.QPen(QtGui.QColor(*P.COL['pot']), k + 2))
                p.drawEllipse(Q(pot), 1.6 * r, 1.6 * r)

    def paint_manual(self, p):
        m = MANUAL
        if not m['felt'] or not m['cue']:
            return
        r = m['r']
        x0, y0, x1, y1 = m['felt']
        ix0, iy0, ix1, iy1 = x0 + r, y0 + r, x1 - r, y1 - r      # where a ball centre can go
        if getattr(self, 'aim_pt', None) is None:
            return
        mouse = np.array(self.aim_pt, float)
        c = np.array(m['cue'], float)
        d = mouse - c
        if np.linalg.norm(d) < 2:
            return
        d = d / np.linalg.norm(d)
        k = max(1, round(P.STATE.get('thick', 1.0) * 3))
        pen = QtGui.QPen(QtGui.QColor(*P.COL['cue']), k)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        p.setPen(pen)
        Q = lambda q: QtCore.QPointF(q[0], q[1])
        corners = [(ix0, iy0), (ix1, iy0), (ix0, iy1), (ix1, iy1)]
        mids = [((ix0 + ix1) / 2, iy0), ((ix0 + ix1) / 2, iy1)]
        pt = c.copy()
        for i in range(m['bounces'] + 1):
            ts = []
            if d[0] > 1e-9: ts.append(((ix1 - pt[0]) / d[0], 0))
            if d[0] < -1e-9: ts.append(((ix0 - pt[0]) / d[0], 0))
            if d[1] > 1e-9: ts.append(((iy1 - pt[1]) / d[1], 1))
            if d[1] < -1e-9: ts.append(((iy0 - pt[1]) / d[1], 1))
            if not ts:
                break
            tt, axis = min(ts)
            nxt = pt + d * max(tt, 0)
            p.drawLine(Q(pt), Q(nxt))
            # pocket?
            if any(np.hypot(*(nxt - np.array(q))) < 2.2 * r for q in corners) or \
                    any(np.hypot(*(nxt - np.array(q))) < 1.3 * r for q in mids):
                p.setPen(QtGui.QPen(QtGui.QColor(*P.COL['pot']), k + 1))
                p.drawEllipse(Q(nxt), 1.6 * r, 1.6 * r)
                break
            if i == m['bounces']:
                break
            d = d.copy()
            d[axis] *= -(P.E_CUSHION if m['real'] else 1.0)
            d /= np.linalg.norm(d)
            pt = nxt
        # ghost ball at the first cushion contact helps judge the rebound
        p.setPen(QtGui.QPen(QtGui.QColor(*P.COL['cue']), 1, QtCore.Qt.DashLine))
        p.drawEllipse(Q(c), r, r)

    def paint(self, p):
        sc = 1 / self.devicePixelRatioF()
        p.scale(sc, sc)
        if MANUAL['guide']:
            if self.on:
                self.paint_guide(p)
            return
        if MANUAL['on']:
            if self.note and time.time() < self.note_until:
                p.setPen(QtGui.QColor(255, 220, 0))
                p.setFont(QtGui.QFont('Sans', 10, QtGui.QFont.Bold))
                p.drawText(12, self.mon['height'] - 60, self.note)
            if self.on:
                self.paint_manual(p)
            return
        region = P.STATE.get('region')
        if region and self.on:                 # faint frame around the chosen game area
            pn = QtGui.QPen(QtGui.QColor(255, 220, 0, 90), 1, QtCore.Qt.DotLine)
            p.setPen(pn)
            p.drawRect(region[0], region[1], region[2] - region[0], region[3] - region[1])
        if self.note and time.time() < self.note_until:
            p.setPen(QtGui.QColor(255, 220, 0))
            p.setFont(QtGui.QFont('Sans', 10, QtGui.QFont.Bold))
            p.drawText(12, self.mon['height'] - 60, self.note)
        if not self.on or not self.res or not self.res[3]:
            return
        t, balls, cue_i, pr = self.res
        r = t.r
        k = P.STATE.get('thick', 1.0)
        p.translate(*t.off)
        Q = lambda q: QtCore.QPointF(q[0], q[1])
        COL = P.COL

        def pen(name, width, dash=False):
            pn = QtGui.QPen(QtGui.QColor(*COL[name]), max(1, round(width * k)))
            pn.setCapStyle(QtCore.Qt.FlatCap)
            if dash:
                pn.setStyle(QtCore.Qt.DashLine)
            p.setPen(pn)

        def path(points, name, width, dash=False, skip_first=False):
            pen(name, width, dash)
            for i, (a, b) in enumerate(zip(points, points[1:])):
                if skip_first and i == 0:
                    continue                    # the game already draws this part
                p.drawLine(Q(a), Q(b))

        def from_edge(points):
            pts = list(points)                  # start at the ball's edge, keep the ball clean
            if len(pts) < 2:
                return []
            d = pts[1] - pts[0]
            L = np.linalg.norm(d)
            if L <= 1.2 * r:
                return pts[1:] if len(pts) > 2 else []
            pts[0] = pts[0] + d / L * 1.2 * r
            return pts

        def ring(c, rad, name, width=3):
            pen(name, width)
            p.drawEllipse(Q(c), rad, rad)

        def tick(points, name, width):         # short arrow: direction even when blocked
            if len(points) < 2:
                return
            d = points[1] - points[0]
            L = np.linalg.norm(d)
            if L < 1e-6:
                return
            d = d / L
            a, b = points[0] + d * 1.1 * r, points[0] + d * 2.4 * r
            pen(name, width)
            p.drawLine(Q(a), Q(b))
            side = np.array([-d[1], d[0]])
            for sgn in (1, -1):
                p.drawLine(Q(b), Q(b - d * 0.6 * r + side * sgn * 0.45 * r))

        def label(xy, text, size=12):
            if SHOW['labels']:
                p.setFont(QtGui.QFont('Sans', size, QtGui.QFont.Bold))
                p.drawText(Q(xy), text)

        # no antialiasing: drawn pixels keep their exact colours (see own_mask)
        p.setRenderHint(QtGui.QPainter.Antialiasing, False)
        if SHOW['cue']:
            cp = list(pr['cue_path'])
            if len(cp) > 2:                    # start just outside the game's ghost ring
                d0 = cp[2] - cp[1]
                L0 = np.linalg.norm(d0)
                if L0 > 1.5 * r:
                    cp[1] = cp[1] + d0 / L0 * 1.5 * r
            path(cp, 'cue', 3, skip_first=True)
        if SHOW['obj'] and len(pr['obj_path']) > 1:
            potted = pr['pot'] is not None
            sure = potted and pr.get('pot_sure', True)
            col = ('pot' if sure else 'maybe') if potted else 'obj'
            tick(pr['obj_path'], col, 3)
            path(from_edge(pr['obj_path']), col, 4)
            if potted:
                ring(pr['pot'], 1.8 * r, col, 4)
                below = pr['pot'][1] < (t.by0 + t.by1) / 2
                label((pr['pot'][0] - 18, pr['pot'][1] + (3.4 * r if below else -2.6 * r)),
                      'POT' if sure else 'POT?')
        if SHOW['chain']:
            for ch, cpot in zip(pr.get('chains', []), pr.get('chain_pots', [])):
                if len(ch) > 1:
                    tick(ch, 'pot' if cpot is not None else 'chain', 2)
                    path(from_edge(ch), 'pot' if cpot is not None else 'chain', 3)
                if cpot is not None:
                    ring(cpot, 1.8 * r, 'pot', 4)
        if SHOW['rest']:                       # where balls stop (while the bar is pulled)
            for key, pts, name in (('cue_stops', 'cue_path', 'cue'), ('obj_stops', 'obj_path', 'obj'),
                                   ('cue_after_stops', 'cue_after', 'after')):
                if pr.get(key) and len(pr[pts]) > 1:
                    pen(name, 2, dash=True)
                    p.drawEllipse(Q(pr[pts][-1]), r, r)
        if SHOW['after'] and len(pr['cue_after']) > 1:
            ca = list(pr['cue_after'])
            dd = ca[1] - ca[0]
            LL = np.linalg.norm(dd)
            if LL > 1.5 * r:
                ca[0] = ca[0] + dd / LL * 1.5 * r
                path(ca, 'after', 2, dash=True)
        if pr['cue_pot'] is not None and (SHOW['cue'] or SHOW['after']):
            ring(pr['cue_pot'], 2 * r, 'scratch', 3)
            label((pr['cue_pot'][0] - 32, pr['cue_pot'][1] - 2.4 * r), 'SCRATCH', 11)


# ---------------------------------------------------------------- game-area selector
class AreaSelector(QtWidgets.QWidget):
    """Full-screen dimmed layer: drag a box around the game. Esc cancels."""
    chosen = QtCore.pyqtSignal(object)

    def __init__(self, mon, prompt='Drag a box around the whole game (table + power bar).   Esc = cancel'):
        super().__init__(None, QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint |
                         QtCore.Qt.X11BypassWindowManagerHint)
        self.prompt = prompt
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setGeometry(mon['left'], mon['top'], mon['width'], mon['height'])
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
        self.b = e.pos()
        r = QtCore.QRect(self.a, self.b).normalized()
        dpr = self.devicePixelRatioF()
        if r.width() > 100 and r.height() > 60:
            self.chosen.emit((int(r.left() * dpr), int(r.top() * dpr),
                              int(r.right() * dpr), int(r.bottom() * dpr)))
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
                   self.prompt)
        if self.a is not None and self.b is not None:
            r = QtCore.QRect(self.a, self.b).normalized()
            p.setCompositionMode(QtGui.QPainter.CompositionMode_Clear)
            p.fillRect(r, QtCore.Qt.transparent)
            p.setCompositionMode(QtGui.QPainter.CompositionMode_SourceOver)
            p.setPen(QtGui.QPen(QtGui.QColor(255, 220, 0), 2))
            p.drawRect(r)
        p.end()


# ---------------------------------------------------------------- control panel
class Panel(QtWidgets.QWidget):
    def __init__(self, overlay, mon, app):
        super().__init__(None, QtCore.Qt.Window | QtCore.Qt.WindowStaysOnTopHint)
        self.ov, self.mon, self.app = overlay, mon, app
        self.setWindowTitle('PoolSight')
        self.setWindowIcon(QtGui.QIcon(os.path.join(HERE, 'icon.png')))
        self.setStyleSheet('''
            QWidget { background: #15181e; color: #e6e6e6; font-size: 12px; }
            QGroupBox { border: 1px solid #333a45; border-radius: 6px; margin-top: 10px; padding: 6px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; color: #ffd84d; }
            QPushButton { background: #263040; border: 1px solid #3b4a60; border-radius: 5px; padding: 6px; }
            QPushButton:hover { background: #31405a; }
            QPushButton#big { font-size: 14px; font-weight: bold; padding: 8px; }
            QCheckBox { spacing: 6px; }
        ''')
        lay = QtWidgets.QVBoxLayout(self)

        top = QtWidgets.QHBoxLayout()
        self.status = QtWidgets.QLabel()
        self.status.setStyleSheet('font-size: 13px; font-weight: bold;')
        top.addWidget(self.status, 1)
        self.fold = QtWidgets.QPushButton('▴')
        self.fold.setFixedWidth(30)
        self.fold.setToolTip('Fold the panel down to a small bar')
        self.fold.clicked.connect(self.toggle_fold)
        top.addWidget(self.fold)
        lay.addLayout(top)
        self.info = QtWidgets.QLabel()
        lay.addWidget(self.info)
        self.foldable = []

        self.onoff = QtWidgets.QPushButton()
        self.onoff.setObjectName('big')
        self.onoff.clicked.connect(lambda: self.ov.toggle.emit())
        lay.addWidget(self.onoff)

        show = QtWidgets.QGroupBox('Show')
        g = QtWidgets.QVBoxLayout(show)
        for key, text, col in (('cue', 'Cue ball path (banks)', 'cue'), ('obj', 'Object ball path + POT', 'obj'),
                               ('chain', 'Knocked-on balls', 'chain'), ('after', 'Cue ball after contact', 'after'),
                               ('rest', 'Where balls stop (while pulling power)', 'obj'),
                               ('labels', 'Labels (POT / SCRATCH)', 'pot')):
            cb = QtWidgets.QCheckBox(text)
            cb.setChecked(SHOW[key])
            c = P.COL[col]
            cb.setStyleSheet(f'QCheckBox::indicator:checked {{ background: rgb{tuple(c)}; border: 1px solid #ddd; }}')
            cb.toggled.connect(lambda v, k=key: (SHOW.__setitem__(k, v), save_config(), self.ov.update()))
            g.addWidget(cb)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel('Line thickness'))
        sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        sl.setRange(5, 30)
        sl.setValue(int(P.STATE.get('thick', 1.0) * 10))
        sl.valueChanged.connect(lambda v: (P.STATE.__setitem__('thick', v / 10), save_config(), self.ov.update()))
        row.addWidget(sl)
        g.addLayout(row)
        lay.addWidget(show)
        self.foldable.append(show)

        man = QtWidgets.QGroupBox('Manual line (no detection)')
        g = QtWidgets.QVBoxLayout(man)
        self.guide_on = QtWidgets.QCheckBox('Pocket guide  (1 = point, 2 = line, ↑↓ = fine turn)')
        self.guide_on.setChecked(MANUAL['guide'])
        self.guide_on.toggled.connect(self.set_guide)
        g.addWidget(self.guide_on)
        self.man_on = QtWidgets.QCheckBox('Cue line follows my aim drag')
        self.man_on.setChecked(MANUAL['on'])
        self.man_on.toggled.connect(self.set_manual)
        g.addWidget(self.man_on)
        b = QtWidgets.QPushButton('Mark table (drag over the felt)…')
        b.setToolTip('Drag a box from cushion edge to cushion edge - the blue felt only')
        b.clicked.connect(self.mark_felt)
        g.addWidget(b)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel('Ball size'))
        self.r_sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.r_sl.setRange(5, 40)
        self.r_sl.setValue(int(MANUAL['r']))
        self.r_sl.valueChanged.connect(lambda v: (MANUAL.__setitem__('r', float(v)), save_config(), self.ov.update()))
        row.addWidget(self.r_sl)
        g.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel('Bounces'))
        sp = QtWidgets.QSpinBox()
        sp.setRange(0, 6)
        sp.setValue(MANUAL['bounces'])
        sp.valueChanged.connect(lambda v: (MANUAL.__setitem__('bounces', v), save_config()))
        row.addWidget(sp)
        cb = QtWidgets.QCheckBox('Real cushions')
        cb.setToolTip('On: bounces as measured in your game (cushions soak up some speed). Off: perfect mirror.')
        cb.setChecked(MANUAL['real'])
        cb.toggled.connect(lambda v: (MANUAL.__setitem__('real', v), save_config()))
        row.addWidget(cb)
        g.addLayout(row)
        hint = QtWidgets.QLabel('Just drag to aim as usual: the line follows the drag and stays put when you let go. Cue ball is found automatically (F2 = set it by hand).')
        hint.setWordWrap(True)
        hint.setStyleSheet('color: #8a93a3;')
        g.addWidget(hint)
        lay.addWidget(man)
        self.foldable.append(man)

        cal = QtWidgets.QGroupBox('Calibration')
        g = QtWidgets.QVBoxLayout(cal)
        self.area = QtWidgets.QLabel()
        g.addWidget(self.area)
        row = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton('Select game area…')
        b.setToolTip('Drag a box around the game window - best when the game is not full screen')
        b.clicked.connect(self.select_area)
        row.addWidget(b)
        b = QtWidgets.QPushButton('Whole screen')
        b.clicked.connect(lambda: self.set_region(None))
        row.addWidget(b)
        g.addLayout(row)
        self.checks = QtWidgets.QLabel()
        self.checks.setWordWrap(True)
        self.checks.setTextFormat(QtCore.Qt.RichText)
        g.addWidget(self.checks)
        b = QtWidgets.QPushButton('Reset calibration')
        b.clicked.connect(self.reset)
        g.addWidget(b)
        lay.addWidget(cal)
        self.foldable.append(cal)

        row = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton('Snapshot (F7)')
        b.clicked.connect(lambda: self.ov.snapshot.emit())
        row.addWidget(b)
        b = QtWidgets.QPushButton('Quit (F9)')
        b.clicked.connect(self.app.quit)
        row.addWidget(b)
        lay.addLayout(row)
        tip = QtWidgets.QLabel('F8 = overlay on/off.  Lines appear when you hold the aim still.')
        tip.setStyleSheet('color: #8a93a3;')
        tip.setWordWrap(True)
        lay.addWidget(tip)
        self.foldable.append(tip)

        self.setFixedWidth(270)
        self.timer = QtCore.QTimer(self, timeout=self.refresh)
        self.timer.start(300)
        self.refresh()
        # start folded in the bottom-right corner (clear of the table and the power bar),
        # or wherever the player last left it
        pp = P.STATE.get('panel', {})
        if pp.get('folded', True):
            self.toggle_fold()
        if 'pos' in pp:
            self.move(*pp['pos'])
        else:
            self.adjustSize()
            # top-right, over the browser's tab strip - never over the table
            self.move(mon['left'] + mon['width'] - self.width() - 8, mon['top'] + 32)

    def toggle_fold(self):
        folded = not getattr(self, 'folded', False)
        self.folded = folded
        for w in self.foldable:
            w.setVisible(not folded)
        self.fold.setText('▾' if folded else '▴')
        self.adjustSize()
        P.STATE.setdefault('panel', {})['folded'] = folded

    def moveEvent(self, e):
        P.STATE.setdefault('panel', {})['pos'] = [self.x(), self.y()]

    def refresh(self):
        ov = self.ov
        st = ov.status
        col = '#5ee06a' if st.startswith('ok') else ('#ffd84d' if st in ('aiming...', 'starting') else '#ff7a7a')
        if MANUAL['guide']:
            st = 'pocket guide' + ('' if MANUAL['felt'] else ': mark the table') + \
                 ('' if MANUAL['point'] else ' · press 1 on a ball')
            col = '#5ee06a' if MANUAL['felt'] and MANUAL['point'] else '#ffd84d'
        elif MANUAL['on']:
            st = 'manual' + ('' if MANUAL['felt'] else ': mark the table')
            col = '#5ee06a' if MANUAL['felt'] and MANUAL['cue'] else '#ffd84d'
        self.status.setText(f"<span style='color:{col}'>●</span> {st}")
        pw = P.STATE.get('power')
        self.info.setText(f"{ov.fps:.0f} fps" + (f"   ·   power {pw * 100:.0f}%" if pw and pw > 0.02 else ''))
        self.onoff.setText('Overlay: ON  (F8)' if ov.on else 'Overlay: OFF  (F8)')
        rg = P.STATE.get('region')
        self.area.setText(f'Game area: {rg[2] - rg[0]}×{rg[3] - rg[1]} at ({rg[0]}, {rg[1]})' if rg
                          else 'Game area: whole screen (automatic)')
        rows = []
        for done, text in checklist():
            mark = "<span style='color:#5ee06a'>✔</span>" if done else "<span style='color:#ffb347'>●</span>"
            rows.append(f'{mark} {text}')
        self.checks.setText('<br>'.join(rows))
        # keep the calibration file current
        if time.time() - getattr(self, 'saved_at', 0) > 5:
            self.saved_at = time.time()
            save_config()

    def select_area(self):
        self.ov.paused = True
        self.hide()
        QtCore.QTimer.singleShot(250, self._open_selector)     # let the panel vanish first

    def _open_selector(self):
        self.sel = AreaSelector(self.mon)
        self.sel.chosen.connect(self._area_done)
        self.sel.show()

    def _area_done(self, rect):
        if rect is not None:
            self.set_region(rect)
        self.ov.paused = False
        self.show()

    def set_region(self, rect):
        P.STATE['region'] = rect
        for k in ('table', 'balls', 'bg', 'smooth', 'pbar'):
            P.STATE[k] = None
        save_config()
        self.refresh()

    def set_guide(self, v):
        MANUAL['guide'] = v
        if v and not MANUAL['felt']:
            self.mark_felt()
        save_config()
        self.ov.res = None
        self.ov.update()

    def set_manual(self, v):
        MANUAL['on'] = v
        if v and not MANUAL['felt']:
            self.mark_felt()
        save_config()
        self.ov.res = None
        self.ov.update()

    def mark_felt(self):
        self.ov.paused = True
        self.hide()
        QtCore.QTimer.singleShot(250, self._open_felt)

    def _open_felt(self):
        self.sel = AreaSelector(self.mon, 'Drag a box over the FELT: from cushion edge to cushion edge.   Esc = cancel')
        self.sel.chosen.connect(self._felt_done)
        self.sel.show()

    def _felt_done(self, rect):
        if rect is not None:
            MANUAL['felt'] = list(rect)
            # sensible ball size: a pool table is ~44 ball-radii... (felt width / 87)
            MANUAL["r"] = round((rect[2] - rect[0]) / 76, 1)       # measured in your game: felt ≈ 76 ball radii wide
            self.r_sl.setValue(int(round(MANUAL['r'])))
            save_config()
        self.ov.paused = False
        self.show()

    def reset(self):
        if QtWidgets.QMessageBox.question(self, 'PoolSight', 'Forget the learned ball size and cushions?') \
                == QtWidgets.QMessageBox.Yes:
            reset_calibration()
            self.refresh()


# ---------------------------------------------------------------- main
def hotkeys(app, ov):
    try:
        from Xlib import X, XK, display
        d = display.Display()
        root = d.screen().root
        keys = {d.keysym_to_keycode(XK.string_to_keysym('F2')): 'pin',
                d.keysym_to_keycode(XK.string_to_keysym('F7')): 'snap',
                d.keysym_to_keycode(XK.string_to_keysym('F8')): 'toggle',
                d.keysym_to_keycode(XK.string_to_keysym('F9')): 'quit'}
        for kc in keys:
            for mod in (0, X.Mod2Mask, X.LockMask, X.Mod2Mask | X.LockMask):
                root.grab_key(kc, mod, True, X.GrabModeAsync, X.GrabModeAsync)
        digit = {d.keysym_to_keycode(XK.string_to_keysym('1')): 1,
                 d.keysym_to_keycode(XK.string_to_keysym('2')): 2,
                 d.keysym_to_keycode(XK.string_to_keysym('Up')): 3,
                 d.keysym_to_keycode(XK.string_to_keysym('Down')): 4}
        grabbed = False
        while True:
            want = bool(MANUAL['guide'])
            if want != grabbed:                  # take keys 1/2 only while the guide is on
                for kc in digit:
                    for mod in (0, X.Mod2Mask, X.LockMask, X.Mod2Mask | X.LockMask,
                                X.ShiftMask, X.ShiftMask | X.Mod2Mask):
                        if want:
                            root.grab_key(kc, mod, True, X.GrabModeAsync, X.GrabModeAsync)
                        else:
                            root.ungrab_key(kc, mod)
                d.flush()
                grabbed = want
            if not d.pending_events():
                time.sleep(0.02)
                continue
            ev = d.next_event()
            if ev.type == X.KeyPress and ev.detail in digit and grabbed:
                code = digit[ev.detail]
                if code in (3, 4) and ev.state & X.ShiftMask:
                    code += 10
                ov.gkey.emit(code)
                continue
            if ev.type == X.KeyPress and ev.detail in keys:
                k = keys[ev.detail]
                log.info(f'hotkey {k}')
                if k == 'quit':
                    QtCore.QMetaObject.invokeMethod(app, 'quit', QtCore.Qt.QueuedConnection)
                elif k == 'snap':
                    ov.snapshot.emit()
                elif k == 'pin':
                    ov.pin.emit()
                else:
                    ov.toggle.emit()
    except Exception:
        log.exception('hotkeys failed')


def main():
    logf = os.path.join(HERE, 'poolsight.log')
    logging.basicConfig(filename=logf, level=logging.INFO, format='%(asctime)s %(message)s')
    faulthandler.enable(open(logf, 'a'))
    sys.excepthook = lambda *a: log.error('uncaught', exc_info=a)
    atexit.register(lambda: (save_config(), log.info('exiting')))
    log.info('started')
    load_config()

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName('PoolSight')
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda n, _: (log.info(f'got signal {n}, quitting'), app.quit()))
    mon = new_grabber().monitors[0]
    ov = Overlay(mon)
    ov.show()
    panel = Panel(ov, mon, app)
    panel.show()
    threading.Thread(target=hotkeys, args=(app, ov), daemon=True).start()
    keepalive = QtCore.QTimer(timeout=lambda: None)   # let Python see signals
    keepalive.start(250)
    code = app.exec_()
    log.info(f'event loop ended ({code})')
    sys.exit(code)


if __name__ == '__main__':
    main()
