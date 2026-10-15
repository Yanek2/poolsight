"""Test driver: raise Chrome, move/drag the real mouse (XTest), grab frames."""
import sys
import time

import mss
import numpy as np
from Xlib import X, display
from Xlib.ext import xtest
from Xlib.protocol import event

d = display.Display()
root = d.screen().root


def chrome_window():
    ids = root.get_full_property(d.intern_atom('_NET_CLIENT_LIST'), X.AnyPropertyType).value
    for wid in ids:
        w = d.create_resource_object('window', wid)
        try:
            cls = w.get_wm_class() or ('', '')
        except Exception:
            continue
        if 'chrome' in ' '.join(cls).lower():
            return w
    return None


def raise_chrome():
    w = chrome_window()
    atom = d.intern_atom('_NET_ACTIVE_WINDOW')
    ev = event.ClientMessage(window=w, client_type=atom, data=(32, [2, X.CurrentTime, 0, 0, 0]))
    root.send_event(ev, event_mask=X.SubstructureRedirectMask | X.SubstructureNotifyMask)
    d.flush()
    time.sleep(0.4)


def move(x, y):
    xtest.fake_input(d, X.MotionNotify, x=int(x), y=int(y))
    d.sync()


def drag(x0, y0, x1, y1, steps=20, hold=0.05):
    move(x0, y0)
    time.sleep(0.05)
    xtest.fake_input(d, X.ButtonPress, 1)
    d.sync()
    for k in range(1, steps + 1):
        move(x0 + (x1 - x0) * k / steps, y0 + (y1 - y0) * k / steps)
        time.sleep(0.01)
    time.sleep(hold)
    xtest.fake_input(d, X.ButtonRelease, 1)
    d.sync()


def click(x, y):
    move(x, y)
    time.sleep(0.05)
    xtest.fake_input(d, X.ButtonPress, 1)
    d.sync()
    time.sleep(0.05)
    xtest.fake_input(d, X.ButtonRelease, 1)
    d.sync()


def grab():
    with mss.MSS() as s:
        return np.ascontiguousarray(np.asarray(s.grab(s.monitors[0]))[..., :3])


if __name__ == '__main__':
    cmd = sys.argv[1]
    a = [float(v) for v in sys.argv[2:]]
    {'raise': lambda: raise_chrome(), 'move': lambda: move(*a), 'drag': lambda: drag(*a),
     'click': lambda: click(*a)}[cmd]()
