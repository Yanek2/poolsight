"""Resize / restore the Chrome window for testing (X11)."""
import sys, time, json
sys.path.insert(0, '/home/cobalt_y/poolsight/dbg')
import drive
from Xlib import X
from Xlib.protocol import event
d, root = drive.d, drive.root

def frame_of(w):
    while True:
        tree = w.query_tree()
        if tree.parent == root or tree.parent.id == root.id:
            return w
        w = tree.parent

def geom(w):
    g = frame_of(w).get_geometry()
    return g.x, g.y, g.width, g.height

def set_state(w, add, props):
    at = d.intern_atom('_NET_WM_STATE')
    ev = event.ClientMessage(window=w, client_type=at,
                             data=(32, [1 if add else 0] + [d.intern_atom(p) for p in props] + [1, 0][:4 - len(props)]))
    root.send_event(ev, event_mask=X.SubstructureRedirectMask | X.SubstructureNotifyMask); d.flush()

w = drive.chrome_window()
cmd = sys.argv[1]
if cmd == 'info':
    print(geom(w))
elif cmd == 'size':
    x, y, W, H = map(int, sys.argv[2:6])
    set_state(w, False, ['_NET_WM_STATE_MAXIMIZED_VERT', '_NET_WM_STATE_MAXIMIZED_HORZ'])
    time.sleep(0.4)
    w.configure(x=x, y=y, width=W, height=H); d.sync(); time.sleep(0.8)
    print(geom(w))
elif cmd == 'max':
    set_state(w, True, ['_NET_WM_STATE_MAXIMIZED_VERT', '_NET_WM_STATE_MAXIMIZED_HORZ']); time.sleep(0.8)
    print(geom(w))
