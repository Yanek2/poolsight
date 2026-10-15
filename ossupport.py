"""Per-OS support for PoolSight Guide: window guard, HiDPI-safe screen grabs,
and global hotkeys on Windows / macOS (Linux keeps the X11 key grabs in guide.py)."""
import platform
import re
import subprocess
import threading
import time

import numpy as np

SYSTEM = platform.system()            # 'Linux', 'Windows', 'Darwin'
GAME_TITLE = "8 Ball Pool (Local)"    # <title> of the user's own game page
_BROWSER = r"(Google Chrome|Chromium|Mozilla Firefox|Microsoft\W*Edge|Brave|Opera|Vivaldi)"
_TITLE_RE = re.compile(r"^" + re.escape(GAME_TITLE) + r"( [-—] .*)? [-—] " + _BROWSER + r"$")
MAC_BROWSERS = ("Google Chrome", "Chromium", "Firefox", "Microsoft Edge", "Brave Browser", "Safari", "Opera", "Vivaldi")


# ----------------------------------------------------------------- guard
def active_window():
    """(app name, window title) of the focused window; ('', '') if unknown."""
    try:
        if SYSTEM == "Windows":
            import ctypes
            u = ctypes.windll.user32
            h = u.GetForegroundWindow()
            n = u.GetWindowTextLengthW(h)
            buf = ctypes.create_unicode_buffer(n + 1)
            u.GetWindowTextW(h, buf, n + 1)
            return "", buf.value
        if SYSTEM == "Darwin":
            script = ('tell application "System Events" to tell (first application process whose frontmost is true)\n'
                      'return name & "\\n" & (name of front window)\nend tell')
            out = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=2).stdout
            app, _, title = out.strip().partition("\n")
            return app, title
        from Xlib import display, X
        d = display.Display()
        try:
            root = d.screen().root
            wid = root.get_full_property(d.intern_atom("_NET_ACTIVE_WINDOW"), X.AnyPropertyType).value[0]
            w = d.create_resource_object("window", wid)
            name = w.get_full_property(d.intern_atom("_NET_WM_NAME"), 0)
            return "", name.value.decode("utf-8", "replace") if name else (w.get_wm_name() or "")
        finally:
            d.close()
    except Exception:
        return "", ""


def on_own_game():
    """True only when the focused window is the user's own game in a browser."""
    app, title = active_window()
    if SYSTEM == "Darwin":                 # macOS titles have no browser suffix
        return title == GAME_TITLE and app in MAC_BROWSERS
    return bool(_TITLE_RE.match(title))


class Guard:
    """Polls the focused window in the background; .ok is read by the overlay."""

    def __init__(self, own_titles=("PoolSight",)):
        self.ok = False
        self.own = own_titles
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            app, title = active_window()
            if any(t in title for t in self.own) or (SYSTEM == "Darwin" and app in ("Python", "python3")):
                pass                       # our own panel: keep the last state
            else:
                self.ok = on_own_game()
            time.sleep(0.4)


# ---------------------------------------------------------- screen grabs
class Grabber:
    """mss wrapper that always works in physical pixels.

    On macOS Retina mss takes coordinates in points but returns 2x images,
    while the overlay works in pixels; this converts the box and resizes so a
    request for a WxH pixel box returns exactly WxH pixels on every OS."""

    def __init__(self):
        import mss
        self.s = mss.MSS() if hasattr(mss, "MSS") else mss.mss()
        m = self.s.monitors[0]
        probe = self.s.grab({"left": m["left"], "top": m["top"], "width": 16, "height": 16})
        self.k = probe.width / 16.0

    @property
    def monitors(self):
        if self.k == 1:
            return self.s.monitors
        return [{key: int(round(v * self.k)) for key, v in mon.items()} for mon in self.s.monitors]

    def grab(self, box):
        if self.k == 1:
            return self.s.grab(box)
        import cv2
        k = self.k
        b = {"left": int(box["left"] / k), "top": int(box["top"] / k),
             "width": max(1, int(round(box["width"] / k))), "height": max(1, int(round(box["height"] / k)))}
        img = np.asarray(self.s.grab(b))
        return cv2.resize(img, (int(box["width"]), int(box["height"])), interpolation=cv2.INTER_LINEAR)


# --------------------------------------------------------------- hotkeys
def pynput_hotkeys(get_keys, is_on, is_binding, on_always, on_guide, codes):
    """Global keys via pynput (Windows/macOS). Keys still reach the game too.

    get_keys() -> {'toggle': 'F8', 'point': '1', ...} in X11 keysym names.
    on_always(action) for 'toggle'/'quit'; on_guide(code) for guide keys."""
    from pynput import keyboard, mouse

    special = {"Up": keyboard.Key.up, "Down": keyboard.Key.down, "Left": keyboard.Key.left,
               "Right": keyboard.Key.right, "space": keyboard.Key.space, "Tab": keyboard.Key.tab,
               "Prior": keyboard.Key.page_up, "Next": keyboard.Key.page_down, "Home": keyboard.Key.home,
               "End": keyboard.Key.end, "Delete": keyboard.Key.delete, "Return": keyboard.Key.enter,
               "BackSpace": keyboard.Key.backspace}
    named = {"grave": "`", "minus": "-", "equal": "=", "bracketleft": "[", "bracketright": "]",
             "semicolon": ";", "apostrophe": "'", "comma": ",", "period": ".", "slash": "/", "backslash": "\\"}
    shift = {"down": False}

    def match(name, key):
        if not name:
            return False
        if name in special:
            return key == special[name]
        if re.fullmatch(r"F\d{1,2}", name):
            return key == getattr(keyboard.Key, name.lower(), None)
        ch = named.get(name, name)
        return len(ch) == 1 and isinstance(key, keyboard.KeyCode) and key.char is not None \
            and key.char.lower() == ch

    def on_press(key):
        if key in (keyboard.Key.shift, keyboard.Key.shift_r):
            shift["down"] = True
            return
        if is_binding():
            return
        keys = get_keys()
        for action in ("toggle", "quit"):
            if match(keys.get(action), key):
                on_always(action)
                return
        if is_on():
            for name, code in codes.items():
                if match(keys.get(name), key):
                    on_guide(code + 10 if code in (3, 4) and shift["down"] else code)
                    return

    def on_release(key):
        if key in (keyboard.Key.shift, keyboard.Key.shift_r):
            shift["down"] = False

    def on_click(x, y, button, pressed):     # 'Mouse8' / 'Mouse9' side buttons
        if not pressed or is_binding():
            return
        n = {getattr(mouse.Button, "x1", None): 8, getattr(mouse.Button, "x2", None): 9}.get(button)
        if n is None:
            return
        keys = get_keys()
        for action in ("toggle", "quit"):
            if keys.get(action) == f"Mouse{n}":
                on_always(action)
                return
        if is_on():
            for name, code in codes.items():
                if keys.get(name) == f"Mouse{n}":
                    on_guide(code)
                    return

    for lst in (keyboard.Listener(on_press=on_press, on_release=on_release), mouse.Listener(on_click=on_click)):
        lst.daemon = True
        lst.start()
