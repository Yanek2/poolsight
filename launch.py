"""One-click start on any OS: serve your 8 Ball Pool game, open it, start PoolSight Guide."""
import functools
import http.server
import os
import socketserver
import sys
import threading
import urllib.request
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = 8081
URL = f"http://localhost:{PORT}"


def game_dir():
    for d in (os.environ.get("POOL_DIR"), os.path.expanduser("~/pool"), os.path.join(HERE, "game")):
        if d and os.path.isfile(os.path.join(d, "index.html")):
            return d
    raise SystemExit("pool game not found (expected ./game)")


def serving():
    try:
        return b"8 Ball Pool (Local)" in urllib.request.urlopen(URL, timeout=1).read()
    except Exception:
        return False


def serve():
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass
    handler = functools.partial(Quiet, directory=game_dir())
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", PORT), handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()


def open_browser():
    for name in ("chrome", "google-chrome", "windows-default", "macosx", None):
        try:
            if (webbrowser.get(name) if name else webbrowser).open(URL):
                return
        except webbrowser.Error:
            continue


if __name__ == "__main__":
    if "--no-game" not in sys.argv:
        if not serving():
            serve()
        open_browser()
    sys.path.insert(0, HERE)
    import guide
    guide.main()
