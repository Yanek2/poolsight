# PoolSight Guide

A see-through overlay that draws aiming lines over **your own 8 Ball Pool game** (the local one in `game/`): lines from every pocket to the ball you lock, plus a movable rebound line with measured cushion bounces. It reads only the screen.

It only draws (and only takes its keys) while the focused window is the local game. That means the browser tab titled **"8 Ball Pool (Local)"**. Over any other window, including other pool games, it shows nothing.

## Setup

Python 3.10+.

| OS | Start |
|---|---|
| Windows | double-click `run.bat` |
| macOS | double-click `run.command` (first time: right-click → Open) |
| Linux | `./setup.sh` once, then `./run.sh` (guide only) or `venv/bin/python launch.py` (game + guide) |

`launch.py` serves `game/` at http://localhost:8081, opens it, and starts the guide.

**macOS permissions:** allow your terminal/Python under **Screen Recording** and **Accessibility** (System Settings → Privacy & Security).

## Keys (rebindable in the panel)

| Key | Action |
|---|---|
| `1` | lock the point (on a ball) |
| `2` | start/freeze the line |
| Up / Down | turn the line 0.1° (Shift = 1°) |
| `3` | learn from the shot |
| F8 | show / hide |
| F9 | quit |

On Windows/macOS the keys are global but also still reach the game. On Linux they're grabbed.

## Notes
- Auto-tune (takes real test shots in your game to measure the physics) is **Linux only**. On other systems, tune with the sliders.
- `app.py` is the older auto-detection overlay (Linux, not launched by default).
- `eval.py` + `snapshots/` = detection regression tests.
- `dbg/` = measurement scripts.
