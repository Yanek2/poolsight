# PoolSight Guide - tuning progress (2026-10-13)

App: ~/poolsight/guide.py (run.sh / desktop icon), settings in guide.json.
Tuning is done by staging shots in the user's game (~/pool/index.html) via a test harness
in a Chrome tab (T.setup / T.runSync drive the game's own functions) and measuring results.
The game's physics code is NOT copied - only measured.

Measured and already in guide.py:
- ball size 14.5/980 of felt width; pocket aim points + mouth widths (fit error 0.2-0.3 deg)
- start speed per power, slide/roll slowdown
- cushion: bounce ratio by speed (0.643 slow .. 0.771 fast), speed after cushion,
  skid distance and small shift (CUSH_* tables); cue-ball-in-the-way warning
- Power slider (planned power) -> 'too soft' marker

Results so far (game-checked):
- straight pots 35/36
- banks: 3/40 before speed model -> 29/42 with first speed model
  (1 cushion 17/21, 2 cushions 10/14, 3 cushions 2/7)

Round 2 (measured cushion grids, skidding vs rolling arrival, speed per power):
- 1-cushion banks 14/15, 2-cushion 6/15, 3-cushion 3/10 (game-checked, 40 shots)
- tried: corner limit 42 deg + cue follow-through blocking -> rejected good shots, reverted
- remaining misses: arrivals along a rail into a corner (jaw), cue ball interference on
  short near-straight banks, error growth over 2-3 cushions

Old next step: validate the measured cushion grid (dbg/genbanks3.py -> dbg/banks3.json, then play
them in the game harness), then cut-shot throw.
