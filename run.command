#!/bin/bash
# macOS: double-click to start (game + PoolSight Guide)
cd "$(dirname "$0")"
[ -x venv/bin/python ] || ./setup.sh
exec venv/bin/python launch.py
