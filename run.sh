#!/bin/sh
# Start PoolSight Guide (restarts it if it's already running).
cd "$(dirname "$0")"
pkill -f "venv/bin/python .*poolsight/(poolsight|guide).py$" 2>/dev/null
exec ./venv/bin/python "$PWD/guide.py"
