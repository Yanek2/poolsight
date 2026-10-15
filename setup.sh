#!/bin/bash
# Linux / macOS one-time setup
cd "$(dirname "$0")"
python3 -m venv venv && venv/bin/pip install -q -r requirements.txt && echo "Done. Start with ./run.sh (Linux) or double-click run.command (Mac)."
