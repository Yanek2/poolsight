@echo off
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (
  py -3 -m venv venv || python -m venv venv
  venv\Scripts\pip install -q -r requirements.txt
)
start "" venv\Scripts\pythonw.exe launch.py
