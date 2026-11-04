@echo off
REM Starts the FLOODCAST web app at http://127.0.0.1:7860 (opens your browser).
REM Close this window to stop the app.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" app.py
pause
