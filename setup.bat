@echo off
REM ============================================================
REM  FLOODCAST (Midnight Cowens) - one-time setup for Windows 10/11
REM  Installs Python 3.12 + MS C++ runtime if missing, creates .venv,
REM  installs pinned packages, then runs the self-tests.
REM ============================================================
setlocal EnableExtensions
cd /d "%~dp0"
set "PY="

call :find_python
if defined PY goto have_python

echo [setup] Python 3.12 not found. Installing it with winget (official python.org build)...
winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
call :find_python
if defined PY goto have_python
echo.
echo [setup] ERROR: Python 3.12 could not be installed automatically.
echo         Install it manually from https://www.python.org/downloads/windows/
echo         (Windows installer 64-bit, Python 3.12.x, tick "Add python.exe to PATH"), then run setup.bat again.
pause
exit /b 1

:have_python
echo [setup] Using Python: %PY%

if exist "%SystemRoot%\System32\vcomp140.dll" if exist "%SystemRoot%\System32\msvcp140.dll" goto have_vcredist
echo [setup] Microsoft Visual C++ runtime missing (LightGBM needs it). Installing... accept the admin prompt.
winget install -e --id Microsoft.VCRedist.2015+.x64 --accept-package-agreements --accept-source-agreements
if not exist "%SystemRoot%\System32\vcomp140.dll" (
  echo [setup] Could not install it automatically. Download and run:
  echo         https://aka.ms/vs/17/release/vc_redist.x64.exe
  echo         then run setup.bat again.
  pause
  exit /b 1
)
:have_vcredist

if not exist ".venv\Scripts\python.exe" (
  echo [setup] Creating virtual environment .venv ...
  "%PY%" -m venv .venv
  if errorlevel 1 goto fail
)
echo [setup] Installing packages (first time takes a few minutes)...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check --upgrade pip
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto fail

echo [setup] Running self-tests...
".venv\Scripts\python.exe" -m pytest -q tests
if errorlevel 1 goto fail

echo.
echo ============================================================
echo  Setup complete.
echo   - Web app:      double-click run_app.bat
echo   - CSV -> predictions.csv:  predict.bat path\to\test.csv
echo ============================================================
pause
exit /b 0

:fail
echo.
echo [setup] FAILED - scroll up for the error message.
pause
exit /b 1

:find_python
set "PY="
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if defined PY exit /b 0
if exist "%ProgramFiles%\Python312\python.exe" set "PY=%ProgramFiles%\Python312\python.exe"
if defined PY exit /b 0
for /f "delims=" %%i in ('py -3.12 -c "import sys; print(sys.executable)" 2^>nul') do set "PY=%%i"
exit /b 0
