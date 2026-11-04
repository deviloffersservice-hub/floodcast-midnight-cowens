@echo off
REM Usage:  predict.bat test.csv [train.csv]
REM   (or drag test.csv onto this file)
REM Writes predictions.csv next to test.csv.
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto no_setup
if "%~1"=="" goto usage
set "OUT=%~dp1predictions.csv"
if "%~2"=="" goto test_only
".venv\Scripts\python.exe" solution.py predict --test "%~f1" --train "%~f2" --out "%OUT%"
goto done
:test_only
".venv\Scripts\python.exe" solution.py predict --test "%~f1" --out "%OUT%"
:done
if errorlevel 1 goto failed
echo.
echo Saved: "%OUT%"
pause
exit /b 0
:failed
echo.
echo FAILED - see the message above.
pause
exit /b 1
:usage
echo Usage: predict.bat path\to\test.csv [path\to\train.csv]
pause
exit /b 1
:no_setup
echo Run setup.bat first.
pause
exit /b 1
