@echo off
rem ==============================================================
rem  Filter Portal for Windows: double-click this file to start.
rem  The first time, it sets up the Python packages in a .venv
rem  folder next to this file (needs internet, takes a few minutes).
rem  Needs Python 3.10 or newer: https://www.python.org/downloads/
rem ==============================================================
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" goto run

echo Setting up Filter Portal for the first time. This can take a few minutes...
where py >nul 2>nul && (py -3 -m venv .venv) || (python -m venv .venv)
if not exist ".venv\Scripts\python.exe" goto nopython
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto setupfailed

:run
rem "--setup-only" just sets up and stops (used by the automatic tests).
if "%~1"=="--setup-only" exit /b 0
".venv\Scripts\python.exe" filter_portal.py
if errorlevel 1 goto crashed
exit /b 0

:crashed
echo.
echo Filter Portal stopped with an error. The message is above.
if not defined CI pause
exit /b 1

:nopython
echo.
echo Python 3 was not found. Install it from https://www.python.org/downloads/
echo and tick "Add python.exe to PATH" in the installer. Then double-click
echo this file again.
if not defined CI pause
exit /b 1

:setupfailed
rmdir /s /q .venv
echo.
echo Setting up failed. Is the internet on? Try again in a moment.
if not defined CI pause
exit /b 1
