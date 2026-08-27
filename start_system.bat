@echo off
rem One-click launcher. ASCII-only, no parenthesized blocks (cmd parse-safe).
rem All Chinese messages live in start_system.py.
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" goto :run
echo [ERROR] Project venv not found at .venv\Scripts\python.exe
echo         Please install dependencies first. See README.md.
pause
exit /b 1

:run
".venv\Scripts\python.exe" start_system.py
pause
