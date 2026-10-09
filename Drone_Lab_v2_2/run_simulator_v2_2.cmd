@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" "drone_sim_3d_2.py" %*
) else (
    python "drone_sim_3d_2.py" %*
)
if errorlevel 1 (
    echo See README_v2_2.md for Python and model setup.
    pause
)
