@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Project environment not found. Run setup_env.bat first.
    pause
    exit /b 1
)
echo Starting Football Prediction FastAPI REST Server...
".venv\Scripts\python.exe" -m uvicorn api:app --reload --port 8000
pause
