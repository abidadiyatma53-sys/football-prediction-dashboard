@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Project environment not found. Run setup_env.bat first.
    pause
    exit /b 1
)
echo Starting Football Prediction Streamlit Web Dashboard...
".venv\Scripts\python.exe" -m streamlit run app.py
pause
