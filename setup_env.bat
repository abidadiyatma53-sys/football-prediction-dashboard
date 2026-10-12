@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating project virtual environment...
    py -3 -m venv .venv
    if errorlevel 1 (
        echo Failed to create .venv. Install Python 3 and ensure the py launcher is available.
        exit /b 1
    )
)

echo Installing project dependencies into .venv...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo.
echo Project environment is ready.
echo VS Code interpreter: %CD%\.venv\Scripts\python.exe
endlocal
