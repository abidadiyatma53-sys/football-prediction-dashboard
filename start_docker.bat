@echo off
setlocal
cd /d "%~dp0"

echo ==============================================================================
echo  Launching Football Prediction Pipeline via Docker Compose
echo ==============================================================================
echo.
echo Building images and launching the FastAPI and Streamlit services...
echo.

where docker >nul 2>&1
if errorlevel 1 goto :docker_missing

docker info >nul 2>&1
if errorlevel 1 goto :docker_not_running

docker compose version >nul 2>&1
if errorlevel 1 goto :compose_missing

docker compose up -d --build
if errorlevel 1 goto :startup_failed

echo.
echo Docker Compose started the application services.
echo Check their status with: docker compose ps
echo - FastAPI Swagger Docs : http://localhost:8000/docs
echo - Streamlit Dashboard  : http://localhost:8501
echo.
pause
exit /b 0

:docker_missing
echo ERROR: Docker CLI was not found.
echo Install Docker Desktop, then reopen Command Prompt and try again.
goto :failed

:docker_not_running
echo ERROR: Docker is installed but the Docker engine is not running.
echo Start Docker Desktop, wait until it is ready, then try again.
goto :failed

:compose_missing
echo ERROR: Docker Compose is unavailable.
echo Update or reinstall Docker Desktop, then try again.
goto :failed

:startup_failed
echo.
echo ERROR: Docker Compose failed to start the services.
echo Review the error above, then check service logs with:
echo docker compose logs --tail=50
goto :failed

:failed
echo.
pause
exit /b 1
