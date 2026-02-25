@echo off
setlocal enabledelayedexpansion
echo ====================================================
echo   Offline RAG Brain - Setup ^& Launcher
echo ====================================================

:: Check for Python
python --version >nul 2>&1
if "!errorlevel!" neq "0" (
    echo [ERROR] Python not found. Please install Python 3.10+
    pause
    exit /b
)

:: Check for Ollama
curl -s http://localhost:11434/api/tags >nul 2>&1
if "!errorlevel!" neq "0" (
    echo [WARNING] Ollama doesn't seem to be running.
    echo Please start Ollama before using the RAG features.
)

:: Setup Virtual Environment
if not exist ".venv" (
    echo [INFO] Creating virtual environment...
    python -m venv .venv
)

echo [INFO] Activating environment and installing dependencies...
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo [ERROR] Virtual environment activation script not found.
    pause
    exit /b
)

:: Use the python from venv directly to be safe
echo [INFO] Updating requirements...
python -m pip install -r requirements.txt

echo [INFO] Starting Backend Server on http://localhost:8000
echo [INFO] ^>^>^> OPEN THIS LINK IN YOUR BROWSER: http://localhost:8000
echo [INFO] (The local link is better for viewing live logs)
python server.py

pause
