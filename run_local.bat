@echo off
echo ====================================================
echo   Offline RAG Brain - Setup & Launcher
echo ====================================================

:: Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.10+
    pause
    exit /b
)

:: Check for Ollama
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Ollama doesn't seem to be running.
    echo Please start Ollama before using the RAG features.
)

:: Setup Virtual Environment
if not exist ".venv" (
    echo [INFO] Creating virtual environment...
    python -m venv .venv
)

echo [INFO] Activating environment and installing dependencies...
call .venv\Scripts\activate
pip install -r requirements.txt

echo [INFO] Starting Backend Server on http://localhost:8000
echo [INFO] You can now open your UI in the browser.
python server.py

pause
