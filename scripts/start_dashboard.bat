@echo off
REM ============================================================
REM Morocco Génération Green 2030 - Dashboard Launcher
REM Automatically sets up environment and starts Flask app
REM ============================================================

echo.
echo ============================================================
echo   Morocco Generation Green 2030 - Dashboard Launcher
echo ============================================================
echo.

REM Check if venv exists
if exist "venv\Scripts\activate.bat" (
    echo [INFO] Virtual environment found.
) else (
    echo [INFO] Virtual environment not found. Creating...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        echo [ERROR] Make sure Python is installed and in PATH.
        pause
        exit /b 1
    )
    echo [SUCCESS] Virtual environment created.
)

echo.
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if requirements are installed by checking for Flask
python -c "import flask" 2>nul
if errorlevel 1 (
    echo [INFO] Installing requirements...
    echo [INFO] This may take a few minutes...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install requirements.
        pause
        exit /b 1
    )
    echo [SUCCESS] Requirements installed.
) else (
    echo [INFO] Requirements already installed.
)

echo.
echo ============================================================
echo   Starting Flask Dashboard...
echo ============================================================
echo.
echo   Open your browser and go to: http://127.0.0.1:5000
echo.
echo   Press Ctrl+C to stop the server
echo ============================================================
echo.

REM Start Flask app
python app.py

REM If Flask exits, pause to see any error messages
if errorlevel 1 (
    echo.
    echo [ERROR] Flask app exited with an error.
    pause
)
