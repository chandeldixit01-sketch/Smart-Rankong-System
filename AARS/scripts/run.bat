@echo off
cd ..
echo ============================================
echo  Redrob AI Ranker
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found.
    pause & exit /b 1
)

:: Create venv if needed
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate

:: Install / upgrade Python deps
echo [1/3] Checking Python dependencies...
pip install -r requirements.txt --upgrade-strategy only-if-needed -q
echo [1/3] Done.
echo.

:: ---- Python ranker ----
echo [2/3] Using Python ranker (parallel, 20 cores)...
echo.
echo [3/3] Ranking candidates...
python -W ignore::RuntimeWarning data\rank.py

:launch_app
echo.
echo ============================================
echo  Starting web interface...
echo  Open: http://localhost:5000
echo ============================================
echo.
python -W ignore::RuntimeWarning app\app.py

pause
