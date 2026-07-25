@echo off
echo ============================================
echo  Lloyds SME Intelligence - Environment Setup
echo  Group 33
echo ============================================
echo.

REM Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

REM Create virtual environment
if not exist ".venv" (
    echo [1/3] Creating virtual environment...
    python -m venv .venv
) else (
    echo [1/3] Virtual environment already exists, skipping.
)

REM Install packages
echo [2/3] Installing packages (this may take a few minutes)...
.venv\Scripts\pip install --upgrade pip -q
.venv\Scripts\pip install -r requirements.txt

REM Set up .env if not present
echo [3/3] Checking API keys...
if not exist "dashboard\.env" (
    copy "dashboard\.env.example" "dashboard\.env" >nul
    echo.
    echo  ACTION REQUIRED: Open dashboard\.env and fill in your API keys:
    echo    OPENAI_API_KEY=sk-...
    echo    GROQ_API_KEY=gsk_...
) else (
    echo  dashboard\.env already exists.
)

echo.
echo ============================================
echo  Setup complete!
echo  To run the dashboard:
echo.
echo    .venv\Scripts\activate
echo    streamlit run dashboard\app.py
echo ============================================
pause
