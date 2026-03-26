@echo off
echo ============================================
echo   AI Media Intelligence System
echo ============================================
echo.

cd /d "%~dp0"

echo [1] Activating virtual environment...
call venv\Scripts\activate.bat

echo [2] Launching Streamlit UI...
echo     Open http://localhost:8501 in your browser
echo.
echo Press Ctrl+C to stop the server.
echo ============================================
streamlit run app.py

pause
