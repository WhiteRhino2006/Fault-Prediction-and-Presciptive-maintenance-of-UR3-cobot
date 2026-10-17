@echo off
REM Double-click to open the dashboard in your browser (keep this window open during the demo).
cd /d "%~dp0"
netstat -ano | findstr /R /C:":8502 .*LISTENING" >nul
if %errorlevel%==0 (
    echo Dashboard is already running - opening it in your browser.
    start "" http://localhost:8502
    timeout /t 3 >nul
    exit /b
)
python -m streamlit run dashboard.py --server.port 8502
pause
