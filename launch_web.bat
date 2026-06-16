@echo off
title CommSense - Web UI
cd /d "%~dp0"
echo.
echo  Starting CommSense Web UI...
echo  Backend: http://localhost:5000
echo  Frontend: http://localhost:5173
echo.

start "CommSense Backend" cmd /k "python ui/web_app.py"
timeout /t 3 /nobreak >nul

start "CommSense Frontend" cmd /k "cd ui/frontend && npm run dev"
timeout /t 4 /nobreak >nul

start "" "http://localhost:5173"
echo  Browser opening...
pause
