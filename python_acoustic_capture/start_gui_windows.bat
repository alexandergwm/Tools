@echo off
setlocal
cd /d "%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch_gui_windows.ps1"
if errorlevel 1 (
    echo.
    echo Acoustic Capture failed to start.
    pause
    exit /b 1
)
exit /b 0
