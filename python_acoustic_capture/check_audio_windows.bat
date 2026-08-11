@echo off
setlocal
cd /d "%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0check_audio_windows.ps1"
if errorlevel 1 (
    echo.
    echo Audio checks failed. Review the message above.
    pause
    exit /b 1
)
pause
