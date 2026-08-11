@echo off
setlocal
cd /d "%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0select_python_windows.ps1" -Force
if errorlevel 1 (
    echo.
    echo Python selection was not changed.
    pause
    exit /b 1
)
echo Python selection saved. Run start_gui_windows.bat to open the application.
pause
