@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"
set "PYTHON_EXE=.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo 请先双击 start_gui_windows.bat 完成安装，再运行本检查脚本。
    pause
    exit /b 1
)

echo ============================================================
echo 当前 PortAudio 能看到的音频设备
echo ============================================================
"%PYTHON_EXE%" -m acoustic_capture devices
echo.

echo ============================================================
echo 检查 RME 配置中的设备、48 kHz 和通道数量
echo ============================================================
"%PYTHON_EXE%" -m acoustic_capture hardware-check configs\rme_ucx.yaml
if errorlevel 1 (
    echo.
    echo 声卡检查失败。请在 GUI 中选择正确的 RME 输入、输出设备后保存配置。
    pause
    exit /b 1
)

echo.
choice /C YN /M "是否进行 5 秒双麦克风仅录制测试"
if errorlevel 2 exit /b 0

"%PYTHON_EXE%" -m acoustic_capture check-input configs\rme_ucx.yaml --duration 5
if errorlevel 1 (
    echo 录制测试失败，请检查 RME 驱动、TotalMix 路由、采样率和麦克风权限。
    pause
    exit /b 1
)

echo 双麦克风录制测试完成，结果已保存到 runs 目录。
pause
