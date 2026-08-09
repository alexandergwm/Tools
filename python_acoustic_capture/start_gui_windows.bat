@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"
set "VENV_DIR=.venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [声学采集工具] 首次启动，正在创建 Python 虚拟环境……
    where py >nul 2>nul
    if errorlevel 1 (
        echo 错误：未找到 Python 启动器 py。
        echo 请安装 Python 3.10 到 3.12：https://www.python.org/
        pause
        exit /b 1
    )
    py -3 -m venv "%VENV_DIR%"
    if errorlevel 1 goto :failed
)

echo [声学采集工具] 正在安装或更新本地程序……
"%PYTHON_EXE%" -m pip install -e .
if errorlevel 1 goto :failed

if not exist "audio\targets\demo_target_001.wav" (
    echo [声学采集工具] 正在生成无版权的流程验收音频……
    "%PYTHON_EXE%" -m acoustic_capture demo-audio --output-dir audio
    if errorlevel 1 goto :failed
)

echo [声学采集工具] 正在打开 Windows/RME 配置界面……
"%PYTHON_EXE%" -m acoustic_capture gui configs\rme_ucx.yaml
if errorlevel 1 goto :failed
exit /b 0

:failed
echo.
echo 错误：声学采集工具启动失败。
pause
exit /b 1
