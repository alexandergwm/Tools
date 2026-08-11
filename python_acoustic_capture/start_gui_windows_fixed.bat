@echo off
chcp 65001 >nul
setlocal EnableExtensions EnableDelayedExpansion

rem Use user's conda environment python (adjust path if needed)
set "PYTHON_EXE=D:\Program_Data\miniconda3\envs\dev_default\python.exe"

rem sounddevice 的 Windows wheel 默认不加载 ASIO DLL；RME 正式采集需要显式开启。
if not "%ACOUSTIC_CAPTURE_ENABLE_ASIO%"=="0" set "SD_ENABLE_ASIO=1"

cd /d "%~dp0"

if not exist "%PYTHON_EXE%" (
    echo 错误：找不到指定的 Python: %PYTHON_EXE%
    echo 请确认 conda 环境路径或修改本脚本中的 PYTHON_EXE 为正确路径。
    pause
    exit /b 1
)

"%PYTHON_EXE%" -c "import struct; assert struct.calcsize('P') == 8" >nul 2>nul
if errorlevel 1 (
    echo 错误：指定的 Python 不是 64 位。请使用 64 位 Python 环境（例如 conda 的 dev_default）。
    pause
    exit /b 1
)

echo [声学采集工具] 检查 acoustic_capture 等依赖...
"%PYTHON_EXE%" -c "from importlib.metadata import version; import acoustic_capture,numpy,scipy,soundfile,sounddevice,matplotlib,yaml,xlsxwriter; print('deps ok')" >nul 2>nul
if errorlevel 1 (
    echo 依赖缺失或 acoustic_capture 未安装，尝试自动安装（需要网络）...
    "%PYTHON_EXE%" -m pip install -e .
    if errorlevel 1 (
        echo 自动安装失败。建议在 dev_default 环境中先用 conda 安装二进制依赖：
        echo conda install -n dev_default -c conda-forge numpy scipy matplotlib soundfile libsndfile pyyaml xlsxwriter sounddevice
        echo 然后运行： pip install -e .
        pause
        exit /b 1
    )
)

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
