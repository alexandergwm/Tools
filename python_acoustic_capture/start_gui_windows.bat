@echo off
chcp 65001 >nul
setlocal EnableExtensions EnableDelayedExpansion

rem sounddevice 的 Windows wheel 默认不加载 ASIO DLL；RME 正式采集需要显式开启。
if not "%ACOUSTIC_CAPTURE_ENABLE_ASIO%"=="0" set "SD_ENABLE_ASIO=1"

cd /d "%~dp0"
set "VENV_DIR=.venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
set "PYTHON_LAUNCHER="
set "NEED_INSTALL=0"

if not exist "%PYTHON_EXE%" (
    set "NEED_INSTALL=1"
    echo [声学采集工具] 首次启动，正在创建 Python 虚拟环境……
    py -3.12 -c "import sys; assert sys.version_info[:2] == (3, 12)" >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_LAUNCHER=py -3.12"
    ) else (
        py -3.11 -c "import sys; assert sys.version_info[:2] == (3, 11)" >nul 2>nul
        if not errorlevel 1 (
            set "PYTHON_LAUNCHER=py -3.11"
        ) else (
            py -3.10 -c "import sys; assert sys.version_info[:2] == (3, 10)" >nul 2>nul
            if not errorlevel 1 set "PYTHON_LAUNCHER=py -3.10"
        )
    )
    if not defined PYTHON_LAUNCHER (
        python -c "import sys; assert (3, 10) ^<= sys.version_info[:2] ^<= (3, 12)" >nul 2>nul
        if not errorlevel 1 set "PYTHON_LAUNCHER=python"
    )
    if not defined PYTHON_LAUNCHER (
        echo 错误：没有找到 64 位 Python 3.10、3.11 或 3.12。
        echo 请从 https://www.python.org/ 安装 64 位 Python 3.12，并勾选 Add Python to PATH。
        pause
        exit /b 1
    )
    !PYTHON_LAUNCHER! -m venv "%VENV_DIR%"
    if errorlevel 1 goto :failed
)

"%PYTHON_EXE%" -c "import struct; assert struct.calcsize('P') == 8" >nul 2>nul
if errorlevel 1 (
    echo 错误：当前虚拟环境不是 64 位 Python。请删除 .venv 后安装 64 位 Python 3.12，再重新启动。
    pause
    exit /b 1
)

"%PYTHON_EXE%" -c "from importlib.metadata import version; import acoustic_capture,numpy,scipy,soundfile,sounddevice,matplotlib,yaml,xlsxwriter; assert tuple(map(int,version('sounddevice').split('.')[:3])) >= (0,5,1)" >nul 2>nul
if errorlevel 1 set "NEED_INSTALL=1"

if "%NEED_INSTALL%"=="1" (
    echo [声学采集工具] 正在安装本地程序及依赖，首次运行需要网络……
    "%PYTHON_EXE%" -m pip install -e .
    if errorlevel 1 goto :failed
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
