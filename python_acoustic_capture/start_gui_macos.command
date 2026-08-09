#!/bin/zsh

set -e
SCRIPT_DIR="${0:A:h}"
VENV_DIR="$SCRIPT_DIR/.venv"
PYTHON_EXE="$VENV_DIR/bin/python"

cd "$SCRIPT_DIR"

if [[ ! -x "$PYTHON_EXE" ]]; then
    echo "[声学采集工具] 正在创建 Python 虚拟环境……"
    if ! command -v python3 >/dev/null 2>&1; then
        echo "错误：未找到 python3，请安装 Python 3.10 或更高版本。"
        read "?按回车键关闭……"
        exit 1
    fi
    python3 -m venv "$VENV_DIR"
fi

echo "[声学采集工具] 正在安装或更新本地程序……"
"$PYTHON_EXE" -m pip install -e .

if [[ ! -f "audio/targets/demo_target_001.wav" ]]; then
    echo "[声学采集工具] 正在生成无版权的流程验收音频……"
    "$PYTHON_EXE" -m acoustic_capture demo-audio --output-dir audio
fi

echo "[声学采集工具] 正在打开图形界面……"
"$PYTHON_EXE" -m acoustic_capture gui configs/simulated.yaml
