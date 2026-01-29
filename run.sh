#!/bin/bash
# ============================================================
# EchoSelf - 数字分身系统启动脚本
# ============================================================

set -e

echo "============================================================"
echo "  EchoSelf - 数字分身聊天系统"
echo "============================================================"

# 检查 Python 版本
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.10"

echo "检查 Python 版本: $python_version"

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source venv/bin/activate

# 检查依赖
if [ ! -f "venv/.deps_installed" ]; then
    echo "安装依赖（首次运行）..."
    pip install --upgrade pip
    pip install -r requirements.txt
    touch venv/.deps_installed
    echo "依赖安装完成"
fi

# 检查 ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "警告: ffmpeg 未安装，视频处理功能可能不可用"
    echo "请安装 ffmpeg:"
    echo "  macOS: brew install ffmpeg"
    echo "  Ubuntu: sudo apt install ffmpeg"
fi

# 设置环境变量（可选覆盖）
export OLLAMA_HOST="${OLLAMA_HOST:-http://192.168.50.218:11434}"
export OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:7b-instruct}"

echo ""
echo "配置信息:"
echo "  Ollama 服务器: $OLLAMA_HOST"
echo "  模型: $OLLAMA_MODEL"
echo ""

# 启动应用
echo "启动 EchoSelf..."
python app.py
