#!/bin/bash
# EchoSelf GPU Server 启动脚本

# 设置环境变量
export FFMPEG_PATH=$(pwd)/ffmpeg-4.4-amd64-static
export PATH=$FFMPEG_PATH:$PATH
export PYTHONPATH=$(pwd)/echomimic_v2_src:$PYTHONPATH

# 激活环境
if [ -d "venv" ]; then
    source venv/bin/activate
elif command -v conda &> /dev/null; then
    conda activate echoself-server
fi

# 启动服务
echo "启动 EchoSelf GPU Server..."
echo "API 地址: http://0.0.0.0:8000"
echo "文档地址: http://0.0.0.0:8000/docs"
echo ""

python api_server.py
