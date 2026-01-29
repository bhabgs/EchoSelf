#!/bin/bash
# EchoSelf GPU Server 启动脚本 (EchoMimic V3)

# 设置环境变量
export PYTHONPATH=$(pwd)/echomimic_v3_src:$PYTHONPATH

# 激活环境
if [ -d "venv" ]; then
    source venv/bin/activate
elif command -v conda &> /dev/null; then
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate echoself-server
fi

# 启动服务
echo "=========================================="
echo "启动 EchoSelf GPU Server (EchoMimic V3)"
echo "=========================================="
echo ""
echo "API 地址: http://0.0.0.0:8000"
echo "文档地址: http://0.0.0.0:8000/docs"
echo ""
echo "V3 优势:"
echo "  - 8 步推理 (比 V2 的 30 步快 4 倍)"
echo "  - 无需 pose 数据"
echo "  - 12GB 显存即可运行"
echo ""

python api_server.py
