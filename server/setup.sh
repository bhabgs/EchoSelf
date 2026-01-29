#!/bin/bash
# EchoSelf GPU Server 安装脚本 (EchoMimic V3)

set -e

echo "=========================================="
echo "EchoSelf GPU Server 安装 (EchoMimic V3)"
echo "=========================================="

# 检查 CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "错误: 未检测到 NVIDIA GPU 驱动"
    exit 1
fi

echo "检测到 GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 创建 conda 环境
echo ""
echo "[1/6] 创建 Python 环境..."
if ! command -v conda &> /dev/null; then
    echo "警告: 未检测到 conda，使用系统 Python"
    python3 -m venv venv
    source venv/bin/activate
else
    # 如果环境已存在，先删除
    conda env remove -n echoself-server -y 2>/dev/null || true
    conda create -n echoself-server python=3.10 -y
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate echoself-server
fi

# 安装 PyTorch (CUDA 12.1)
echo ""
echo "[2/6] 安装 PyTorch..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 安装依赖
echo ""
echo "[3/6] 安装依赖包..."
pip install -r requirements.txt

# 克隆 EchoMimic V3 源码
echo ""
echo "[4/6] 获取 EchoMimic V3 源码..."
if [ ! -d "echomimic_v3_src" ]; then
    git clone https://github.com/antgroup/echomimic_v3 echomimic_v3_src
    echo "✓ EchoMimic V3 源码克隆完成"
else
    echo "EchoMimic V3 源码已存在，跳过克隆"
    cd echomimic_v3_src && git pull && cd ..
fi

# 下载模型
echo ""
echo "[5/6] 下载模型..."
mkdir -p models/echomimic_v3

# 下载 EchoMimic V3 flash-pro 模型
if [ ! -f "models/echomimic_v3/transformer/diffusion_pytorch_model.safetensors" ]; then
    echo "下载 EchoMimic V3 flash-pro 模型..."

    # 下载 Wan2.1 基础模型
    echo "  [5.1] 下载 Wan2.1-Fun-V1.1-1.3B-InP 基础模型..."
    huggingface-cli download alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP \
        --local-dir models/echomimic_v3/Wan2.1-Fun-V1.1-1.3B-InP \
        --local-dir-use-symlinks False

    # 下载中文 wav2vec2 音频编码器
    echo "  [5.2] 下载 chinese-wav2vec2-base 音频编码器..."
    huggingface-cli download TencentGameMate/chinese-wav2vec2-base \
        --local-dir models/echomimic_v3/chinese-wav2vec2-base \
        --local-dir-use-symlinks False

    # 下载 EchoMimic V3 flash-pro transformer 权重
    echo "  [5.3] 下载 EchoMimic V3 flash-pro transformer..."
    mkdir -p models/echomimic_v3/transformer
    huggingface-cli download BadToBest/EchoMimicV3 \
        echomimicv3-flash-pro/transformer/diffusion_pytorch_model.safetensors \
        --local-dir models/echomimic_v3 \
        --local-dir-use-symlinks False

    # 移动文件到正确位置
    if [ -d "models/echomimic_v3/echomimicv3-flash-pro" ]; then
        mv models/echomimic_v3/echomimicv3-flash-pro/transformer/* models/echomimic_v3/transformer/
        rm -rf models/echomimic_v3/echomimicv3-flash-pro
    fi

    echo "✓ 模型下载完成"
else
    echo "模型已存在，跳过下载"
fi

# 创建目录
echo ""
echo "[6/6] 创建工作目录..."
mkdir -p temp outputs logs

# 复制配置文件
if [ -f "echomimic_v3_src/config/wan2.1/wan_civitai.yaml" ]; then
    mkdir -p config/wan2.1
    cp echomimic_v3_src/config/wan2.1/wan_civitai.yaml config/wan2.1/
fi

echo ""
echo "=========================================="
echo "安装完成！"
echo "=========================================="
echo ""
echo "模型位置:"
echo "  - 基础模型: models/echomimic_v3/Wan2.1-Fun-V1.1-1.3B-InP"
echo "  - 音频编码器: models/echomimic_v3/chinese-wav2vec2-base"
echo "  - Transformer: models/echomimic_v3/transformer/"
echo ""
echo "启动服务: bash start.sh"
echo "或手动启动: python api_server.py"
