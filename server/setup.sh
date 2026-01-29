#!/bin/bash
# EchoSelf GPU Server 安装脚本

set -e

echo "=========================================="
echo "EchoSelf GPU Server 安装"
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
echo "[1/5] 创建 Python 环境..."
if ! command -v conda &> /dev/null; then
    echo "警告: 未检测到 conda，使用系统 Python"
    python3 -m venv venv
    source venv/bin/activate
else
    conda create -n echoself-server python=3.10 -y
    conda activate echoself-server
fi

# 安装 PyTorch
echo ""
echo "[2/5] 安装 PyTorch..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# 安装依赖
echo ""
echo "[3/5] 安装依赖包..."
pip install -r requirements.txt

# 下载 ffmpeg-static
echo ""
echo "[4/5] 配置 FFmpeg..."
if [ ! -d "ffmpeg-4.4-amd64-static" ]; then
    wget https://www.johnvansickle.com/ffmpeg/old-releases/ffmpeg-4.4-amd64-static.tar.xz
    tar xf ffmpeg-4.4-amd64-static.tar.xz
    rm ffmpeg-4.4-amd64-static.tar.xz
fi
export FFMPEG_PATH=$(pwd)/ffmpeg-4.4-amd64-static

# 下载模型
echo ""
echo "[5/5] 下载模型..."
mkdir -p models

# 下载 EchoMimic V2
if [ ! -f "models/echomimic_v2/denoising_unet.pth" ]; then
    echo "下载 EchoMimic V2 模型..."
    pip install huggingface_hub
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='BadToBest/EchoMimicV2',
    local_dir='models/echomimic_v2',
    local_dir_use_symlinks=False
)
print('EchoMimic V2 下载完成')
"
fi

# 克隆 EchoMimic V2 源码
if [ ! -d "echomimic_v2_src" ]; then
    echo "克隆 EchoMimic V2 源码..."
    git clone https://github.com/antgroup/echomimic_v2 echomimic_v2_src
fi

# 下载 Fish Speech (可选，如果需要 TTS)
if [ ! -d "models/fish-speech-1.5" ]; then
    echo "下载 Fish Speech 模型..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='fishaudio/fish-speech-1.5',
    local_dir='models/fish-speech-1.5',
    local_dir_use_symlinks=False
)
print('Fish Speech 下载完成')
" || echo "Fish Speech 下载失败，TTS 将使用备选方案"
fi

# 创建目录
mkdir -p temp outputs logs

echo ""
echo "=========================================="
echo "安装完成！"
echo "=========================================="
echo ""
echo "启动服务: bash start.sh"
echo "或手动启动: python api_server.py"
