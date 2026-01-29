#!/bin/bash
# EchoSelf GPU Server 安装脚本 (Duix-Avatar)

set -e

echo "=========================================="
echo "EchoSelf GPU Server 安装 (Duix-Avatar)"
echo "=========================================="

# 检查 Docker
if ! command -v docker &> /dev/null; then
    echo "错误: 未安装 Docker"
    echo "请先安装 Docker: https://docs.docker.com/engine/install/"
    exit 1
fi

# 检查 docker-compose
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "错误: 未安装 docker-compose"
    echo "请先安装 docker-compose"
    exit 1
fi

# 检查 NVIDIA Docker
if ! docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi &> /dev/null; then
    echo "错误: NVIDIA Container Toolkit 未正确安装"
    echo "请参考: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    exit 1
fi

echo "✓ Docker 环境检查通过"

# 检查 GPU
echo ""
echo "检测到 GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 创建数据目录
echo ""
echo "[1/3] 创建数据目录..."
sudo mkdir -p /data/duix_avatar_data/voice/data
sudo mkdir -p /data/duix_avatar_data/face2face
sudo chmod -R 777 /data/duix_avatar_data

mkdir -p temp outputs logs

# 拉取 Docker 镜像
echo ""
echo "[2/3] 拉取 Docker 镜像 (约 70GB，请耐心等待)..."
echo "  - guiji2025/fish-speech-ziming (TTS)"
echo "  - guiji2025/fun-asr (ASR)"
echo "  - guiji2025/duix.avatar (Video)"

docker pull guiji2025/fish-speech-ziming
docker pull guiji2025/fun-asr
docker pull guiji2025/duix.avatar

echo "✓ 镜像拉取完成"

# 构建 API 网关镜像
echo ""
echo "[3/3] 构建 API 网关..."
docker build -t echoself-api:latest .

echo ""
echo "=========================================="
echo "安装完成！"
echo "=========================================="
echo ""
echo "启动服务: docker-compose up -d"
echo "查看日志: docker-compose logs -f"
echo "停止服务: docker-compose down"
echo ""
echo "API 地址: http://localhost:8000"
echo "API 文档: http://localhost:8000/docs"
