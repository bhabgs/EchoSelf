#!/bin/bash
# EchoSelf GPU Server 启动脚本 (Duix-Avatar)

set -e

echo "=========================================="
echo "EchoSelf GPU Server (Duix-Avatar)"
echo "=========================================="
echo ""

# 检查 Docker
if ! command -v docker &> /dev/null; then
    echo "错误: 未安装 Docker"
    echo "请先运行 setup.sh 或手动安装 Docker"
    exit 1
fi

# 检查 docker-compose
COMPOSE_CMD="docker-compose"
if ! command -v docker-compose &> /dev/null; then
    if docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
    else
        echo "错误: 未安装 docker-compose"
        exit 1
    fi
fi

# 启动服务
echo "启动 Docker 服务..."
echo ""

$COMPOSE_CMD up -d

echo ""
echo "=========================================="
echo "服务已启动"
echo "=========================================="
echo ""
echo "服务端口:"
echo "  - API 网关:  http://localhost:8000"
echo "  - API 文档:  http://localhost:8000/docs"
echo "  - TTS:       http://localhost:18180 (内部)"
echo "  - Video:     http://localhost:8383  (内部)"
echo "  - ASR:       http://localhost:10095 (内部)"
echo ""
echo "常用命令:"
echo "  查看日志:     $COMPOSE_CMD logs -f"
echo "  查看状态:     $COMPOSE_CMD ps"
echo "  停止服务:     $COMPOSE_CMD down"
echo "  重启服务:     $COMPOSE_CMD restart"
echo ""
echo "健康检查:"
echo "  curl http://localhost:8000/health"
echo ""
