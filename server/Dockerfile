# EchoSelf API Server Dockerfile

FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY api_server.py .
COPY config.yaml .

# 创建目录
RUN mkdir -p temp outputs logs

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "api_server.py"]
