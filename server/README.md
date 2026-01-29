# EchoSelf GPU Server 部署指南

远程 GPU 服务器部署，提供 TTS 语音合成和 Avatar 视频生成 API。

## 服务器要求

- **GPU**: 2x RTX 4090 (或同等算力，>=24GB 显存)
- **CUDA**: >= 11.7
- **Python**: 3.10
- **系统**: Ubuntu 22.04 (推荐) 或 CentOS 7+
- **内存**: >= 32GB
- **硬盘**: >= 100GB (模型文件约 50GB)

## 快速部署

### 1. 复制文件到服务器

```bash
# 在本地执行
scp -r server/ user@192.168.50.218:/home/user/echoself-server/
```

### 2. SSH 到服务器

```bash
ssh user@192.168.50.218
cd /home/user/echoself-server
```

### 3. 运行安装脚本

```bash
# 给脚本添加执行权限
chmod +x setup.sh start.sh

# 运行安装（会安装依赖和下载模型，约需 30-60 分钟）
bash setup.sh
```

### 4. 启动服务

```bash
bash start.sh
```

服务启动后，访问：
- API 地址: `http://192.168.50.218:8000`
- 文档地址: `http://192.168.50.218:8000/docs`

## 手动安装（如果自动安装失败）

### 创建 Python 环境

```bash
conda create -n echoself-server python=3.10 -y
conda activate echoself-server
```

### 安装 PyTorch

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

### 安装依赖

```bash
pip install -r requirements.txt
```

### 下载 FFmpeg

```bash
wget https://www.johnvansickle.com/ffmpeg/old-releases/ffmpeg-4.4-amd64-static.tar.xz
tar xf ffmpeg-4.4-amd64-static.tar.xz
export FFMPEG_PATH=$(pwd)/ffmpeg-4.4-amd64-static
```

### 下载模型

```bash
mkdir -p models

# 下载 EchoMimic V2 模型（约 30GB）
pip install huggingface_hub
huggingface-cli download BadToBest/EchoMimicV2 --local-dir models/echomimic_v2

# 克隆 EchoMimic V2 源码
git clone https://github.com/antgroup/echomimic_v2 echomimic_v2_src
```

### 启动

```bash
export FFMPEG_PATH=$(pwd)/ffmpeg-4.4-amd64-static
export PYTHONPATH=$(pwd)/echomimic_v2_src:$PYTHONPATH
python api_server.py
```

## API 接口

### 健康检查

```bash
curl http://192.168.50.218:8000/health
```

### TTS 语音合成

```bash
curl -X POST http://192.168.50.218:8000/tts \
  -F "text=你好，我是数字分身" \
  -F "voice=zh-CN-XiaoxiaoNeural" \
  -o output.wav
```

### Avatar 视频生成

```bash
curl -X POST http://192.168.50.218:8000/avatar \
  -F "audio=@audio.wav" \
  -F "reference_image=@photo.jpg" \
  -F "max_frames=240" \
  -o output.mp4
```

## 配置说明

编辑 `config.yaml`:

```yaml
server:
  host: "0.0.0.0"
  port: 8000

tts:
  device: "cuda:0"  # TTS 使用第一张 GPU

avatar:
  device: "cuda:1"  # Avatar 使用第二张 GPU
  width: 768
  height: 768
  fps: 24
  steps: 30  # 推理步数，越多质量越好但越慢
```

## 后台运行

使用 screen 或 tmux 保持服务在后台运行：

```bash
# 使用 screen
screen -S echoself
bash start.sh
# Ctrl+A+D 退出 screen

# 重新进入
screen -r echoself
```

或使用 systemd 服务：

```bash
# 创建 /etc/systemd/system/echoself-gpu.service
sudo systemctl enable echoself-gpu
sudo systemctl start echoself-gpu
```

## 故障排除

### 1. CUDA 内存不足

- 减少 `max_frames` 参数
- 在 `config.yaml` 中启用 int8 量化

### 2. 模型加载失败

- 检查模型文件是否完整下载
- 确认 CUDA 版本兼容

### 3. FFmpeg 错误

- 确保 FFMPEG_PATH 环境变量正确设置
- 确认 ffmpeg 有执行权限

## GPU 分配说明

两张 4090 的推荐分配：

| GPU | 显存使用 | 用途 |
|-----|---------|------|
| cuda:0 | ~8GB | TTS (Fish Speech / edge-tts) |
| cuda:1 | ~16GB | Avatar (EchoMimic V2) |

总显存 48GB 完全够用，还有余量。
