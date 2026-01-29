# EchoSelf GPU Server 部署指南 (Duix-Avatar)

远程 GPU 服务器部署，提供 TTS 语音合成、语音识别和数字人视频生成 API。

**基于 [Duix-Avatar](https://github.com/duixcom/Duix-Avatar) Docker 方案，开箱即用。**

## Duix-Avatar 服务组件

| 服务 | 镜像 | 端口 | 功能 |
|------|------|------|------|
| TTS | guiji2025/fish-speech-ziming | 18180 | Fish Speech 语音合成/克隆 |
| ASR | guiji2025/fun-asr | 10095 | FunASR 语音识别 |
| Video | guiji2025/duix.avatar | 8383 | 数字人视频生成 |
| API | echoself-api | 8000 | 统一 API 网关 |

## 服务器要求

- **GPU**: RTX 4090 (24GB) 推荐，至少需要 16GB 显存
- **CUDA**: >= 11.0
- **Docker**: >= 20.10
- **docker-compose**: >= 1.29
- **NVIDIA Container Toolkit**: 已安装
- **系统**: Ubuntu 22.04 (推荐)
- **内存**: >= 64GB (推荐)
- **硬盘**: >= 100GB (Docker 镜像约 70GB)

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
chmod +x setup.sh start.sh
bash setup.sh
```

安装内容：
- 检查 Docker 和 NVIDIA Container Toolkit
- 创建数据目录 `/data/duix_avatar_data/`
- 拉取 Docker 镜像（约 70GB）
- 构建 API 网关镜像

### 4. 启动服务

```bash
bash start.sh
```

服务启动后：
- API 地址: `http://192.168.50.218:8000`
- 文档地址: `http://192.168.50.218:8000/docs`

## API 接口

### 健康检查

```bash
curl http://192.168.50.218:8000/health
```

返回各服务状态：
```json
{
  "status": "ok",
  "version": "3.0.0 (Duix-Avatar)",
  "services": {
    "tts": true,
    "video": true,
    "asr": true
  }
}
```

### TTS 语音合成

```bash
curl -X POST http://192.168.50.218:8000/tts \
  -F "text=你好，我是数字分身" \
  -o output.wav
```

使用克隆声音：
```bash
curl -X POST http://192.168.50.218:8000/tts \
  -F "text=你好，我是数字分身" \
  -F "speaker_id=my_voice" \
  -o output.wav
```

### 声音克隆

```bash
curl -X POST http://192.168.50.218:8000/clone-voice \
  -F "speaker_id=my_voice" \
  -F "reference_audio=@reference.wav"
```

参数：
- `speaker_id`: 说话人唯一标识
- `reference_audio`: 参考音频文件（建议 5-30 秒清晰语音）

克隆成功后，可在 `/tts` 接口中使用 `speaker_id` 参数。

### Avatar 视频生成

```bash
curl -X POST http://192.168.50.218:8000/avatar \
  -F "audio=@audio.wav" \
  -F "reference_video=@video.mp4" \
  -o output.mp4
```

参数：
- `audio`: 驱动音频文件
- `reference_video`: 参考视频/图片文件

### 语音识别 (ASR)

```bash
curl -X POST http://192.168.50.218:8000/asr \
  -F "audio=@audio.wav"
```

返回：
```json
{
  "status": "ok",
  "text": "识别的文本内容"
}
```

### 获取服务信息

```bash
curl http://192.168.50.218:8000/info
```

### 列出已注册说话人

```bash
curl http://192.168.50.218:8000/speakers
```

## 配置说明

编辑 `config.yaml`:

```yaml
server:
  host: "0.0.0.0"
  port: 8000

duix:
  tts_url: "http://127.0.0.1:18180"
  video_url: "http://127.0.0.1:8383"
  asr_url: "http://127.0.0.1:10095"

tts:
  format: "wav"
  topP: 0.7
  temperature: 0.7
  max_new_tokens: 1024
  chunk_length: 100
  repetition_penalty: 1.2

video:
  chaofen: 0           # 超分辨率
  watermark_switch: 0  # 水印开关
  pn: 1                # 并行数

paths:
  voice_data: "/data/duix_avatar_data/voice/data"
  video_data: "/data/duix_avatar_data/face2face"
  temp_dir: "./temp"
  output_dir: "./outputs"
```

## Docker 管理

```bash
# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f

# 查看特定服务日志
docker-compose logs -f duix-avatar-tts

# 重启服务
docker-compose restart

# 停止服务
docker-compose down

# 完全清理（包括数据卷）
docker-compose down -v
```

## 数据目录结构

```
/data/duix_avatar_data/
├── voice/data/          # TTS 声音数据
│   └── {speaker_id}.wav # 克隆的参考音频
└── face2face/           # 视频生成数据
    └── {request_id}/    # 临时文件
```

## 故障排除

### 1. Docker 镜像拉取失败

检查网络连接，或使用镜像加速：
```bash
# 配置 Docker 镜像加速
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json <<EOF
{
  "registry-mirrors": ["https://mirror.ccs.tencentyun.com"]
}
EOF
sudo systemctl restart docker
```

### 2. GPU 未识别

检查 NVIDIA Container Toolkit：
```bash
# 测试 GPU 访问
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

如果失败，重新安装 NVIDIA Container Toolkit：
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 3. 服务启动失败

查看详细日志：
```bash
docker-compose logs duix-avatar-tts
docker-compose logs duix-avatar-gen-video
docker-compose logs duix-avatar-asr
```

### 4. 显存不足

如果显存不足，可以只启动部分服务：
```bash
# 只启动 TTS 和 API
docker-compose up -d duix-avatar-tts echoself-api
```

### 5. 健康检查失败

等待服务完全启动（首次启动可能需要几分钟）：
```bash
# 持续检查健康状态
watch -n 5 'curl -s http://localhost:8000/health | jq'
```

## 性能调优

### 多 GPU 配置

编辑 `docker-compose.yml`，为不同服务分配不同 GPU：

```yaml
duix-avatar-tts:
  environment:
    - NVIDIA_VISIBLE_DEVICES=0

duix-avatar-gen-video:
  environment:
    - NVIDIA_VISIBLE_DEVICES=1
```

### 增加共享内存

如果视频生成出现问题，增加 `shm_size`：

```yaml
duix-avatar-gen-video:
  shm_size: '16g'  # 默认 8g
```
