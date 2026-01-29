# EchoSelf GPU Server 部署指南 (EchoMimic V3)

远程 GPU 服务器部署，提供 TTS 语音合成和 Avatar 视频生成 API。

**使用 EchoMimic V3 flash-pro 版本，相比 V2 有显著提升。**

## V3 vs V2 对比

| 特性 | V2 | V3 flash-pro |
|------|-----|--------------|
| 推理步数 | 30 步 | **8 步** |
| 显存需求 | ~16GB | **12GB** |
| 需要 Pose 数据 | ✅ | **❌** |
| 需要 Face Mask | ✅ | **❌** |
| 模型参数 | 较大 | **1.3B** |
| 速度 | 约 7 分钟/120帧 | **约 50 秒/120帧** |

## 服务器要求

- **GPU**: RTX 4090 (24GB) 或同等算力，12GB 显存即可
- **CUDA**: >= 12.1
- **Python**: 3.10
- **系统**: Ubuntu 22.04 (推荐)
- **内存**: >= 32GB
- **硬盘**: >= 50GB (模型文件)

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
- Python 3.10 环境
- PyTorch 2.5.1 (CUDA 12.1)
- EchoMimic V3 源码
- 模型文件：
  - Wan2.1-Fun-V1.1-1.3B-InP (基础模型)
  - chinese-wav2vec2-base (音频编码器)
  - flash-pro transformer 权重

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

### 预热模型（首次使用前建议调用）

```bash
curl -X POST http://192.168.50.218:8000/warmup
```

### TTS 语音合成

```bash
curl -X POST http://192.168.50.218:8000/tts \
  -F "text=你好，我是数字分身" \
  -F "voice=zh-CN-XiaoxiaoNeural" \
  -o output.wav
```

常用声音：
- `zh-CN-XiaoxiaoNeural` (女)
- `zh-CN-YunxiNeural` (男)
- `zh-CN-YunjianNeural` (男)

### Avatar 视频生成

```bash
curl -X POST http://192.168.50.218:8000/avatar \
  -F "audio=@audio.wav" \
  -F "reference_image=@photo.jpg" \
  -F "prompt=A person is speaking." \
  -F "max_frames=81" \
  -o output.mp4
```

参数说明：
- `audio`: 驱动音频文件
- `reference_image`: 参考人像图片
- `prompt`: 文本提示（可选）
- `max_frames`: 最大帧数（默认 81，约 3.2 秒）

### 获取服务信息

```bash
curl http://192.168.50.218:8000/info
```

## 配置说明

编辑 `config.yaml`:

```yaml
avatar:
  device: "cuda:0"  # GPU 设备
  num_inference_steps: 8  # 推理步数
  guidance_scale: 6.0     # 文本引导强度
  audio_guidance_scale: 3.0  # 音频引导强度 (1.8-2 最佳唇同步)
  enable_teacache: true   # TeaCache 加速
  max_frames: 81          # 最大帧数
```

**调优建议**：
- `audio_guidance_scale`: 1.8-2.0 获得最佳唇同步，降低可提升画质
- `guidance_scale`: 3-6 控制 prompt 跟随程度
- `num_inference_steps`: 8 步已足够，增加可略微提升质量

## 后台运行

```bash
# 使用 screen
screen -S echoself
bash start.sh
# Ctrl+A+D 退出

# 重新进入
screen -r echoself
```

## 故障排除

### 1. 模型加载失败

检查模型文件是否完整：
```bash
ls -la models/echomimic_v3/
# 应该有：
# - Wan2.1-Fun-V1.1-1.3B-InP/
# - chinese-wav2vec2-base/
# - transformer/diffusion_pytorch_model.safetensors
```

### 2. CUDA 版本不匹配

确保 CUDA >= 12.1：
```bash
nvidia-smi
nvcc --version
```

### 3. 显存不足

- 减少 `max_frames`
- 使用较小的图片分辨率

## 模型文件结构

```
models/echomimic_v3/
├── Wan2.1-Fun-V1.1-1.3B-InP/   # 基础模型 (~10GB)
│   ├── transformer/
│   ├── vae/
│   ├── tokenizer/
│   ├── text_encoder/
│   └── image_encoder/
├── chinese-wav2vec2-base/       # 音频编码器 (~400MB)
└── transformer/
    └── diffusion_pytorch_model.safetensors  # flash-pro 权重 (~5GB)
```
