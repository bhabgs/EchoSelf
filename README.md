# EchoSelf - 数字分身聊天系统

一个基于 AI 的数字分身系统，可以根据用户的照片、语音和聊天记录，创建一个会模仿用户说话风格的数字人。

## 功能特点

- **数字人形象生成**：上传照片，生成可说话的数字人视频
- **声音克隆**：分析语音样本，合成与用户声音相似的语音
- **性格模仿**：从聊天记录中学习说话风格和性格特点
- **实时对话**：支持文字输入，数字人用克隆声音+视频回复

## 技术架构

```
用户输入
    │
    ▼
┌─────────────────┐
│  Gradio 前端    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│                   后端处理                       │
│  ┌─────────┐  ┌─────────┐  ┌─────────────────┐  │
│  │   LLM   │  │   TTS   │  │  Avatar 生成    │  │
│  │ (远程)  │  │ (本地)  │  │    (本地)       │  │
│  └────┬────┘  └────┬────┘  └────────┬────────┘  │
│       │            │                 │           │
│       ▼            ▼                 ▼           │
│    Ollama     Fish Speech      EchoMimic V2     │
│    qwen2.5       V1.5                           │
└─────────────────────────────────────────────────┘
```

## 项目结构

```
EchoSelf/
├── app.py                    # Gradio 主应用
├── config.py                 # 配置文件
├── setup.py                  # 安装设置脚本
├── run.sh                    # 快速启动脚本
├── requirements.txt          # Python 依赖
│
├── modules/                  # 核心模块
│   ├── __init__.py
│   ├── llm_client.py         # Ollama LLM 客户端
│   ├── voice_cloning.py      # 声音克隆 TTS
│   ├── avatar_generator.py   # 数字人视频生成
│   ├── personality_extractor.py  # 性格/风格提取
│   └── rag_engine.py         # RAG 检索增强
│
├── utils/                    # 工具函数
│   ├── __init__.py
│   ├── file_utils.py         # 文件处理
│   └── audio_utils.py        # 音频处理
│
├── models/                   # 模型文件目录
│   ├── fish-speech-1.5/      # TTS 模型
│   └── echomimic_v2/         # Avatar 模型
│
├── uploads/                  # 用户上传文件
│   ├── photos/
│   ├── voice_samples/
│   └── chat_history/
│
├── outputs/                  # 生成的输出文件
└── data/
    └── chroma_db/            # RAG 向量数据库
```

## 系统要求

### 硬件要求
- **GPU**: NVIDIA GPU，建议 16GB+ 显存（RTX 3090/4090 或更高）
- **内存**: 32GB+ RAM
- **存储**: 50GB+ 可用空间（用于模型和临时文件）

### 软件要求
- Python 3.10+
- CUDA 11.8+ / CUDA 12.x
- FFmpeg
- Git

### 远程服务
- Ollama 服务器（运行在 `192.168.50.218:11434`）

## 安装指南

### 1. 克隆项目

```bash
cd /your/project/path
git clone <repository_url> EchoSelf
cd EchoSelf
```

### 2. 创建虚拟环境

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# 或 venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. 安装系统依赖

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
```bash
choco install ffmpeg
# 或下载安装包: https://ffmpeg.org/download.html
```

### 5. 下载模型

运行设置脚本：
```bash
python setup.py
```

或手动下载：

```bash
# Fish Speech TTS 模型
huggingface-cli download fishaudio/fish-speech-1.5 --local-dir models/fish-speech-1.5

# EchoMimic V2 Avatar 模型
huggingface-cli download antgroup/echomimic_v2 --local-dir models/echomimic_v2

# 中国区可使用 ModelScope:
# pip install modelscope
# modelscope download fishaudio/fish-speech-1.5 --local_dir models/fish-speech-1.5
```

### 6. 配置 Ollama 服务器

确保远程 Ollama 服务器已启动并可访问：

```bash
# 在 Ollama 服务器上
ollama serve

# 拉取模型
ollama pull qwen2.5:7b-instruct
```

检查连接：
```bash
curl http://192.168.50.218:11434/api/tags
```

### 7. 修改配置（可选）

编辑 `config.py` 修改默认配置：

```python
OLLAMA_HOST = "http://192.168.50.218:11434"  # Ollama 服务器地址
OLLAMA_MODEL = "qwen2.5:7b-instruct"          # 使用的模型
```

或通过环境变量覆盖：

```bash
export OLLAMA_HOST="http://your-server:11434"
export OLLAMA_MODEL="llama3.2:3b"
```

## 启动应用

### 方式 1: 使用启动脚本

```bash
./run.sh
```

### 方式 2: 直接运行

```bash
python app.py
```

### 方式 3: 指定参数

```bash
OLLAMA_HOST=http://192.168.50.218:11434 python app.py
```

启动后访问: http://localhost:7860

## 使用指南

### Step 1: 系统初始化
1. 打开浏览器访问 `http://localhost:7860`
2. 在"系统设置"标签页点击"初始化系统"
3. 等待所有模块加载完成

### Step 2: 上传材料
1. **照片**: 上传 1-3 张正面清晰的照片（支持 jpg/png）
2. **语音样本**: 上传 2-5 段 5-60 秒的语音（支持 mp3/wav）
3. **聊天记录**: 上传聊天记录文件（支持 txt/json/csv）

### Step 3: 开始聊天
1. 切换到"开始聊天"标签页
2. 在输入框中输入消息
3. 等待数字人生成视频回复

## 聊天记录格式

### 格式 1: 简单文本 (.txt)
```
张三: 你好啊
李四: 你好！最近怎么样？
张三: 挺好的，在忙工作
李四: 加油加油~
```

### 格式 2: JSON (.json)
```json
[
  {"role": "user", "content": "你好"},
  {"role": "assistant", "content": "你好！有什么可以帮你的？"}
]
```

### 格式 3: 带时间戳
```
[2024-01-01 10:00] 张三: 早上好
[2024-01-01 10:01] 李四: 早！
```

## 常见问题

### Q1: 无法连接 Ollama 服务器

**症状**: "无法连接到 Ollama 服务器" 错误

**解决方案**:
1. 确认服务器地址正确
2. 检查服务器是否运行: `curl http://192.168.50.218:11434/api/tags`
3. 检查防火墙设置
4. Ollama 默认只监听 localhost，需要配置允许远程访问：
   ```bash
   # 在 Ollama 服务器上
   OLLAMA_HOST=0.0.0.0 ollama serve
   ```

### Q2: GPU 内存不足

**症状**: CUDA out of memory 错误

**解决方案**:
1. 减少推理步数: 在 `config.py` 中设置 `AVATAR_INFERENCE_STEPS = 10`
2. 降低分辨率: 设置 `AVATAR_VIDEO_WIDTH = 256`
3. 使用 CPU（较慢）: 设置 `DEVICE = "cpu"`

### Q3: 模型下载失败

**症状**: 无法从 HuggingFace 下载模型

**解决方案**:
1. 检查网络连接
2. 使用镜像:
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```
3. 使用 ModelScope（中国区）:
   ```bash
   pip install modelscope
   modelscope download <model_name> --local_dir <path>
   ```

### Q4: 视频生成失败

**症状**: 生成静态图片而非视频

**解决方案**:
1. 确认 EchoMimic 模型已正确下载
2. 检查 FFmpeg 是否安装: `ffmpeg -version`
3. 查看日志中的具体错误信息

### Q5: TTS 声音不像本人

**症状**: 合成的声音与参考音频差异较大

**解决方案**:
1. 使用更长、更清晰的参考音频（建议 30-60 秒）
2. 避免有背景噪音的音频
3. 多上传几段不同情绪的语音样本

### Q6: 响应速度慢

**症状**: 生成回复需要很长时间

**解决方案**:
1. 使用更小的 LLM 模型（如 `llama3.2:3b`）
2. 减少 Avatar 推理步数
3. 关闭 RAG 功能
4. 使用更强的 GPU

## API 参考

### Ollama API 调用示例

```python
import requests

OLLAMA_HOST = "http://192.168.50.218:11434"
OLLAMA_MODEL = "qwen2.5:7b-instruct"

def generate_reply(messages):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False
    }
    resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload)
    return resp.json()["message"]["content"]

# 使用
messages = [
    {"role": "system", "content": "你是一个友好的助手"},
    {"role": "user", "content": "你好"}
]
reply = generate_reply(messages)
print(reply)
```

## 后续优化建议

### 短期优化
1. [ ] 添加流式输出支持（边生成边播放）
2. [ ] 支持语音输入（STT）
3. [ ] 优化视频生成速度（缓存、批处理）
4. [ ] 添加更多 TTS 引擎选项

### 中期优化
1. [ ] 支持多人数字人切换
2. [ ] 添加表情和手势控制
3. [ ] 支持实时视频通话模式
4. [ ] 添加记忆系统（长期对话上下文）

### 长期优化
1. [ ] 支持全身数字人
2. [ ] 集成更先进的 Avatar 模型（如 EMO、VASA）
3. [ ] 支持多模态输入（图片、视频）
4. [ ] 分布式部署支持

## 技术栈

- **前端**: Gradio 4.x
- **LLM**: Ollama + Qwen2.5
- **TTS**: Fish Speech V1.5 / edge-tts (备选)
- **Avatar**: EchoMimic V2
- **RAG**: ChromaDB + sentence-transformers
- **音视频处理**: FFmpeg, librosa, moviepy

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

## 致谢

- [Ollama](https://github.com/ollama/ollama)
- [Fish Speech](https://github.com/fishaudio/fish-speech)
- [EchoMimic](https://github.com/antgroup/echomimic_v2)
- [Gradio](https://github.com/gradio-app/gradio)
