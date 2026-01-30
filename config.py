"""
EchoSelf - 数字分身系统配置文件
Configuration for the Digital Avatar Chat System
"""

import os
from pathlib import Path

# ============================================================
# 项目路径配置
# ============================================================
PROJECT_ROOT = Path(__file__).parent.absolute()
MODELS_DIR = PROJECT_ROOT / "models"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"
STATIC_DIR = PROJECT_ROOT / "static"

# 确保目录存在
for dir_path in [MODELS_DIR, UPLOADS_DIR, OUTPUTS_DIR, DATA_DIR, CHROMA_DB_DIR, STATIC_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================
# Ollama LLM 配置 (远程服务器)
# ============================================================
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://192.168.50.218:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
OLLAMA_TIMEOUT = 120  # 请求超时时间（秒）
OLLAMA_MAX_TOKENS = 2048  # 最大生成token数

# ============================================================
# 远程 GPU 服务配置 (Duix-Avatar)
# ============================================================
# 启用远程 GPU 服务（推荐，解决本地算力不足问题）
# 服务包含: TTS (Fish Speech), Video (Duix Avatar), ASR (FunASR)
USE_REMOTE_GPU = os.getenv("USE_REMOTE_GPU", "true").lower() == "true"
REMOTE_GPU_SERVER = os.getenv("REMOTE_GPU_SERVER", "http://192.168.50.218:8000")
REMOTE_GPU_TIMEOUT = 300  # 远程请求超时时间（秒），视频生成需要较长时间

# ============================================================
# TTS 声音克隆配置 (Fish Speech V1.5)
# ============================================================
# Fish Speech 模型路径 (需要手动下载)
FISH_SPEECH_MODEL_DIR = MODELS_DIR / "fish-speech-1.5"
FISH_SPEECH_CHECKPOINT = FISH_SPEECH_MODEL_DIR / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
FISH_SPEECH_VOCODER = FISH_SPEECH_MODEL_DIR / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"

# TTS 输出配置
TTS_SAMPLE_RATE = 44100
TTS_OUTPUT_FORMAT = "wav"

# 声音克隆配置
VOICE_CLONE_MIN_DURATION = 5  # 最小参考音频时长（秒）
VOICE_CLONE_MAX_DURATION = 60  # 最大参考音频时长（秒）

# ============================================================
# 数字人/Avatar 配置 (Duix-Avatar 或本地备用)
# ============================================================
ECHOMIMIC_MODEL_DIR = MODELS_DIR / "echomimic_v2"
ECHOMIMIC_PRETRAINED_DIR = ECHOMIMIC_MODEL_DIR / "pretrained_weights"

# 视频输出配置
AVATAR_VIDEO_FPS = 25
AVATAR_VIDEO_WIDTH = 512
AVATAR_VIDEO_HEIGHT = 512
AVATAR_VIDEO_CODEC = "libx264"
AVATAR_AUDIO_CODEC = "aac"

# 推理配置
AVATAR_INFERENCE_STEPS = 20  # 降低以加速生成
AVATAR_CFG_SCALE = 3.5  # classifier-free guidance scale

# ============================================================
# RAG 配置 (可选，用于处理大量聊天记录)
# ============================================================
RAG_ENABLED = True
RAG_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RAG_CHUNK_SIZE = 500  # 文本分块大小
RAG_CHUNK_OVERLAP = 50  # 分块重叠
RAG_TOP_K = 5  # 检索返回的最相关文档数

# ============================================================
# 性格/风格提取配置
# ============================================================
PERSONALITY_EXTRACT_MAX_SAMPLES = 20  # 用于few-shot的最大示例数
PERSONALITY_EXTRACT_MAX_CHARS = 5000  # 处理的最大聊天记录字符数

# ============================================================
# Gradio 界面配置
# ============================================================
GRADIO_SERVER_PORT = 7860
GRADIO_SERVER_NAME = "0.0.0.0"  # 允许外部访问
GRADIO_SHARE = False  # 是否创建公共链接
GRADIO_MAX_FILE_SIZE = "100MB"

# ============================================================
# 系统提示词模板
# ============================================================
SYSTEM_PROMPT_TEMPLATE = """你是一个数字分身AI助手，你需要模仿以下人物的说话风格和性格特点进行对话。

## 人物性格描述：
{personality_description}

## 说话风格示例：
{style_examples}

## 重要规则：
1. 保持该人物的语气、用词习惯和表达方式
2. 回复要自然、口语化，像真人聊天一样
3. 避免过于正式或机械的回复
4. 根据上下文适当使用该人物常用的口头禅或表达
5. 回复长度适中，通常1-3句话即可，除非话题需要详细说明

请开始对话，记住你现在就是这个人的数字分身。"""

# 默认性格描述（当没有提供聊天记录时使用）
DEFAULT_PERSONALITY = """
- 友善、热情、乐于助人
- 说话简洁明了，不啰嗦
- 偶尔会用一些网络用语
- 对技术话题感兴趣
"""

DEFAULT_STYLE_EXAMPLES = """
用户: 今天天气怎么样？
回复: 今天挺不错的，适合出门溜达~

用户: 你在干嘛？
回复: 刚忙完手头的事，现在歇会儿，你呢？

用户: 能帮我看看这个问题吗？
回复: 当然可以，发过来我看看。
"""

# ============================================================
# 日志配置
# ============================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ============================================================
# GPU/设备配置
# ============================================================
import torch

# 设备检测：CUDA > MPS (Apple Silicon) > CPU
if torch.cuda.is_available():
    DEVICE = "cuda"
    TORCH_DTYPE = torch.float16
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = "mps"  # Apple Silicon GPU
    TORCH_DTYPE = torch.float32  # MPS 对 float16 支持有限
else:
    DEVICE = "cpu"
    TORCH_DTYPE = torch.float32

# 打印配置信息
def print_config():
    """打印当前配置信息"""
    print("=" * 60)
    print("EchoSelf 数字分身系统 - 配置信息")
    print("=" * 60)
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"本地设备: {DEVICE}")
    print(f"Ollama 服务器: {OLLAMA_HOST}")
    print(f"Ollama 模型: {OLLAMA_MODEL}")
    print(f"远程 GPU 服务: {'启用' if USE_REMOTE_GPU else '禁用'}")
    if USE_REMOTE_GPU:
        print(f"远程 GPU 地址: {REMOTE_GPU_SERVER}")
    print(f"RAG 启用: {RAG_ENABLED}")
    print("=" * 60)

if __name__ == "__main__":
    print_config()
