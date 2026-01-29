你现在是一个资深全栈AI工程师 + AI数字人系统架构师，精通Python、PyTorch、Gradio、FastAPI、语音处理、LLM API调用、唇同步技术。

任务：为我完整开发一个“数字分身/数字人聊天系统”的后端+前端最小可行产品（MVP），目标是：

输入：
- 一张或多张用户照片（用于生成数字人形象）
- 几段用户语音样本（5-60秒，用于声音克隆）
- 用户提供的聊天记录文本文件（.txt 或 .json，用于提取说话风格/性格）

输出：
- 一个可视化数字人（说话时有唇部同步、头部轻微动作，最好半身）
- 支持实时文字聊天（用户打字 → 数字人用克隆声音+形象回复）
- 未来可扩展语音输入/输出

关键前提：
- LLM 部分**已经**在另一台服务器上用 Ollama 部署好（远程可访问），API 地址是：http://192.168.50.218:11434 （端口默认11434）
- 使用的 Ollama 模型名称是：qwen2.5:7b-instruct （或你实际用的模型名，比如 llama3.2:3b、deepseek-r1 等，请在代码中用变量 OLLAMA_MODEL = "qwen2.5:7b-instruct" 方便修改）
- 不要在代码里包含 ollama serve / ollama pull / ollama run 等本地命令，所有 LLM 调用都通过 HTTP API 走远程（用 requests 或 ollama python 客户端）
- Ollama API 文档参考：https://github.com/ollama/ollama/blob/main/docs/api.md （/api/chat 或 /api/generate 端点）

核心技术栈要求（2026年主流开源方案，必须本地优先运行 avatar 和 TTS，LLM 走远程）：
1. 形象 + 唇同步 + 头部/半身驱动 → 使用 antgroup/echomimic_v2 （或最新 echomimic_v3，如果已发布且更强，GitHub: https://github.com/antgroup/echomimic_v3），支持单张图+音频驱动半身数字人
2. 声音克隆TTS → 优先 CosyVoice2 / Fish Speech V1.5 / Qwen3-TTS（选中文效果最好、克隆最自然的那个，推荐 Fish Speech V1.5 或 Qwen3-TTS 根据2026评测）
3. 对话大脑（LLM + 性格拟合） → 通过远程 Ollama API 调用已部署的模型，用 system prompt + 从聊天记录提取的少量示例来模仿性格；如果记录很多，可简单用 RAG（Chroma + sentence-transformers）
4. 前端界面 → Gradio（最快上手），支持上传照片/音频/文本，显示数字人视频流 + 聊天框
5. 整体流程：
   用户上传材料 → 后台一次性处理（提取声音 embedding / 生成性格 prompt / 预生成数字人参考特征）
   → 进入聊天界面
   用户输入文字 → 调用远程 Ollama API 生成回复文字（带 system prompt 模仿性格） → TTS(克隆声) 生成音频 → EchoMimic 用音频+参考图/特征 生成带唇动的视频片段 → 前端实时播放

要求：
- 代码结构清晰：模块化（分开 voice_cloning.py、llm_client.py、avatar_generator.py、app.py 等）
- 支持本地运行 avatar 和 TTS（假设用户有至少 16-24GB 显存的NVIDIA卡）
- LLM 调用使用 requests.post 到 http://OLLAMA_HOST:11434/api/chat （推荐 chat 端点，支持 messages 格式）
- 示例 LLM 调用代码：
  import requests
  OLLAMA_HOST = "http://192.168.x.x:11434"  # 用户替换成实际地址
  OLLAMA_MODEL = "qwen2.5:7b-instruct"
  def generate_reply(messages):
      payload = {"model": OLLAMA_MODEL, "messages": messages, "stream": False}
      resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload)
      return resp.json()["message"]["content"]
- 处理好依赖安装（给出 requirements.txt）
- 包含错误处理、加载动画、进度条
- 聊天历史保存在 Gradio session state
- 优先实现“文字输入 → 数字人说话视频输出”的核心链路
- 如果生成慢，先做非流式（完整回复后再生成视频），注释说明如何改流式（stream=True + yield）
- 在每个主要函数/模块加详细注释

输出格式：
1. 先给出项目目录结构
2. 给出完整的 requirements.txt
3. 完整代码（从 main app.py 开始，其他模块分开写）
4. 运行命令 & 可能遇到的坑 + 解决方案（尤其是远程 Ollama 连接、跨服务器网络、防火墙）
5. 后续优化建议列表

现在就开始写代码，不要只写伪代码，要写可运行的真实代码（可以先写核心链路，逐步完整）。
