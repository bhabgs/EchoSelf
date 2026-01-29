"""
EchoSelf - 数字分身聊天系统
Digital Avatar Chat System - Main Application

基于 Gradio 的前端界面，整合所有模块实现完整的数字人聊天系统

核心流程：
1. 用户上传照片、语音样本、聊天记录
2. 系统处理材料：提取声音特征、分析性格、生成数字人参考
3. 进入聊天界面
4. 用户输入文字 → LLM生成回复 → TTS合成语音 → Avatar生成视频 → 播放
"""

import os
import sys
import time
import json
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import gradio as gr
from loguru import logger

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# 导入配置
from config import (
    OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_TIMEOUT,
    UPLOADS_DIR, OUTPUTS_DIR, DATA_DIR,
    SYSTEM_PROMPT_TEMPLATE, DEFAULT_PERSONALITY, DEFAULT_STYLE_EXAMPLES,
    RAG_ENABLED, DEVICE, GRADIO_SERVER_PORT, GRADIO_SERVER_NAME
)

# 导入模块
from modules.llm_client import OllamaClient, ConversationManager
from modules.voice_cloning import VoiceCloner
from modules.avatar_generator import AvatarGenerator
from modules.personality_extractor import PersonalityExtractor
from modules.rag_engine import RAGEngine

from utils.file_utils import ensure_dir, save_uploaded_file, validate_file_type
from utils.audio_utils import get_audio_duration, get_audio_info

# 配置日志
logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>")


# ============================================================
# 全局状态管理
# ============================================================
class EchoSelfState:
    """管理应用全局状态"""

    def __init__(self):
        self.llm_client: Optional[OllamaClient] = None
        self.voice_cloner: Optional[VoiceCloner] = None
        self.avatar_generator: Optional[AvatarGenerator] = None
        self.personality_extractor: Optional[PersonalityExtractor] = None
        self.rag_engine: Optional[RAGEngine] = None

        self.system_prompt: str = ""
        self.is_initialized: bool = False
        self.user_id: str = "default"

        # 聊天历史
        self.chat_history: List[Dict[str, str]] = []

    def initialize(self) -> Tuple[bool, str]:
        """初始化所有模块"""
        messages = []

        try:
            # 1. 初始化 LLM 客户端
            logger.info("初始化 LLM 客户端...")
            self.llm_client = OllamaClient(
                host=OLLAMA_HOST,
                model=OLLAMA_MODEL,
                timeout=OLLAMA_TIMEOUT
            )

            if self.llm_client.check_connection():
                messages.append("LLM 客户端连接成功")
            else:
                messages.append("警告: LLM 连接失败，请检查 Ollama 服务器")

            # 2. 初始化声音克隆器
            logger.info("初始化声音克隆器...")
            self.voice_cloner = VoiceCloner(device=DEVICE)
            if self.voice_cloner.load_model():
                messages.append("TTS 模型加载成功")
            else:
                messages.append("警告: TTS 模型加载失败，将使用备选方案")

            # 3. 初始化 Avatar 生成器
            logger.info("初始化 Avatar 生成器...")
            self.avatar_generator = AvatarGenerator(device=DEVICE)
            if self.avatar_generator.load_model():
                messages.append("Avatar 模型加载成功")
            else:
                messages.append("警告: Avatar 模型加载失败，将使用备选方案")

            # 4. 初始化性格提取器
            self.personality_extractor = PersonalityExtractor()
            messages.append("性格提取器就绪")

            # 5. 初始化 RAG 引擎（可选）
            if RAG_ENABLED:
                logger.info("初始化 RAG 引擎...")
                self.rag_engine = RAGEngine(persist_directory=DATA_DIR / "chroma_db")
                if self.rag_engine.initialize():
                    messages.append("RAG 引擎初始化成功")
                else:
                    messages.append("警告: RAG 引擎初始化失败")

            self.is_initialized = True
            return True, "\n".join(messages)

        except Exception as e:
            logger.error(f"初始化失败: {e}")
            return False, f"初始化失败: {str(e)}"


# 全局状态实例
app_state = EchoSelfState()


# ============================================================
# Gradio 界面函数
# ============================================================

def initialize_system(progress=gr.Progress()):
    """初始化系统（带进度条）"""
    progress(0, desc="正在初始化系统...")

    progress(0.2, desc="连接 LLM 服务器...")
    success, message = app_state.initialize()

    progress(1.0, desc="初始化完成")

    status = "系统已就绪" if success else "系统初始化部分失败"

    return (
        gr.update(value=f"{status}\n\n{message}"),
        gr.update(interactive=True) if success else gr.update(interactive=False)
    )


def process_uploaded_materials(
    photos: List,
    voice_samples: List,
    chat_history_file,
    user_name: str,
    progress=gr.Progress()
) -> Tuple[str, str]:
    """
    处理用户上传的材料

    Args:
        photos: 用户照片列表
        voice_samples: 语音样本列表
        chat_history_file: 聊天记录文件
        user_name: 用户名/标识符

    Returns:
        Tuple[str, str]: (处理状态, 生成的系统提示词)
    """
    if not app_state.is_initialized:
        return "错误: 系统未初始化", ""

    messages = []
    app_state.user_id = user_name or "default"

    try:
        # 1. 处理照片
        progress(0.1, desc="处理照片...")
        if photos and len(photos) > 0:
            # 使用第一张照片作为参考
            photo_path = photos[0]
            if hasattr(photo_path, 'name'):
                photo_path = photo_path.name

            saved_photo, _ = save_uploaded_file(
                photo_path,
                UPLOADS_DIR / "photos",
                filename=f"{app_state.user_id}_avatar.jpg"
            )

            if app_state.avatar_generator.set_reference_image(saved_photo, app_state.user_id):
                messages.append(f"参考照片已设置")
            else:
                messages.append("警告: 照片处理失败")
        else:
            messages.append("未上传照片，将使用默认形象")

        # 2. 处理语音样本
        progress(0.3, desc="处理语音样本...")
        if voice_samples and len(voice_samples) > 0:
            for i, sample in enumerate(voice_samples):
                sample_path = sample
                if hasattr(sample, 'name'):
                    sample_path = sample.name

                # 检查音频时长
                try:
                    duration = get_audio_duration(sample_path)
                    if duration < 5:
                        messages.append(f"警告: 语音样本 {i+1} 时长不足5秒")
                        continue
                except Exception as e:
                    messages.append(f"警告: 无法读取语音样本 {i+1}")
                    continue

                # 保存并提取特征
                saved_sample, _ = save_uploaded_file(
                    sample_path,
                    UPLOADS_DIR / "voice_samples",
                    filename=f"{app_state.user_id}_voice_{i}.wav"
                )

                app_state.voice_cloner.extract_speaker_embedding(
                    saved_sample,
                    speaker_id=app_state.user_id
                )

            messages.append(f"已处理 {len(voice_samples)} 个语音样本")
        else:
            messages.append("未上传语音样本，将使用默认声音")

        # 3. 处理聊天记录
        progress(0.5, desc="分析聊天记录...")
        if chat_history_file is not None:
            file_path = chat_history_file
            if hasattr(chat_history_file, 'name'):
                file_path = chat_history_file.name

            # 保存文件
            saved_chat, _ = save_uploaded_file(
                file_path,
                UPLOADS_DIR / "chat_history",
                filename=f"{app_state.user_id}_chat_history{Path(file_path).suffix}"
            )

            # 加载和分析
            if app_state.personality_extractor.load_chat_history(saved_chat, user_name):
                analysis = app_state.personality_extractor.analyze()
                app_state.system_prompt = app_state.personality_extractor.generate_system_prompt()
                messages.append(f"聊天记录分析完成，提取了 {len(analysis.get('traits', []))} 个性格特点")

                # 添加到 RAG（如果启用）
                if app_state.rag_engine:
                    progress(0.7, desc="建立对话索引...")
                    app_state.rag_engine.add_chat_history(
                        app_state.personality_extractor.messages
                    )
                    messages.append("RAG 索引已建立")
            else:
                messages.append("警告: 聊天记录解析失败")
                app_state.system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
                    personality_description=DEFAULT_PERSONALITY,
                    style_examples=DEFAULT_STYLE_EXAMPLES
                )
        else:
            messages.append("未上传聊天记录，使用默认性格设定")
            app_state.system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
                personality_description=DEFAULT_PERSONALITY,
                style_examples=DEFAULT_STYLE_EXAMPLES
            )

        progress(1.0, desc="处理完成")

        status = "材料处理完成\n\n" + "\n".join(f"- {m}" for m in messages)
        return status, app_state.system_prompt

    except Exception as e:
        logger.error(f"处理材料时发生错误: {e}")
        return f"处理失败: {str(e)}", ""


def chat_with_avatar(
    user_message: str,
    chat_history: List,
    progress=gr.Progress()
) -> Tuple[List, str, Optional[str]]:
    """
    与数字人聊天

    Args:
        user_message: 用户消息
        chat_history: Gradio 聊天历史格式

    Returns:
        Tuple[List, str, Optional[str]]: (更新的聊天历史, 清空的输入框, 视频路径)
    """
    if not user_message.strip():
        return chat_history, "", None

    if not app_state.is_initialized:
        chat_history = chat_history or []
        chat_history.append((user_message, "系统未初始化，请先完成初始化"))
        return chat_history, "", None

    try:
        # 1. 获取 LLM 回复
        progress(0.2, desc="正在思考...")

        # 构建消息历史
        messages = []
        if chat_history:
            for user_msg, assistant_msg in chat_history:
                messages.append({"role": "user", "content": user_msg})
                if assistant_msg:
                    messages.append({"role": "assistant", "content": assistant_msg})

        messages.append({"role": "user", "content": user_message})

        # 如果启用 RAG，获取相关上下文
        enhanced_prompt = app_state.system_prompt
        if app_state.rag_engine:
            context = app_state.rag_engine.get_relevant_context(user_message, top_k=3)
            if context:
                enhanced_prompt += f"\n\n## 相关历史对话：\n{context}"

        # 调用 LLM
        reply = app_state.llm_client.generate_reply(
            messages,
            system_prompt=enhanced_prompt
        )

        progress(0.4, desc="生成语音...")

        # 2. TTS 合成语音
        audio_path = app_state.voice_cloner.synthesize(
            reply,
            speaker_id=app_state.user_id,
            output_path=OUTPUTS_DIR / f"reply_{int(time.time())}.wav"
        )

        # 3. 生成数字人视频
        video_path = None
        if audio_path:
            progress(0.6, desc="生成视频...")

            if app_state.user_id in app_state.avatar_generator.list_avatars():
                video_path = app_state.avatar_generator.generate_video(
                    audio_path,
                    avatar_id=app_state.user_id,
                    output_path=OUTPUTS_DIR / f"video_{int(time.time())}.mp4"
                )

        progress(1.0, desc="完成")

        # 更新聊天历史
        chat_history = chat_history or []
        chat_history.append((user_message, reply))

        # 保存到内部历史
        app_state.chat_history.append({"role": "user", "content": user_message})
        app_state.chat_history.append({"role": "assistant", "content": reply})

        return chat_history, "", video_path

    except Exception as e:
        logger.error(f"聊天时发生错误: {e}")
        chat_history = chat_history or []
        chat_history.append((user_message, f"抱歉，发生了错误: {str(e)}"))
        return chat_history, "", None


def clear_chat():
    """清空聊天历史"""
    app_state.chat_history = []
    return [], None


def update_system_prompt(new_prompt: str) -> str:
    """更新系统提示词"""
    app_state.system_prompt = new_prompt
    return "系统提示词已更新"


def test_llm_connection() -> str:
    """测试 LLM 连接"""
    if not app_state.llm_client:
        return "LLM 客户端未初始化"

    if app_state.llm_client.check_connection():
        # 发送测试消息
        test_reply = app_state.llm_client.generate_reply(
            [{"role": "user", "content": "你好，请用一句话回复"}]
        )
        return f"连接成功\n\n测试回复: {test_reply}"
    else:
        return f"连接失败\n\n服务器地址: {OLLAMA_HOST}\n模型: {OLLAMA_MODEL}"


def test_tts(text: str) -> Optional[str]:
    """测试 TTS"""
    if not app_state.voice_cloner:
        return None

    audio_path = app_state.voice_cloner.synthesize(
        text or "这是一段测试语音",
        speaker_id=app_state.user_id,
        output_path=OUTPUTS_DIR / "test_tts.wav"
    )
    return audio_path


def export_chat_history() -> Optional[str]:
    """导出聊天历史"""
    if not app_state.chat_history:
        return None

    export_path = OUTPUTS_DIR / f"chat_export_{int(time.time())}.json"
    with open(export_path, 'w', encoding='utf-8') as f:
        json.dump(app_state.chat_history, f, ensure_ascii=False, indent=2)

    return str(export_path)


# ============================================================
# 构建 Gradio 界面
# ============================================================

def create_interface():
    """创建 Gradio 界面"""

    # Gradio 6.0+ 兼容：theme 和 css 移到 launch()
    with gr.Blocks(
        title="EchoSelf - 数字分身系统"
    ) as demo:

        # 标题
        gr.Markdown("""
        # EchoSelf - 数字分身聊天系统

        上传您的照片、语音和聊天记录，创建一个会模仿您说话风格的数字分身！

        ---
        """)

        with gr.Tabs():
            # ==================== Tab 1: 系统设置 ====================
            with gr.TabItem("1. 系统设置"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### 系统初始化")
                        init_btn = gr.Button("初始化系统", variant="primary", size="lg")
                        init_status = gr.Textbox(
                            label="初始化状态",
                            lines=6,
                            interactive=False,
                            value="点击上方按钮开始初始化..."
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("### 连接测试")
                        test_llm_btn = gr.Button("测试 LLM 连接")
                        llm_test_result = gr.Textbox(
                            label="测试结果",
                            lines=4,
                            interactive=False
                        )

                gr.Markdown(f"""
                ### 当前配置
                - **Ollama 服务器**: `{OLLAMA_HOST}`
                - **模型**: `{OLLAMA_MODEL}`
                - **设备**: `{DEVICE}`
                - **RAG 启用**: `{RAG_ENABLED}`
                """)

            # ==================== Tab 2: 材料上传 ====================
            with gr.TabItem("2. 上传材料"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 上传您的照片")
                        photo_input = gr.File(
                            label="照片（支持 jpg/png，建议正面清晰照片）",
                            file_count="multiple",
                            file_types=["image"]
                        )
                        gr.Markdown("*建议上传1-3张正面清晰的照片*")

                    with gr.Column():
                        gr.Markdown("### 上传语音样本")
                        voice_input = gr.File(
                            label="语音样本（5-60秒，支持 mp3/wav）",
                            file_count="multiple",
                            file_types=["audio"]
                        )
                        gr.Markdown("*建议上传2-5段自然说话的语音*")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 上传聊天记录")
                        chat_history_input = gr.File(
                            label="聊天记录（支持 txt/json）",
                            file_count="single",
                            file_types=[".txt", ".json", ".csv"]
                        )
                        gr.Markdown("*用于分析您的说话风格和性格特点*")

                    with gr.Column():
                        gr.Markdown("### 用户信息")
                        user_name_input = gr.Textbox(
                            label="您的名字/昵称",
                            placeholder="用于在聊天记录中识别您的消息",
                            value=""
                        )

                with gr.Row():
                    process_btn = gr.Button("开始处理材料", variant="primary", size="lg")

                process_status = gr.Textbox(
                    label="处理状态",
                    lines=8,
                    interactive=False
                )

                with gr.Accordion("生成的系统提示词（可编辑）", open=False):
                    system_prompt_display = gr.Textbox(
                        label="系统提示词",
                        lines=15,
                        interactive=True
                    )
                    update_prompt_btn = gr.Button("更新提示词")
                    prompt_update_status = gr.Textbox(
                        label="",
                        lines=1,
                        interactive=False
                    )

            # ==================== Tab 3: 与数字人聊天 ====================
            with gr.TabItem("3. 开始聊天"):
                with gr.Row():
                    # 左侧：视频显示
                    with gr.Column(scale=1):
                        gr.Markdown("### 数字人形象")
                        avatar_video = gr.Video(
                            label="",
                            autoplay=True,
                            elem_classes=["video-container"]
                        )

                    # 右侧：聊天界面
                    with gr.Column(scale=1):
                        gr.Markdown("### 对话")
                        chatbot = gr.Chatbot(
                            label="",
                            height=400
                        )

                        with gr.Row():
                            user_input = gr.Textbox(
                                label="",
                                placeholder="输入消息...",
                                scale=4,
                                container=False
                            )
                            send_btn = gr.Button("发送", variant="primary", scale=1)

                        with gr.Row():
                            clear_btn = gr.Button("清空对话")
                            export_btn = gr.Button("导出记录")

            # ==================== Tab 4: 高级设置 ====================
            with gr.TabItem("4. 高级设置"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### TTS 测试")
                        tts_test_text = gr.Textbox(
                            label="测试文本",
                            value="你好，我是你的数字分身，很高兴见到你！"
                        )
                        tts_test_btn = gr.Button("测试语音合成")
                        tts_test_audio = gr.Audio(label="合成结果")

                    with gr.Column():
                        gr.Markdown("### 参数调整")
                        temperature_slider = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=0.7,
                            step=0.1,
                            label="LLM 温度（越高越随机）"
                        )
                        max_tokens_slider = gr.Slider(
                            minimum=50,
                            maximum=2048,
                            value=512,
                            step=50,
                            label="最大回复长度"
                        )

                gr.Markdown("""
                ### 使用说明

                1. **系统设置**：首先点击"初始化系统"按钮，等待所有模块加载完成
                2. **上传材料**：
                   - 上传1-3张正面清晰的照片（用于生成数字人形象）
                   - 上传2-5段5-60秒的语音样本（用于声音克隆）
                   - 上传聊天记录文件（用于学习说话风格）
                3. **开始聊天**：在聊天框中输入消息，数字人会用您的声音和风格回复

                ### 注意事项

                - 首次生成视频可能较慢，请耐心等待
                - 如果没有上传照片，将使用静态图片
                - 如果没有上传语音样本，将使用默认声音
                - 聊天记录文件支持 txt、json、csv 格式
                """)

        # ==================== 事件绑定 ====================

        # 初始化
        init_btn.click(
            fn=initialize_system,
            outputs=[init_status, process_btn]
        )

        # 测试 LLM
        test_llm_btn.click(
            fn=test_llm_connection,
            outputs=llm_test_result
        )

        # 处理材料
        process_btn.click(
            fn=process_uploaded_materials,
            inputs=[photo_input, voice_input, chat_history_input, user_name_input],
            outputs=[process_status, system_prompt_display]
        )

        # 更新提示词
        update_prompt_btn.click(
            fn=update_system_prompt,
            inputs=[system_prompt_display],
            outputs=[prompt_update_status]
        )

        # 聊天
        send_btn.click(
            fn=chat_with_avatar,
            inputs=[user_input, chatbot],
            outputs=[chatbot, user_input, avatar_video]
        )

        user_input.submit(
            fn=chat_with_avatar,
            inputs=[user_input, chatbot],
            outputs=[chatbot, user_input, avatar_video]
        )

        # 清空对话
        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot, avatar_video]
        )

        # 导出记录
        export_btn.click(
            fn=export_chat_history,
            outputs=gr.File(label="导出文件")
        )

        # TTS 测试
        tts_test_btn.click(
            fn=test_tts,
            inputs=[tts_test_text],
            outputs=[tts_test_audio]
        )

    return demo


# ============================================================
# 主程序入口
# ============================================================

def main():
    """主程序入口"""
    print("=" * 60)
    print("  EchoSelf - 数字分身聊天系统")
    print("=" * 60)
    print(f"  Ollama 服务器: {OLLAMA_HOST}")
    print(f"  模型: {OLLAMA_MODEL}")
    print(f"  设备: {DEVICE}")
    print("=" * 60)

    # 确保必要目录存在
    ensure_dir(UPLOADS_DIR)
    ensure_dir(OUTPUTS_DIR)
    ensure_dir(DATA_DIR)

    # 创建并启动界面
    demo = create_interface()

    demo.launch(
        server_name=GRADIO_SERVER_NAME,
        server_port=GRADIO_SERVER_PORT,
        share=False,
        show_error=True,
        quiet=False
    )


if __name__ == "__main__":
    main()
