"""
LLM 客户端模块 - 通过远程 Ollama API 调用大语言模型
LLM Client Module - Remote Ollama API Integration

功能：
1. 连接远程 Ollama 服务器
2. 发送对话请求并获取回复
3. 支持流式和非流式响应
4. 管理对话历史
"""

import requests
import json
import time
from typing import List, Dict, Optional, Generator, Any
from loguru import logger
import sys

# 配置 loguru
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>")


class OllamaClient:
    """
    Ollama API 客户端类
    用于与远程 Ollama 服务器通信
    """

    def __init__(
        self,
        host: str = "http://192.168.50.218:11434",
        model: str = "qwen2.5:7b-instruct",
        timeout: int = 120,
        max_retries: int = 3
    ):
        """
        初始化 Ollama 客户端

        Args:
            host: Ollama 服务器地址 (例如 "http://192.168.50.218:11434")
            model: 使用的模型名称 (例如 "qwen2.5:7b-instruct")
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
        """
        self.host = host.rstrip("/")  # 移除尾部斜杠
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

        # API 端点
        self.chat_endpoint = f"{self.host}/api/chat"
        self.generate_endpoint = f"{self.host}/api/generate"
        self.tags_endpoint = f"{self.host}/api/tags"

        logger.info(f"OllamaClient 初始化 - 服务器: {self.host}, 模型: {self.model}")

    def check_connection(self) -> bool:
        """
        检查与 Ollama 服务器的连接

        Returns:
            bool: 连接是否成功
        """
        try:
            response = requests.get(
                self.tags_endpoint,
                timeout=10
            )
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                logger.info(f"连接成功！可用模型: {model_names}")

                # 检查目标模型是否可用
                if not any(self.model in name for name in model_names):
                    logger.warning(f"目标模型 '{self.model}' 未在服务器上找到")
                    logger.warning(f"可用模型: {model_names}")
                    return True  # 连接成功，但模型可能不存在

                return True
            else:
                logger.error(f"连接失败，状态码: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError as e:
            logger.error(f"无法连接到 Ollama 服务器 {self.host}: {e}")
            return False
        except requests.exceptions.Timeout:
            logger.error(f"连接 Ollama 服务器超时")
            return False
        except Exception as e:
            logger.error(f"检查连接时发生错误: {e}")
            return False

    def generate_reply(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False
    ) -> str:
        """
        生成对话回复（非流式）

        Args:
            messages: 对话历史，格式为 [{"role": "user", "content": "..."}, ...]
            system_prompt: 系统提示词（可选，会自动添加到 messages 开头）
            temperature: 生成温度，控制随机性 (0.0-2.0)
            max_tokens: 最大生成 token 数
            stream: 是否使用流式响应

        Returns:
            str: 模型生成的回复文本
        """
        # 构建完整的消息列表
        full_messages = []

        # 添加系统提示词
        if system_prompt:
            full_messages.append({
                "role": "system",
                "content": system_prompt
            })

        # 添加对话历史
        full_messages.extend(messages)

        # 构建请求 payload
        payload = {
            "model": self.model,
            "messages": full_messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }

        logger.debug(f"发送请求到 {self.chat_endpoint}")
        logger.debug(f"消息数量: {len(full_messages)}")

        # 带重试的请求
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.chat_endpoint,
                    json=payload,
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    result = response.json()
                    reply = result.get("message", {}).get("content", "")
                    logger.info(f"成功获取回复 (长度: {len(reply)} 字符)")
                    return reply
                else:
                    logger.error(f"请求失败，状态码: {response.status_code}")
                    logger.error(f"响应内容: {response.text}")

            except requests.exceptions.Timeout:
                logger.warning(f"请求超时，尝试 {attempt + 1}/{self.max_retries}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避

            except requests.exceptions.ConnectionError as e:
                logger.error(f"连接错误: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)

            except json.JSONDecodeError as e:
                logger.error(f"JSON 解析错误: {e}")
                break

            except Exception as e:
                logger.error(f"未知错误: {e}")
                break

        # 所有重试都失败
        return "[错误] 无法获取模型回复，请检查 Ollama 服务器连接。"

    def generate_reply_stream(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> Generator[str, None, None]:
        """
        流式生成对话回复

        Args:
            messages: 对话历史
            system_prompt: 系统提示词
            temperature: 生成温度
            max_tokens: 最大 token 数

        Yields:
            str: 逐步生成的文本片段
        """
        # 构建消息
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        payload = {
            "model": self.model,
            "messages": full_messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }

        try:
            with requests.post(
                self.chat_endpoint,
                json=payload,
                stream=True,
                timeout=self.timeout
            ) as response:

                if response.status_code != 200:
                    yield f"[错误] 状态码: {response.status_code}"
                    return

                # 逐行读取流式响应
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            content = data.get("message", {}).get("content", "")
                            if content:
                                yield content

                            # 检查是否完成
                            if data.get("done", False):
                                break

                        except json.JSONDecodeError:
                            continue

        except requests.exceptions.Timeout:
            yield "[错误] 请求超时"
        except requests.exceptions.ConnectionError:
            yield "[错误] 无法连接到服务器"
        except Exception as e:
            yield f"[错误] {str(e)}"

    def create_chat_context(
        self,
        user_input: str,
        chat_history: List[Dict[str, str]],
        max_history: int = 10
    ) -> List[Dict[str, str]]:
        """
        创建对话上下文，包含历史消息和当前输入

        Args:
            user_input: 用户当前输入
            chat_history: 历史对话记录
            max_history: 保留的最大历史轮数

        Returns:
            List[Dict]: 格式化的消息列表
        """
        # 截取最近的历史记录
        recent_history = chat_history[-max_history * 2:] if chat_history else []

        # 添加当前用户输入
        messages = recent_history + [{"role": "user", "content": user_input}]

        return messages


class ConversationManager:
    """
    对话管理器
    管理多轮对话的历史记录和上下文
    """

    def __init__(
        self,
        llm_client: OllamaClient,
        system_prompt: str = "",
        max_history: int = 20
    ):
        """
        初始化对话管理器

        Args:
            llm_client: Ollama 客户端实例
            system_prompt: 系统提示词
            max_history: 保留的最大历史消息数
        """
        self.llm_client = llm_client
        self.system_prompt = system_prompt
        self.max_history = max_history
        self.history: List[Dict[str, str]] = []

    def chat(self, user_input: str) -> str:
        """
        进行一轮对话

        Args:
            user_input: 用户输入

        Returns:
            str: 模型回复
        """
        # 创建上下文
        messages = self.llm_client.create_chat_context(
            user_input,
            self.history,
            self.max_history // 2
        )

        # 获取回复
        reply = self.llm_client.generate_reply(
            messages,
            system_prompt=self.system_prompt
        )

        # 更新历史
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": reply})

        # 裁剪历史
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        return reply

    def reset(self):
        """清空对话历史"""
        self.history = []
        logger.info("对话历史已清空")

    def get_history(self) -> List[Dict[str, str]]:
        """获取对话历史"""
        return self.history.copy()

    def set_system_prompt(self, prompt: str):
        """设置系统提示词"""
        self.system_prompt = prompt
        logger.info("系统提示词已更新")


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    # 从配置文件导入（如果可用）
    try:
        from config import OLLAMA_HOST, OLLAMA_MODEL
    except ImportError:
        OLLAMA_HOST = "http://192.168.50.218:11434"
        OLLAMA_MODEL = "qwen2.5:7b-instruct"

    print("=" * 60)
    print("OllamaClient 测试")
    print("=" * 60)

    # 创建客户端
    client = OllamaClient(host=OLLAMA_HOST, model=OLLAMA_MODEL)

    # 测试连接
    print("\n1. 测试连接...")
    if client.check_connection():
        print("✓ 连接成功！")
    else:
        print("✗ 连接失败，请检查服务器地址和网络")
        exit(1)

    # 测试简单对话
    print("\n2. 测试简单对话...")
    messages = [{"role": "user", "content": "你好，请用一句话介绍你自己"}]
    reply = client.generate_reply(messages)
    print(f"问: 你好，请用一句话介绍你自己")
    print(f"答: {reply}")

    # 测试带系统提示词的对话
    print("\n3. 测试带性格设定的对话...")
    system_prompt = "你是一个活泼可爱的女孩，说话喜欢用表情符号，语气轻松愉快。"
    messages = [{"role": "user", "content": "今天天气真好"}]
    reply = client.generate_reply(messages, system_prompt=system_prompt)
    print(f"系统: {system_prompt}")
    print(f"问: 今天天气真好")
    print(f"答: {reply}")

    # 测试流式输出
    print("\n4. 测试流式输出...")
    print("问: 讲一个简短的笑话")
    print("答: ", end="", flush=True)
    messages = [{"role": "user", "content": "讲一个简短的笑话"}]
    for chunk in client.generate_reply_stream(messages):
        print(chunk, end="", flush=True)
    print()

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
