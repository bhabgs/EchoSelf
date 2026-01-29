"""
远程 GPU 服务客户端

用于调用远程服务器上的 TTS 和 Avatar 生成 API
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, Union
import requests
from loguru import logger


class RemoteGPUClient:
    """
    远程 GPU 服务客户端

    用于调用远程服务器上的 TTS 语音合成和 Avatar 视频生成 API
    """

    def __init__(
        self,
        server_url: str = "http://192.168.50.218:8000",
        timeout: int = 300  # 5 分钟超时（视频生成较慢）
    ):
        """
        初始化客户端

        Args:
            server_url: 远程服务器地址
            timeout: 请求超时时间（秒）
        """
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        self.is_available = False

        logger.info(f"RemoteGPUClient 初始化 - 服务器: {self.server_url}")

    def check_connection(self) -> bool:
        """
        检查远程服务器连接

        Returns:
            bool: 是否可用
        """
        try:
            response = requests.get(
                f"{self.server_url}/health",
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                self.is_available = data.get('status') == 'ok'

                logger.info(f"远程 GPU 服务状态:")
                logger.info(f"  - GPU 可用: {data.get('gpu_available')}")
                logger.info(f"  - GPU 数量: {data.get('gpu_count')}")
                logger.info(f"  - Avatar 已加载: {data.get('avatar_loaded')}")
                logger.info(f"  - TTS 已加载: {data.get('tts_loaded')}")

                return self.is_available
            else:
                logger.error(f"服务器返回错误: {response.status_code}")
                return False

        except requests.exceptions.ConnectionError:
            logger.error(f"无法连接到远程服务器: {self.server_url}")
            return False
        except Exception as e:
            logger.error(f"检查连接时出错: {e}")
            return False

    def synthesize_speech(
        self,
        text: str,
        output_path: Optional[Union[str, Path]] = None,
        reference_audio: Optional[Union[str, Path]] = None,
        voice: str = "zh-CN-XiaoxiaoNeural"
    ) -> Optional[str]:
        """
        调用远程 TTS 服务合成语音

        Args:
            text: 要合成的文本
            output_path: 输出音频路径（可选，不指定则自动生成）
            reference_audio: 参考音频路径（用于声音克隆）
            voice: edge-tts 声音名称

        Returns:
            生成的音频文件路径，失败返回 None
        """
        if not text:
            logger.error("文本不能为空")
            return None

        # 准备输出路径
        if output_path is None:
            output_path = tempfile.mktemp(suffix=".wav")
        output_path = Path(output_path)

        logger.info(f"调用远程 TTS 服务...")
        logger.info(f"  文本: {text[:50]}{'...' if len(text) > 50 else ''}")

        try:
            # 准备请求数据
            data = {
                "text": text,
                "voice": voice
            }

            files = {}
            if reference_audio and Path(reference_audio).exists():
                files["reference_audio"] = open(reference_audio, "rb")

            # 发送请求
            response = requests.post(
                f"{self.server_url}/tts",
                data=data,
                files=files if files else None,
                timeout=self.timeout
            )

            # 关闭文件
            for f in files.values():
                f.close()

            if response.status_code == 200:
                # 保存音频文件
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(response.content)

                logger.info(f"✓ TTS 合成完成: {output_path}")
                return str(output_path)
            else:
                logger.error(f"TTS 服务返回错误: {response.status_code}")
                logger.error(f"  详情: {response.text}")
                return None

        except requests.exceptions.Timeout:
            logger.error("TTS 请求超时")
            return None
        except Exception as e:
            logger.error(f"TTS 合成失败: {e}")
            return None

    def generate_avatar(
        self,
        audio_path: Union[str, Path],
        reference_image_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        max_frames: int = 240
    ) -> Optional[str]:
        """
        调用远程 Avatar 服务生成视频

        Args:
            audio_path: 驱动音频路径
            reference_image_path: 参考图片路径
            output_path: 输出视频路径（可选）
            max_frames: 最大帧数

        Returns:
            生成的视频文件路径，失败返回 None
        """
        audio_path = Path(audio_path)
        reference_image_path = Path(reference_image_path)

        if not audio_path.exists():
            logger.error(f"音频文件不存在: {audio_path}")
            return None

        if not reference_image_path.exists():
            logger.error(f"图片文件不存在: {reference_image_path}")
            return None

        # 准备输出路径
        if output_path is None:
            output_path = tempfile.mktemp(suffix=".mp4")
        output_path = Path(output_path)

        logger.info(f"调用远程 Avatar 服务...")
        logger.info(f"  音频: {audio_path.name}")
        logger.info(f"  图片: {reference_image_path.name}")
        logger.info(f"  最大帧数: {max_frames}")

        try:
            # 准备文件
            files = {
                "audio": open(audio_path, "rb"),
                "reference_image": open(reference_image_path, "rb")
            }

            data = {
                "max_frames": max_frames
            }

            # 发送请求
            response = requests.post(
                f"{self.server_url}/avatar",
                data=data,
                files=files,
                timeout=self.timeout
            )

            # 关闭文件
            for f in files.values():
                f.close()

            if response.status_code == 200:
                # 保存视频文件
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(response.content)

                logger.info(f"✓ Avatar 视频生成完成: {output_path}")
                return str(output_path)
            else:
                logger.error(f"Avatar 服务返回错误: {response.status_code}")
                logger.error(f"  详情: {response.text}")
                return None

        except requests.exceptions.Timeout:
            logger.error("Avatar 请求超时（视频生成可能需要较长时间）")
            return None
        except Exception as e:
            logger.error(f"Avatar 生成失败: {e}")
            return None


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("RemoteGPUClient 测试")
    print("=" * 60)

    # 创建客户端
    client = RemoteGPUClient(server_url="http://192.168.50.218:8000")

    # 测试连接
    print("\n1. 测试连接...")
    if client.check_connection():
        print("✓ 连接成功")
    else:
        print("✗ 连接失败")
        exit(1)

    # 测试 TTS
    print("\n2. 测试 TTS...")
    audio_path = client.synthesize_speech(
        text="你好，我是数字分身。",
        output_path="outputs/test_remote_tts.wav"
    )
    if audio_path:
        print(f"✓ TTS 成功: {audio_path}")
    else:
        print("✗ TTS 失败")

    # 测试 Avatar（需要图片和音频）
    print("\n3. 测试 Avatar...")
    if audio_path and Path("uploads/photos").exists():
        # 查找一张测试图片
        photos = list(Path("uploads/photos").glob("*.jpg")) + list(Path("uploads/photos").glob("*.png"))
        if photos:
            video_path = client.generate_avatar(
                audio_path=audio_path,
                reference_image_path=photos[0],
                output_path="outputs/test_remote_avatar.mp4"
            )
            if video_path:
                print(f"✓ Avatar 成功: {video_path}")
            else:
                print("✗ Avatar 失败")
        else:
            print("跳过（未找到测试图片）")
    else:
        print("跳过（需要先测试 TTS 并准备图片）")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
