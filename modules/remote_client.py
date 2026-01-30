"""
远程 GPU 服务客户端 (Duix-Avatar)

用于调用远程服务器上的 TTS、Avatar 和 ASR API
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, Union
import requests
from loguru import logger


class RemoteGPUClient:
    """
    远程 GPU 服务客户端 (Duix-Avatar)

    用于调用远程服务器上的:
    - TTS 语音合成 (Fish Speech)
    - Avatar 视频生成 (Duix Avatar)
    - ASR 语音识别 (FunASR)
    - 声音克隆
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
        self.services_status = {}

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
                self.services_status = data.get('services', {})
                self.is_available = data.get('status') in ['ok', 'degraded']

                logger.info(f"远程 GPU 服务状态: {data.get('status')}")
                logger.info(f"  - TTS 服务: {'可用' if self.services_status.get('tts') else '不可用'}")
                logger.info(f"  - Video 服务: {'可用' if self.services_status.get('video') else '不可用'}")
                logger.info(f"  - ASR 服务: {'可用' if self.services_status.get('asr') else '不可用'}")

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

    def clone_voice(
        self,
        speaker_id: str,
        reference_audio: Union[str, Path]
    ) -> bool:
        """
        克隆声音

        Args:
            speaker_id: 说话人唯一标识
            reference_audio: 参考音频路径 (建议 5-30 秒)

        Returns:
            是否成功
        """
        reference_audio = Path(reference_audio)

        if not reference_audio.exists():
            logger.error(f"参考音频不存在: {reference_audio}")
            return False

        logger.info(f"克隆声音 - speaker_id: {speaker_id}")

        try:
            with open(reference_audio, "rb") as f:
                files = {"reference_audio": f}
                data = {"speaker_id": speaker_id}

                response = requests.post(
                    f"{self.server_url}/clone-voice",
                    data=data,
                    files=files,
                    timeout=60
                )

            if response.status_code == 200:
                logger.info(f"✓ 声音克隆成功: {speaker_id}")
                return True
            else:
                logger.error(f"声音克隆失败: {response.status_code}")
                logger.error(f"  详情: {response.text}")
                return False

        except Exception as e:
            logger.error(f"声音克隆异常: {e}")
            return False

    def synthesize_speech(
        self,
        text: str,
        output_path: Optional[Union[str, Path]] = None,
        speaker_id: Optional[str] = None
    ) -> Optional[str]:
        """
        调用远程 TTS 服务合成语音 (Fish Speech)

        Args:
            text: 要合成的文本
            output_path: 输出音频路径（可选，不指定则自动生成）
            speaker_id: 说话人 ID（用于声音克隆，需先调用 clone_voice）

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
        if speaker_id:
            logger.info(f"  说话人: {speaker_id}")

        try:
            # 准备请求数据
            data = {"text": text}
            if speaker_id:
                data["speaker_id"] = speaker_id

            # 发送请求
            response = requests.post(
                f"{self.server_url}/tts",
                data=data,
                timeout=self.timeout
            )

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
        reference_video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None
    ) -> Optional[str]:
        """
        调用远程 Avatar 服务生成视频 (Duix Avatar)

        Args:
            audio_path: 驱动音频路径
            reference_video_path: 参考视频/图片路径
            output_path: 输出视频路径（可选）

        Returns:
            生成的视频文件路径，失败返回 None
        """
        audio_path = Path(audio_path)
        reference_video_path = Path(reference_video_path)

        if not audio_path.exists():
            logger.error(f"音频文件不存在: {audio_path}")
            return None

        if not reference_video_path.exists():
            logger.error(f"参考视频/图片不存在: {reference_video_path}")
            return None

        # 准备输出路径
        if output_path is None:
            output_path = tempfile.mktemp(suffix=".mp4")
        output_path = Path(output_path)

        logger.info(f"调用远程 Avatar 服务...")
        logger.info(f"  音频: {audio_path.name}")
        logger.info(f"  参考: {reference_video_path.name}")

        try:
            # 准备文件
            with open(audio_path, "rb") as audio_file, \
                 open(reference_video_path, "rb") as video_file:

                files = {
                    "audio": audio_file,
                    "reference_video": video_file
                }

                # 发送请求
                response = requests.post(
                    f"{self.server_url}/avatar",
                    files=files,
                    timeout=self.timeout
                )

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

    def transcribe(
        self,
        audio_path: Union[str, Path]
    ) -> Optional[str]:
        """
        调用远程 ASR 服务进行语音识别 (FunASR)

        Args:
            audio_path: 音频文件路径

        Returns:
            识别的文本，失败返回 None
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            logger.error(f"音频文件不存在: {audio_path}")
            return None

        logger.info(f"调用远程 ASR 服务...")
        logger.info(f"  音频: {audio_path.name}")

        try:
            with open(audio_path, "rb") as f:
                files = {"audio": f}

                response = requests.post(
                    f"{self.server_url}/asr",
                    files=files,
                    timeout=60
                )

            if response.status_code == 200:
                result = response.json()
                text = result.get("text", "")
                logger.info(f"✓ ASR 识别完成: {text[:50]}...")
                return text
            else:
                logger.error(f"ASR 服务返回错误: {response.status_code}")
                logger.error(f"  详情: {response.text}")
                return None

        except requests.exceptions.Timeout:
            logger.error("ASR 请求超时")
            return None
        except Exception as e:
            logger.error(f"ASR 识别失败: {e}")
            return None

    def get_speakers(self) -> list:
        """
        获取已注册的说话人列表

        Returns:
            说话人 ID 列表
        """
        try:
            response = requests.get(
                f"{self.server_url}/speakers",
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("speakers", [])
            else:
                return []

        except Exception as e:
            logger.error(f"获取说话人列表失败: {e}")
            return []


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("RemoteGPUClient 测试 (Duix-Avatar)")
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
        text="你好，我是数字分身。这是使用 Fish Speech 合成的语音。",
        output_path="outputs/test_remote_tts.wav"
    )
    if audio_path:
        print(f"✓ TTS 成功: {audio_path}")
    else:
        print("✗ TTS 失败")

    # 测试已注册说话人
    print("\n3. 获取已注册说话人...")
    speakers = client.get_speakers()
    print(f"  说话人列表: {speakers}")

    # 测试 Avatar（需要视频和音频）
    print("\n4. 测试 Avatar...")
    if audio_path and Path("uploads/videos").exists():
        # 查找一个测试视频
        videos = list(Path("uploads/videos").glob("*.mp4"))
        if videos:
            video_path = client.generate_avatar(
                audio_path=audio_path,
                reference_video_path=videos[0],
                output_path="outputs/test_remote_avatar.mp4"
            )
            if video_path:
                print(f"✓ Avatar 成功: {video_path}")
            else:
                print("✗ Avatar 失败")
        else:
            print("跳过（未找到测试视频）")
    else:
        print("跳过（需要先测试 TTS 并准备视频）")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
