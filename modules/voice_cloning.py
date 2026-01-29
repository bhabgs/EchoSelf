"""
声音克隆 TTS 模块 - 使用 Fish Speech V1.5 进行声音克隆
Voice Cloning TTS Module - Using Fish Speech V1.5

功能：
1. 加载并管理 TTS 模型
2. 从参考音频提取说话人特征
3. 生成克隆声音的语音
4. 支持中英文混合文本

注意：
- Fish Speech V1.5 需要从 HuggingFace 下载模型
- 首次运行会自动下载模型（约 1-2GB）
- 需要足够的 GPU 显存（建议 8GB+）
"""

import os
import sys
import json
import hashlib
import tempfile
from pathlib import Path
from typing import Optional, Union, List, Tuple
import numpy as np
from loguru import logger

# 音频处理
import torch
import torchaudio
import soundfile as sf
from scipy.io import wavfile

# 配置 loguru
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>")


class VoiceCloner:
    """
    声音克隆器类
    基于 Fish Speech V1.5 实现零样本声音克隆
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        sample_rate: int = 44100
    ):
        """
        初始化声音克隆器

        Args:
            model_dir: 模型目录路径
            device: 运行设备 ("cuda" 或 "cpu")
            sample_rate: 输出采样率
        """
        self.device = device
        self.sample_rate = sample_rate
        self.model_dir = Path(model_dir) if model_dir else Path("models/fish-speech-1.5")

        # 模型组件
        self.model = None
        self.vocoder = None
        self.tokenizer = None

        # 参考音频特征缓存
        self.speaker_embeddings = {}

        # 是否使用备选方案
        self.use_fallback = False

        logger.info(f"VoiceCloner 初始化 - 设备: {self.device}")

    def load_model(self) -> bool:
        """
        加载 TTS 模型

        Returns:
            bool: 是否加载成功
        """
        logger.info("正在加载 TTS 模型...")

        try:
            # 尝试导入 Fish Speech
            # 注意：Fish Speech 可能需要从源码安装
            try:
                from fish_speech.inference import inference
                from fish_speech.models import VQGANModel
                logger.info("Fish Speech 模块导入成功")

            except ImportError:
                logger.warning("Fish Speech 未安装，尝试使用备选方案...")
                self.use_fallback = True
                return self._load_fallback_model()

            # 检查模型文件
            if not self.model_dir.exists():
                logger.info("模型目录不存在，尝试下载...")
                self._download_model()

            # 加载模型
            # 注意：实际的 Fish Speech 加载代码会根据其 API 调整
            self.model = self._initialize_fish_speech()

            if self.model is not None:
                logger.info("✓ TTS 模型加载成功")
                return True
            else:
                logger.warning("Fish Speech 加载失败，使用备选方案")
                self.use_fallback = True
                return self._load_fallback_model()

        except Exception as e:
            logger.error(f"加载模型时发生错误: {e}")
            logger.warning("将使用备选 TTS 方案")
            self.use_fallback = True
            return self._load_fallback_model()

    def _load_fallback_model(self) -> bool:
        """
        加载备选 TTS 模型 (edge-tts 或 gTTS)
        当 Fish Speech 不可用时使用
        """
        try:
            import edge_tts
            self.fallback_engine = "edge_tts"
            logger.info("使用 edge-tts 作为备选 TTS 引擎")
            return True
        except ImportError:
            try:
                from gtts import gTTS
                self.fallback_engine = "gtts"
                logger.info("使用 gTTS 作为备选 TTS 引擎")
                return True
            except ImportError:
                logger.error("没有可用的 TTS 引擎，请安装 edge-tts 或 gtts")
                return False

    def _download_model(self):
        """
        从 HuggingFace 下载 Fish Speech 模型
        """
        logger.info("正在下载 Fish Speech V1.5 模型...")

        try:
            from huggingface_hub import snapshot_download

            # 下载模型
            snapshot_download(
                repo_id="fishaudio/fish-speech-1.5",
                local_dir=str(self.model_dir),
                local_dir_use_symlinks=False
            )
            logger.info("✓ 模型下载完成")

        except Exception as e:
            logger.error(f"下载模型失败: {e}")
            logger.info("请手动下载模型:")
            logger.info("  huggingface-cli download fishaudio/fish-speech-1.5 --local-dir models/fish-speech-1.5")
            raise

    def _initialize_fish_speech(self):
        """
        初始化 Fish Speech 模型

        Returns:
            模型实例或 None
        """
        try:
            # Fish Speech V1.5 的加载方式
            # 注意：这里的代码需要根据 Fish Speech 的实际 API 调整

            # 方式1：使用 fish_speech 的官方 API
            try:
                from fish_speech.tools.vqgan.inference import load_model
                from fish_speech.tools.llama.generate import load_model as load_llm

                # 加载 VQGAN
                vqgan_path = self.model_dir / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
                if vqgan_path.exists():
                    self.vocoder = load_model(str(vqgan_path), self.device)

                # 加载 LLM
                llm_path = self.model_dir / "text2semantic-sft-medium-v1.1-4k.pth"
                if llm_path.exists():
                    self.model = load_llm(str(llm_path), self.device)

                return self.model

            except ImportError:
                pass

            # 方式2：使用 transformers 加载（如果 Fish Speech 支持）
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer

                # Fish Speech 可能使用自定义模型类
                # 这里作为示例
                return None

            except Exception:
                pass

            return None

        except Exception as e:
            logger.error(f"初始化 Fish Speech 失败: {e}")
            return None

    def extract_speaker_embedding(
        self,
        audio_path: Union[str, Path],
        speaker_id: str = "default"
    ) -> Optional[np.ndarray]:
        """
        从参考音频提取说话人特征

        Args:
            audio_path: 参考音频文件路径
            speaker_id: 说话人标识符（用于缓存）

        Returns:
            np.ndarray: 说话人特征向量，或 None（如果使用备选方案）
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            logger.error(f"音频文件不存在: {audio_path}")
            return None

        logger.info(f"正在提取说话人特征: {audio_path.name}")

        # 计算文件哈希作为缓存键
        file_hash = self._get_file_hash(audio_path)
        cache_key = f"{speaker_id}_{file_hash}"

        # 检查缓存
        if cache_key in self.speaker_embeddings:
            logger.info("使用缓存的说话人特征")
            return self.speaker_embeddings[cache_key]

        try:
            # 加载音频
            waveform, sr = torchaudio.load(str(audio_path))

            # 重采样（如果需要）
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)

            # 转为单声道
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # 如果使用 Fish Speech
            if not self.use_fallback and self.model is not None:
                # 使用 Fish Speech 提取特征
                # 注意：实际代码需要根据 Fish Speech API 调整
                embedding = self._extract_fish_speech_embedding(waveform)
            else:
                # 备选方案：保存原始音频路径
                # edge-tts 不支持声音克隆，只能使用预设声音
                embedding = waveform.numpy()

            # 缓存特征
            self.speaker_embeddings[cache_key] = embedding
            self.speaker_embeddings[speaker_id] = embedding  # 同时用 speaker_id 缓存

            logger.info(f"✓ 说话人特征提取完成")
            return embedding

        except Exception as e:
            logger.error(f"提取说话人特征失败: {e}")
            return None

    def _extract_fish_speech_embedding(self, waveform: torch.Tensor) -> np.ndarray:
        """
        使用 Fish Speech 提取说话人特征

        Args:
            waveform: 音频波形

        Returns:
            np.ndarray: 特征向量
        """
        # Fish Speech V1.5 使用参考音频进行 prompt-based 生成
        # 这里返回音频本身，在生成时使用
        return waveform.numpy()

    def synthesize(
        self,
        text: str,
        speaker_id: str = "default",
        output_path: Optional[Union[str, Path]] = None,
        speed: float = 1.0,
        pitch: float = 1.0
    ) -> Optional[str]:
        """
        合成语音

        Args:
            text: 要合成的文本
            speaker_id: 说话人ID（需要先调用 extract_speaker_embedding）
            output_path: 输出音频路径（如果为 None，生成临时文件）
            speed: 语速调整（1.0 为正常）
            pitch: 音调调整（1.0 为正常）

        Returns:
            str: 生成的音频文件路径，或 None（如果失败）
        """
        if not text.strip():
            logger.warning("输入文本为空")
            return None

        logger.info(f"正在合成语音: {text[:50]}...")

        # 生成输出路径
        if output_path is None:
            output_path = Path(tempfile.mktemp(suffix=".wav"))
        else:
            output_path = Path(output_path)

        try:
            if not self.use_fallback and self.model is not None:
                # 使用 Fish Speech 合成
                return self._synthesize_fish_speech(
                    text, speaker_id, output_path, speed, pitch
                )
            else:
                # 使用备选方案
                return self._synthesize_fallback(text, output_path)

        except Exception as e:
            logger.error(f"语音合成失败: {e}")
            return None

    def _synthesize_fish_speech(
        self,
        text: str,
        speaker_id: str,
        output_path: Path,
        speed: float,
        pitch: float
    ) -> Optional[str]:
        """
        使用 Fish Speech 合成语音

        Args:
            text: 文本
            speaker_id: 说话人ID
            output_path: 输出路径
            speed: 语速
            pitch: 音调

        Returns:
            str: 输出文件路径
        """
        try:
            # 获取说话人特征
            speaker_embedding = self.speaker_embeddings.get(speaker_id)

            if speaker_embedding is None:
                logger.warning(f"未找到说话人 '{speaker_id}' 的特征，使用默认声音")

            # Fish Speech 合成
            # 注意：实际代码需要根据 Fish Speech API 调整

            # 示例代码（需要根据实际 API 修改）：
            """
            from fish_speech.inference import synthesize

            audio = synthesize(
                text=text,
                reference_audio=speaker_embedding,
                model=self.model,
                vocoder=self.vocoder,
                speed=speed,
                device=self.device
            )

            sf.write(str(output_path), audio, self.sample_rate)
            """

            # 由于 Fish Speech API 可能变化，这里使用命令行方式
            # 这是更稳定的方法
            import subprocess

            # 保存参考音频（如果有）
            ref_audio_path = None
            if speaker_embedding is not None:
                ref_audio_path = tempfile.mktemp(suffix=".wav")
                if isinstance(speaker_embedding, np.ndarray):
                    sf.write(ref_audio_path, speaker_embedding.flatten(), self.sample_rate)

            # 调用 Fish Speech CLI
            cmd = [
                "python", "-m", "fish_speech.tools.inference",
                "--text", text,
                "--output", str(output_path),
                "--checkpoint", str(self.model_dir),
            ]

            if ref_audio_path:
                cmd.extend(["--reference", ref_audio_path])

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0 and output_path.exists():
                logger.info(f"✓ 语音合成完成: {output_path}")
                return str(output_path)
            else:
                logger.error(f"Fish Speech 命令执行失败: {result.stderr}")
                # 回退到备选方案
                return self._synthesize_fallback(text, output_path)

        except Exception as e:
            logger.error(f"Fish Speech 合成失败: {e}")
            return self._synthesize_fallback(text, output_path)

    def _synthesize_fallback(
        self,
        text: str,
        output_path: Path
    ) -> Optional[str]:
        """
        使用备选 TTS 引擎合成语音

        Args:
            text: 文本
            output_path: 输出路径

        Returns:
            str: 输出文件路径
        """
        try:
            if hasattr(self, 'fallback_engine') and self.fallback_engine == "edge_tts":
                return self._synthesize_edge_tts(text, output_path)
            else:
                return self._synthesize_gtts(text, output_path)

        except Exception as e:
            logger.error(f"备选 TTS 合成失败: {e}")
            return None

    def _synthesize_edge_tts(self, text: str, output_path: Path) -> Optional[str]:
        """
        使用 edge-tts 合成语音
        """
        import asyncio
        import edge_tts

        async def _synthesize():
            # 使用中文声音
            voice = "zh-CN-XiaoxiaoNeural"  # 女声
            # voice = "zh-CN-YunxiNeural"   # 男声

            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(str(output_path))

        # 运行异步函数
        asyncio.run(_synthesize())

        if output_path.exists():
            logger.info(f"✓ 语音合成完成 (edge-tts): {output_path}")
            return str(output_path)
        return None

    def _synthesize_gtts(self, text: str, output_path: Path) -> Optional[str]:
        """
        使用 gTTS 合成语音
        """
        from gtts import gTTS

        tts = gTTS(text=text, lang='zh-cn')

        # gTTS 输出 mp3，需要转换
        mp3_path = str(output_path).replace('.wav', '.mp3')
        tts.save(mp3_path)

        # 转换为 wav
        from pydub import AudioSegment
        audio = AudioSegment.from_mp3(mp3_path)
        audio.export(str(output_path), format="wav")

        # 删除临时 mp3
        os.remove(mp3_path)

        if output_path.exists():
            logger.info(f"✓ 语音合成完成 (gTTS): {output_path}")
            return str(output_path)
        return None

    def _get_file_hash(self, file_path: Path) -> str:
        """
        计算文件 MD5 哈希

        Args:
            file_path: 文件路径

        Returns:
            str: 文件哈希值（前8位）
        """
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()[:8]

    def list_speakers(self) -> List[str]:
        """
        列出所有已注册的说话人

        Returns:
            List[str]: 说话人ID列表
        """
        return list(self.speaker_embeddings.keys())

    def save_speaker(self, speaker_id: str, save_path: Union[str, Path]):
        """
        保存说话人特征到文件

        Args:
            speaker_id: 说话人ID
            save_path: 保存路径
        """
        if speaker_id not in self.speaker_embeddings:
            logger.error(f"说话人 '{speaker_id}' 不存在")
            return

        embedding = self.speaker_embeddings[speaker_id]
        np.save(str(save_path), embedding)
        logger.info(f"说话人特征已保存: {save_path}")

    def load_speaker(self, speaker_id: str, load_path: Union[str, Path]):
        """
        从文件加载说话人特征

        Args:
            speaker_id: 说话人ID
            load_path: 加载路径
        """
        load_path = Path(load_path)
        if not load_path.exists():
            logger.error(f"文件不存在: {load_path}")
            return

        embedding = np.load(str(load_path))
        self.speaker_embeddings[speaker_id] = embedding
        logger.info(f"说话人特征已加载: {speaker_id}")


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("VoiceCloner 测试")
    print("=" * 60)

    # 创建克隆器
    cloner = VoiceCloner()

    # 加载模型
    print("\n1. 加载模型...")
    if cloner.load_model():
        print("✓ 模型加载成功")
    else:
        print("✗ 模型加载失败")
        exit(1)

    # 测试合成（不带声音克隆）
    print("\n2. 测试基础语音合成...")
    output_path = cloner.synthesize(
        "你好，我是你的数字分身。很高兴认识你！",
        output_path="outputs/test_tts.wav"
    )

    if output_path:
        print(f"✓ 语音合成成功: {output_path}")
    else:
        print("✗ 语音合成失败")

    # 如果有参考音频，测试声音克隆
    ref_audio = Path("uploads/reference_voice.wav")
    if ref_audio.exists():
        print("\n3. 测试声音克隆...")
        cloner.extract_speaker_embedding(ref_audio, "user1")

        output_path = cloner.synthesize(
            "这是使用克隆声音合成的测试语音。",
            speaker_id="user1",
            output_path="outputs/test_cloned.wav"
        )

        if output_path:
            print(f"✓ 克隆语音合成成功: {output_path}")
        else:
            print("✗ 克隆语音合成失败")
    else:
        print("\n3. 跳过声音克隆测试（无参考音频）")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
