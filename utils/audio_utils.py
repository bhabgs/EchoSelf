"""
音频处理工具模块
Audio Utilities Module

提供音频加载、处理、保存等工具函数
"""

import os
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Union, Tuple
import numpy as np
from loguru import logger


def load_audio(
    file_path: Union[str, Path],
    target_sr: Optional[int] = None,
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """
    加载音频文件

    Args:
        file_path: 音频文件路径
        target_sr: 目标采样率（None 表示保持原始）
        mono: 是否转换为单声道

    Returns:
        Tuple[np.ndarray, int]: (音频数据, 采样率)
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"音频文件不存在: {file_path}")

    try:
        import librosa

        # 加载音频
        audio, sr = librosa.load(
            str(file_path),
            sr=target_sr,
            mono=mono
        )

        return audio, sr

    except ImportError:
        # 备选方案：使用 soundfile
        import soundfile as sf

        audio, sr = sf.read(str(file_path))

        # 转换为单声道
        if mono and len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # 重采样（如果需要）
        if target_sr and sr != target_sr:
            audio = resample_audio(audio, sr, target_sr)
            sr = target_sr

        return audio, sr


def save_audio(
    audio: np.ndarray,
    file_path: Union[str, Path],
    sample_rate: int,
    format: str = "wav"
) -> Path:
    """
    保存音频文件

    Args:
        audio: 音频数据
        file_path: 保存路径
        sample_rate: 采样率
        format: 音频格式

    Returns:
        Path: 保存的文件路径
    """
    file_path = Path(file_path)

    # 确保目录存在
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import soundfile as sf

        # 确保音频数据是正确的范围
        if audio.dtype == np.float64 or audio.dtype == np.float32:
            audio = np.clip(audio, -1.0, 1.0)

        sf.write(str(file_path), audio, sample_rate, format=format)

        logger.debug(f"音频已保存: {file_path}")
        return file_path

    except ImportError:
        # 备选方案：使用 scipy
        from scipy.io import wavfile

        # 转换为 16-bit PCM
        if audio.dtype in [np.float32, np.float64]:
            audio = (audio * 32767).astype(np.int16)

        wavfile.write(str(file_path), sample_rate, audio)

        return file_path


def get_audio_duration(file_path: Union[str, Path]) -> float:
    """
    获取音频文件时长（秒）

    Args:
        file_path: 音频文件路径

    Returns:
        float: 音频时长（秒）
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"音频文件不存在: {file_path}")

    try:
        import librosa
        duration = librosa.get_duration(filename=str(file_path))
        return duration

    except ImportError:
        import soundfile as sf
        info = sf.info(str(file_path))
        return info.duration


def resample_audio(
    audio: np.ndarray,
    original_sr: int,
    target_sr: int
) -> np.ndarray:
    """
    重采样音频

    Args:
        audio: 音频数据
        original_sr: 原始采样率
        target_sr: 目标采样率

    Returns:
        np.ndarray: 重采样后的音频
    """
    if original_sr == target_sr:
        return audio

    try:
        import librosa
        resampled = librosa.resample(
            audio,
            orig_sr=original_sr,
            target_sr=target_sr
        )
        return resampled

    except ImportError:
        # 备选方案：使用 scipy
        from scipy import signal

        duration = len(audio) / original_sr
        num_samples = int(duration * target_sr)
        resampled = signal.resample(audio, num_samples)

        return resampled


def normalize_audio(
    audio: np.ndarray,
    target_db: float = -20.0
) -> np.ndarray:
    """
    归一化音频响度

    Args:
        audio: 音频数据
        target_db: 目标响度（dB）

    Returns:
        np.ndarray: 归一化后的音频
    """
    # 计算当前 RMS
    rms = np.sqrt(np.mean(audio ** 2))

    if rms == 0:
        return audio

    # 计算目标 RMS
    target_rms = 10 ** (target_db / 20)

    # 应用增益
    gain = target_rms / rms
    normalized = audio * gain

    # 防止削波
    normalized = np.clip(normalized, -1.0, 1.0)

    return normalized


def convert_to_wav(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    sample_rate: int = 44100,
    channels: int = 1
) -> Path:
    """
    将音频转换为 WAV 格式

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径（可选，自动生成）
        sample_rate: 目标采样率
        channels: 声道数

    Returns:
        Path: 输出文件路径
    """
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path.with_suffix('.wav')
    else:
        output_path = Path(output_path)

    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 使用 ffmpeg 转换
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-acodec", "pcm_s16le",
        str(output_path)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        logger.debug(f"音频已转换: {output_path}")
        return output_path

    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg 转换失败: {e.stderr}")

        # 备选方案：使用 pydub
        try:
            from pydub import AudioSegment

            audio = AudioSegment.from_file(str(input_path))
            audio = audio.set_frame_rate(sample_rate).set_channels(channels)
            audio.export(str(output_path), format="wav")

            return output_path

        except ImportError:
            raise RuntimeError("无法转换音频：ffmpeg 失败且 pydub 未安装")


def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    top_db: float = 30.0
) -> np.ndarray:
    """
    去除音频首尾的静音部分

    Args:
        audio: 音频数据
        sample_rate: 采样率
        top_db: 静音阈值（dB）

    Returns:
        np.ndarray: 处理后的音频
    """
    try:
        import librosa

        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed

    except ImportError:
        # 简单的静音检测
        threshold = np.max(np.abs(audio)) * (10 ** (-top_db / 20))

        # 找到开始和结束位置
        above_threshold = np.abs(audio) > threshold
        if not np.any(above_threshold):
            return audio

        start = np.argmax(above_threshold)
        end = len(audio) - np.argmax(above_threshold[::-1])

        return audio[start:end]


def split_audio(
    file_path: Union[str, Path],
    segment_duration: float = 30.0,
    output_dir: Optional[Union[str, Path]] = None
) -> list:
    """
    将长音频分割为多个片段

    Args:
        file_path: 音频文件路径
        segment_duration: 每段时长（秒）
        output_dir: 输出目录

    Returns:
        list: 分割后的音频文件路径列表
    """
    file_path = Path(file_path)

    if output_dir is None:
        output_dir = file_path.parent / f"{file_path.stem}_segments"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载音频
    audio, sr = load_audio(file_path)
    total_duration = len(audio) / sr

    segments = []
    segment_samples = int(segment_duration * sr)

    for i, start in enumerate(range(0, len(audio), segment_samples)):
        end = min(start + segment_samples, len(audio))
        segment = audio[start:end]

        # 保存片段
        segment_path = output_dir / f"{file_path.stem}_segment_{i:03d}.wav"
        save_audio(segment, segment_path, sr)
        segments.append(segment_path)

    logger.info(f"音频已分割为 {len(segments)} 个片段")
    return segments


def concatenate_audio(
    audio_files: list,
    output_path: Union[str, Path],
    crossfade_ms: int = 0
) -> Path:
    """
    拼接多个音频文件

    Args:
        audio_files: 音频文件路径列表
        output_path: 输出文件路径
        crossfade_ms: 交叉淡入淡出时长（毫秒）

    Returns:
        Path: 输出文件路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from pydub import AudioSegment

        combined = AudioSegment.empty()

        for i, audio_file in enumerate(audio_files):
            audio = AudioSegment.from_file(str(audio_file))

            if i == 0:
                combined = audio
            else:
                if crossfade_ms > 0:
                    combined = combined.append(audio, crossfade=crossfade_ms)
                else:
                    combined += audio

        combined.export(str(output_path), format="wav")
        logger.info(f"音频已拼接: {output_path}")
        return output_path

    except ImportError:
        # 备选方案：使用 numpy
        all_audio = []
        sample_rate = None

        for audio_file in audio_files:
            audio, sr = load_audio(audio_file)
            if sample_rate is None:
                sample_rate = sr
            elif sr != sample_rate:
                audio = resample_audio(audio, sr, sample_rate)
            all_audio.append(audio)

        concatenated = np.concatenate(all_audio)
        save_audio(concatenated, output_path, sample_rate)

        return output_path


def get_audio_info(file_path: Union[str, Path]) -> dict:
    """
    获取音频文件信息

    Args:
        file_path: 音频文件路径

    Returns:
        dict: 音频信息
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"音频文件不存在: {file_path}")

    try:
        import soundfile as sf

        info = sf.info(str(file_path))

        return {
            "path": str(file_path),
            "duration": info.duration,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "format": info.format,
            "subtype": info.subtype,
            "frames": info.frames
        }

    except ImportError:
        # 基本信息
        audio, sr = load_audio(file_path)

        return {
            "path": str(file_path),
            "duration": len(audio) / sr,
            "sample_rate": sr,
            "channels": 1 if len(audio.shape) == 1 else audio.shape[1],
            "frames": len(audio)
        }


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Audio Utils 测试")
    print("=" * 60)

    # 创建测试音频
    print("\n1. 创建测试音频...")
    sr = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz 正弦波

    test_file = Path(tempfile.gettempdir()) / "test_audio.wav"
    save_audio(audio, test_file, sr)
    print(f"✓ 测试音频已创建: {test_file}")

    # 测试加载
    print("\n2. 测试音频加载...")
    loaded_audio, loaded_sr = load_audio(test_file)
    print(f"✓ 采样率: {loaded_sr}, 时长: {len(loaded_audio)/loaded_sr:.2f}秒")

    # 测试时长获取
    print("\n3. 测试时长获取...")
    duration = get_audio_duration(test_file)
    print(f"✓ 音频时长: {duration:.2f}秒")

    # 测试归一化
    print("\n4. 测试音频归一化...")
    normalized = normalize_audio(loaded_audio, target_db=-20)
    print(f"✓ 归一化完成")

    # 测试信息获取
    print("\n5. 测试音频信息获取...")
    info = get_audio_info(test_file)
    for key, value in info.items():
        print(f"  {key}: {value}")

    # 清理
    print("\n6. 清理测试文件...")
    test_file.unlink()
    print("✓ 清理完成")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
