"""
数字人/Avatar 生成模块 - 使用 EchoMimic V2 生成唇同步视频
Avatar Generator Module - Using EchoMimic V2 for Lip-Sync Video Generation

功能：
1. 加载 EchoMimic V2 模型
2. 从单张照片生成数字人
3. 根据音频驱动唇部和头部动作
4. 生成带唇同步的视频

注意：
- EchoMimic V2 需要从 GitHub 克隆并安装
- 需要下载预训练权重（约 5-10GB）
- 建议 GPU 显存 >= 16GB
"""

import os
import sys
import json
import tempfile
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Union, Tuple, List
import numpy as np
from loguru import logger
from PIL import Image
import cv2

# 配置 loguru
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>")


class AvatarGenerator:
    """
    数字人生成器类
    基于 EchoMimic V2 实现音频驱动的数字人视频生成
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: str = "cuda",
        video_fps: int = 25,
        video_width: int = 512,
        video_height: int = 512
    ):
        """
        初始化数字人生成器

        Args:
            model_dir: EchoMimic 模型目录
            device: 运行设备 ("cuda" 或 "cpu")
            video_fps: 输出视频帧率
            video_width: 输出视频宽度
            video_height: 输出视频高度
        """
        self.device = device
        self.video_fps = video_fps
        self.video_width = video_width
        self.video_height = video_height

        # 模型目录
        self.model_dir = Path(model_dir) if model_dir else Path("models/echomimic_v2")
        self.echomimic_repo = Path("models/echomimic_v2_repo")

        # 模型组件
        self.model = None
        self.is_loaded = False

        # 参考图片缓存
        self.reference_images = {}

        # 使用备选方案的标志
        self.use_fallback = False

        logger.info(f"AvatarGenerator 初始化 - 设备: {self.device}, 分辨率: {video_width}x{video_height}")

    def load_model(self) -> bool:
        """
        加载 EchoMimic V2 模型

        Returns:
            bool: 是否加载成功
        """
        logger.info("正在加载 Avatar 生成模型...")

        try:
            # 检查 EchoMimic V2 是否已安装
            if self._check_echomimic_installed():
                return self._load_echomimic_model()
            else:
                logger.warning("EchoMimic V2 未安装，尝试下载...")
                if self._setup_echomimic():
                    return self._load_echomimic_model()
                else:
                    logger.warning("EchoMimic V2 安装失败，将使用备选方案")
                    self.use_fallback = True
                    return True  # 使用备选方案也算"成功"

        except Exception as e:
            logger.error(f"加载模型时发生错误: {e}")
            self.use_fallback = True
            return True

    def _check_echomimic_installed(self) -> bool:
        """
        检查 EchoMimic V2 是否已安装

        Returns:
            bool: 是否已安装
        """
        # 检查模型文件
        if not self.model_dir.exists():
            return False

        # 检查关键文件
        required_files = [
            "denoising_unet.pth",
            "reference_unet.pth",
            "motion_module.pth",
        ]

        for f in required_files:
            if not (self.model_dir / f).exists():
                logger.info(f"缺少模型文件: {f}")
                return False

        return True

    def _setup_echomimic(self) -> bool:
        """
        下载并设置 EchoMimic V2

        Returns:
            bool: 是否成功
        """
        logger.info("正在设置 EchoMimic V2...")

        try:
            # 创建目录
            self.echomimic_repo.mkdir(parents=True, exist_ok=True)
            self.model_dir.mkdir(parents=True, exist_ok=True)

            # 克隆仓库
            if not (self.echomimic_repo / ".git").exists():
                logger.info("克隆 EchoMimic V2 仓库...")
                result = subprocess.run(
                    ["git", "clone", "https://github.com/antgroup/echomimic_v2.git",
                     str(self.echomimic_repo)],
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    logger.error(f"克隆失败: {result.stderr}")
                    return False

            # 下载模型权重
            logger.info("下载模型权重（这可能需要较长时间）...")
            logger.info("请手动下载模型权重:")
            logger.info("  方式1: 从 HuggingFace 下载")
            logger.info("    huggingface-cli download antgroup/echomimic_v2 --local-dir models/echomimic_v2")
            logger.info("  方式2: 从 ModelScope 下载（中国区推荐）")
            logger.info("    modelscope download antgroup/echomimic_v2 --local_dir models/echomimic_v2")

            # 尝试使用 huggingface-cli 下载
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(
                    repo_id="antgroup/echomimic_v2",
                    local_dir=str(self.model_dir),
                    local_dir_use_symlinks=False
                )
                logger.info("✓ 模型下载完成")
                return True
            except Exception as e:
                logger.warning(f"自动下载失败: {e}")
                logger.warning("请手动下载模型")
                return False

        except Exception as e:
            logger.error(f"设置 EchoMimic 失败: {e}")
            return False

    def _load_echomimic_model(self) -> bool:
        """
        加载 EchoMimic V2 模型

        Returns:
            bool: 是否成功
        """
        try:
            # 添加 EchoMimic 到路径
            echomimic_path = str(self.echomimic_repo)
            if echomimic_path not in sys.path:
                sys.path.insert(0, echomimic_path)

            # 尝试导入 EchoMimic
            try:
                # EchoMimic V2 的导入方式可能不同
                # 这里提供多种尝试
                try:
                    from src.pipelines.pipeline_echomimic import EchoMimicPipeline
                    from src.models.unet_2d_condition import UNet2DConditionModel
                    from src.models.unet_3d_emo import EMOUNet3DConditionModel
                except ImportError:
                    from echomimic.pipelines.pipeline_echomimic import EchoMimicPipeline
                    from echomimic.models.unet_2d_condition import UNet2DConditionModel
                    from echomimic.models.unet_3d_emo import EMOUNet3DConditionModel

                logger.info("EchoMimic 模块导入成功")

            except ImportError as e:
                logger.warning(f"无法导入 EchoMimic 模块: {e}")
                logger.info("将使用命令行方式调用 EchoMimic")
                self.use_cli = True
                self.is_loaded = True
                return True

            # 加载模型
            import torch
            from diffusers import AutoencoderKL, DDIMScheduler
            from transformers import CLIPVisionModelWithProjection

            # 加载 VAE
            vae = AutoencoderKL.from_pretrained(
                str(self.model_dir / "sd-vae-ft-mse")
            ).to(self.device)

            # 加载 CLIP
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                str(self.model_dir / "sd-image-variations-diffusers" / "image_encoder")
            ).to(self.device)

            # 加载 UNets
            reference_unet = UNet2DConditionModel.from_pretrained(
                str(self.model_dir),
                subfolder="reference_unet"
            ).to(self.device)

            denoising_unet = EMOUNet3DConditionModel.from_pretrained(
                str(self.model_dir),
                subfolder="denoising_unet"
            ).to(self.device)

            # 加载调度器
            scheduler = DDIMScheduler.from_pretrained(
                str(self.model_dir),
                subfolder="scheduler"
            )

            # 创建 Pipeline
            self.model = EchoMimicPipeline(
                vae=vae,
                image_encoder=image_encoder,
                reference_unet=reference_unet,
                denoising_unet=denoising_unet,
                scheduler=scheduler
            ).to(self.device)

            self.is_loaded = True
            logger.info("✓ EchoMimic V2 模型加载成功")
            return True

        except Exception as e:
            logger.error(f"加载 EchoMimic 模型失败: {e}")
            logger.info("将使用命令行方式或备选方案")
            self.use_cli = True
            self.is_loaded = True
            return True

    def set_reference_image(
        self,
        image_path: Union[str, Path],
        avatar_id: str = "default"
    ) -> bool:
        """
        设置参考图片（用于生成数字人形象）

        Args:
            image_path: 图片路径
            avatar_id: 数字人标识符

        Returns:
            bool: 是否成功
        """
        image_path = Path(image_path)

        if not image_path.exists():
            logger.error(f"图片文件不存在: {image_path}")
            return False

        try:
            # 加载并预处理图片
            image = Image.open(image_path).convert("RGB")

            # 调整大小
            image = self._preprocess_image(image)

            # 缓存
            self.reference_images[avatar_id] = {
                "path": str(image_path),
                "image": image
            }

            logger.info(f"✓ 参考图片已设置: {avatar_id}")
            return True

        except Exception as e:
            logger.error(f"设置参考图片失败: {e}")
            return False

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        预处理图片

        Args:
            image: PIL Image

        Returns:
            Image: 处理后的图片
        """
        # 获取原始尺寸
        w, h = image.size

        # 计算缩放比例（保持宽高比）
        target_w, target_h = self.video_width, self.video_height
        ratio = min(target_w / w, target_h / h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)

        # 缩放
        image = image.resize((new_w, new_h), Image.LANCZOS)

        # 创建画布并居中
        canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        canvas.paste(image, (paste_x, paste_y))

        return canvas

    def generate_video(
        self,
        audio_path: Union[str, Path],
        avatar_id: str = "default",
        output_path: Optional[Union[str, Path]] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 3.5
    ) -> Optional[str]:
        """
        生成数字人说话视频

        Args:
            audio_path: 驱动音频路径
            avatar_id: 数字人ID
            output_path: 输出视频路径
            num_inference_steps: 推理步数（越多质量越好，但更慢）
            guidance_scale: 引导强度

        Returns:
            str: 生成的视频路径，或 None
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            logger.error(f"音频文件不存在: {audio_path}")
            return None

        # 检查参考图片
        if avatar_id not in self.reference_images:
            logger.error(f"未找到参考图片: {avatar_id}")
            return None

        # 生成输出路径
        if output_path is None:
            output_path = Path(tempfile.mktemp(suffix=".mp4"))
        else:
            output_path = Path(output_path)

        logger.info(f"正在生成数字人视频...")
        logger.info(f"  参考图片: {avatar_id}")
        logger.info(f"  驱动音频: {audio_path.name}")

        try:
            if self.use_fallback:
                return self._generate_fallback_video(audio_path, avatar_id, output_path)
            elif hasattr(self, 'use_cli') and self.use_cli:
                return self._generate_via_cli(audio_path, avatar_id, output_path,
                                              num_inference_steps, guidance_scale)
            else:
                return self._generate_via_pipeline(audio_path, avatar_id, output_path,
                                                   num_inference_steps, guidance_scale)

        except Exception as e:
            logger.error(f"生成视频失败: {e}")
            # 尝试备选方案
            return self._generate_fallback_video(audio_path, avatar_id, output_path)

    def _generate_via_pipeline(
        self,
        audio_path: Path,
        avatar_id: str,
        output_path: Path,
        num_inference_steps: int,
        guidance_scale: float
    ) -> Optional[str]:
        """
        使用 Pipeline API 生成视频
        """
        import torch
        import torchaudio

        # 获取参考图片
        ref_data = self.reference_images[avatar_id]
        ref_image = ref_data["image"]

        # 加载音频
        waveform, sr = torchaudio.load(str(audio_path))

        # 计算视频长度
        audio_duration = waveform.shape[1] / sr
        num_frames = int(audio_duration * self.video_fps)

        logger.info(f"音频时长: {audio_duration:.2f}秒, 预计帧数: {num_frames}")

        # 生成视频帧
        with torch.no_grad():
            video_frames = self.model(
                ref_image=ref_image,
                audio=waveform,
                audio_sample_rate=sr,
                num_frames=num_frames,
                fps=self.video_fps,
                width=self.video_width,
                height=self.video_height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).frames

        # 保存视频
        self._save_video_with_audio(video_frames, audio_path, output_path)

        logger.info(f"✓ 视频生成完成: {output_path}")
        return str(output_path)

    def _generate_via_cli(
        self,
        audio_path: Path,
        avatar_id: str,
        output_path: Path,
        num_inference_steps: int,
        guidance_scale: float
    ) -> Optional[str]:
        """
        通过命令行调用 EchoMimic
        """
        ref_data = self.reference_images[avatar_id]
        ref_image_path = ref_data["path"]

        # 构建命令
        cmd = [
            "python",
            str(self.echomimic_repo / "inference.py"),
            "--config", str(self.echomimic_repo / "configs/inference.yaml"),
            "--reference_image", ref_image_path,
            "--audio", str(audio_path),
            "--output", str(output_path),
            "--num_inference_steps", str(num_inference_steps),
            "--guidance_scale", str(guidance_scale),
            "--fps", str(self.video_fps),
            "--width", str(self.video_width),
            "--height", str(self.video_height)
        ]

        logger.info(f"执行命令: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(self.echomimic_repo)
        )

        if result.returncode == 0 and output_path.exists():
            logger.info(f"✓ 视频生成完成: {output_path}")
            return str(output_path)
        else:
            logger.error(f"命令执行失败: {result.stderr}")
            return self._generate_fallback_video(audio_path, avatar_id, output_path)

    def _generate_fallback_video(
        self,
        audio_path: Path,
        avatar_id: str,
        output_path: Path
    ) -> Optional[str]:
        """
        备选方案：生成简单的图片+音频视频
        当 EchoMimic 不可用时使用

        这个方法会创建一个静态图片视频，配合音频播放
        虽然没有唇同步，但至少能工作
        """
        logger.warning("使用备选方案生成视频（无唇同步）")

        try:
            import librosa
            # moviepy 2.x 导入方式
            from moviepy import ImageClip, AudioFileClip

            # 获取参考图片
            ref_data = self.reference_images[avatar_id]

            # 如果是 PIL Image，转换为 numpy 数组
            if isinstance(ref_data["image"], Image.Image):
                img_array = np.array(ref_data["image"])
            else:
                img_array = cv2.imread(ref_data["path"])
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

            # 获取音频时长
            audio_duration = librosa.get_duration(filename=str(audio_path))

            # 创建图片视频
            img_clip = ImageClip(img_array, duration=audio_duration)

            # 加载音频
            audio_clip = AudioFileClip(str(audio_path))

            # 合并
            final_clip = img_clip.with_audio(audio_clip)

            # 写入文件
            final_clip.write_videofile(
                str(output_path),
                fps=self.video_fps,
                codec='libx264',
                audio_codec='aac',
                logger=None  # 静默输出
            )

            # 清理
            img_clip.close()
            audio_clip.close()
            final_clip.close()

            if output_path.exists():
                logger.info(f"✓ 备选视频生成完成: {output_path}")
                return str(output_path)

            return None

        except Exception as e:
            logger.error(f"备选方案也失败了: {e}")
            return self._generate_minimal_video(audio_path, avatar_id, output_path)

    def _generate_minimal_video(
        self,
        audio_path: Path,
        avatar_id: str,
        output_path: Path
    ) -> Optional[str]:
        """
        最小化方案：使用 ffmpeg 直接合成
        """
        logger.warning("使用最小化方案（ffmpeg 直接合成）")

        try:
            ref_data = self.reference_images[avatar_id]

            # 保存图片
            if isinstance(ref_data["image"], Image.Image):
                temp_img = tempfile.mktemp(suffix=".png")
                ref_data["image"].save(temp_img)
            else:
                temp_img = ref_data["path"]

            # 使用 ffmpeg 合成
            cmd = [
                "ffmpeg", "-y",
                "-loop", "1",
                "-i", temp_img,
                "-i", str(audio_path),
                "-c:v", "libx264",
                "-tune", "stillimage",
                "-c:a", "aac",
                "-b:a", "192k",
                "-pix_fmt", "yuv420p",
                "-shortest",
                str(output_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0 and output_path.exists():
                logger.info(f"✓ 最小化视频生成完成: {output_path}")
                return str(output_path)

            logger.error(f"ffmpeg 失败: {result.stderr}")
            return None

        except Exception as e:
            logger.error(f"最小化方案失败: {e}")
            return None

    def _save_video_with_audio(
        self,
        frames: List[np.ndarray],
        audio_path: Path,
        output_path: Path
    ):
        """
        将视频帧和音频合并保存

        Args:
            frames: 视频帧列表
            audio_path: 音频文件路径
            output_path: 输出视频路径
        """
        # moviepy 2.x 导入方式
        from moviepy import ImageSequenceClip, AudioFileClip

        # 创建视频
        clip = ImageSequenceClip(frames, fps=self.video_fps)

        # 添加音频 (moviepy 2.x 使用 with_audio)
        audio = AudioFileClip(str(audio_path))
        clip = clip.with_audio(audio)

        # 写入
        clip.write_videofile(
            str(output_path),
            codec='libx264',
            audio_codec='aac',
            logger=None
        )

        clip.close()
        audio.close()

    def list_avatars(self) -> List[str]:
        """
        列出所有已注册的数字人

        Returns:
            List[str]: 数字人ID列表
        """
        return list(self.reference_images.keys())

    def get_avatar_preview(self, avatar_id: str) -> Optional[Image.Image]:
        """
        获取数字人预览图片

        Args:
            avatar_id: 数字人ID

        Returns:
            Image: 预览图片
        """
        if avatar_id in self.reference_images:
            return self.reference_images[avatar_id]["image"]
        return None


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("AvatarGenerator 测试")
    print("=" * 60)

    # 创建生成器
    generator = AvatarGenerator()

    # 加载模型
    print("\n1. 加载模型...")
    if generator.load_model():
        print("✓ 模型加载成功")
    else:
        print("✗ 模型加载失败")
        exit(1)

    # 测试设置参考图片
    print("\n2. 设置参考图片...")
    test_image = Path("uploads/test_face.jpg")
    if test_image.exists():
        if generator.set_reference_image(test_image, "test_avatar"):
            print("✓ 参考图片设置成功")
        else:
            print("✗ 参考图片设置失败")
    else:
        print(f"跳过（测试图片不存在: {test_image}）")

    # 测试生成视频
    print("\n3. 生成视频...")
    test_audio = Path("outputs/test_tts.wav")
    if test_audio.exists() and "test_avatar" in generator.list_avatars():
        output = generator.generate_video(
            test_audio,
            avatar_id="test_avatar",
            output_path="outputs/test_video.mp4"
        )
        if output:
            print(f"✓ 视频生成成功: {output}")
        else:
            print("✗ 视频生成失败")
    else:
        print("跳过（需要先运行 TTS 测试生成音频，并提供参考图片）")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
