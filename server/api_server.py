"""
EchoSelf GPU Server - 提供 TTS 和 Avatar 生成 API (EchoMimic V3)

运行: python api_server.py
文档: http://localhost:8000/docs
"""

import os
import sys
import uuid
import math
import tempfile
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import yaml
import torch
import numpy as np
from PIL import Image
from loguru import logger
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# 配置日志
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>")
logger.add("logs/server.log", rotation="100 MB", level="DEBUG")


# ============================================================
# 全局变量
# ============================================================
config = None
avatar_service = None
tts_service = None


# ============================================================
# 配置加载
# ============================================================
def load_config(config_path: str = "config.yaml") -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# ============================================================
# EchoMimic V3 Avatar 生成服务
# ============================================================
class AvatarServiceV3:
    """EchoMimic V3 Avatar 生成服务 (flash-pro 版本)"""

    def __init__(self, config: dict):
        self.config = config
        self.device = config.get('device', 'cuda:0')
        self.model_dir = Path(config.get('model_dir', './models/echomimic_v3'))
        self.base_model = config.get('base_model', './models/echomimic_v3/Wan2.1-Fun-V1.1-1.3B-InP')
        self.wav2vec_model = config.get('wav2vec_model', './models/echomimic_v3/chinese-wav2vec2-base')
        self.transformer_path = config.get('transformer_path', './models/echomimic_v3/transformer/diffusion_pytorch_model.safetensors')

        self.width = config.get('width', 768)
        self.height = config.get('height', 768)
        self.fps = config.get('fps', 25)
        self.max_frames = config.get('max_frames', 81)
        self.num_inference_steps = config.get('num_inference_steps', 8)
        self.guidance_scale = config.get('guidance_scale', 6.0)
        self.audio_guidance_scale = config.get('audio_guidance_scale', 3.0)
        self.enable_teacache = config.get('enable_teacache', True)
        self.teacache_threshold = config.get('teacache_threshold', 0.1)
        self.weight_dtype_str = config.get('weight_dtype', 'bfloat16')

        self.pipeline = None
        self.audio_encoder = None
        self.wav2vec_feature_extractor = None
        self.vae = None
        self.is_loaded = False

    def load_model(self):
        """加载 EchoMimic V3 模型"""
        if self.is_loaded:
            return True

        logger.info(f"加载 EchoMimic V3 模型到 {self.device}...")

        try:
            # 添加源码路径
            echomimic_src = Path("echomimic_v3_src")
            if str(echomimic_src) not in sys.path:
                sys.path.insert(0, str(echomimic_src))

            from omegaconf import OmegaConf
            from diffusers import FlowMatchEulerDiscreteScheduler
            from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor
            from einops import rearrange
            import librosa
            import pyloudnorm as pyln

            # 导入 V3 模块
            from src.wan_vae import AutoencoderKLWan
            from src.wan_image_encoder import CLIPModel
            from src.wan_text_encoder import WanT5EncoderModel
            from src.wan_transformer3d_audio_2512 import WanTransformerAudioMask3DModel as WanTransformer
            from src.pipeline_wan_fun_inpaint_audio_2512 import WanFunInpaintAudioPipeline
            from src.wav2vec2 import Wav2Vec2Model
            from src.fm_solvers_unipc import FlowUniPCMultistepScheduler
            from src.utils import filter_kwargs
            from src.cache_utils import get_teacache_coefficients

            # 设置精度
            weight_dtype = torch.bfloat16 if self.weight_dtype_str == "bfloat16" else torch.float16

            # 加载配置
            config_path = "config/wan2.1/wan_civitai.yaml"
            if not os.path.exists(config_path):
                config_path = "echomimic_v3_src/config/wan2.1/wan_civitai.yaml"
            model_config = OmegaConf.load(config_path)

            logger.info("  [1/7] 加载音频编码器 (wav2vec2)...")
            self.audio_encoder = Wav2Vec2Model.from_pretrained(
                self.wav2vec_model, local_files_only=True
            ).to('cpu')  # 音频处理在 CPU
            self.audio_encoder.feature_extractor._freeze_parameters()
            self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                self.wav2vec_model, local_files_only=True
            )

            logger.info("  [2/7] 加载 Transformer...")
            transformer = WanTransformer.from_pretrained(
                os.path.join(self.base_model, model_config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
                transformer_additional_kwargs=OmegaConf.to_container(model_config['transformer_additional_kwargs']),
                low_cpu_mem_usage=True,
                torch_dtype=weight_dtype,
            )

            # 加载 flash-pro 权重
            logger.info("  [3/7] 加载 flash-pro 权重...")
            from safetensors.torch import load_file
            state_dict = load_file(self.transformer_path)
            m, u = transformer.load_state_dict(state_dict, strict=False)
            logger.info(f"    missing keys: {len(m)}, unexpected keys: {len(u)}")

            logger.info("  [4/7] 加载 VAE...")
            self.vae = AutoencoderKLWan.from_pretrained(
                os.path.join(self.base_model, model_config['vae_kwargs'].get('vae_subpath', 'vae')),
                additional_kwargs=OmegaConf.to_container(model_config['vae_kwargs']),
            ).to(weight_dtype)

            logger.info("  [5/7] 加载 Tokenizer 和 Text Encoder...")
            tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(self.base_model, model_config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
            )
            text_encoder = WanT5EncoderModel.from_pretrained(
                os.path.join(self.base_model, model_config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
                additional_kwargs=OmegaConf.to_container(model_config['text_encoder_kwargs']),
                low_cpu_mem_usage=True,
                torch_dtype=weight_dtype,
            ).eval()

            logger.info("  [6/7] 加载 CLIP Image Encoder...")
            clip_image_encoder = CLIPModel.from_pretrained(
                os.path.join(self.base_model, model_config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
            ).to(weight_dtype).eval()

            logger.info("  [7/7] 创建 Pipeline...")
            # 使用 UniPC 调度器
            scheduler_kwargs = OmegaConf.to_container(model_config['scheduler_kwargs'])
            scheduler_kwargs['shift'] = 1
            scheduler = FlowUniPCMultistepScheduler(
                **filter_kwargs(FlowUniPCMultistepScheduler, scheduler_kwargs)
            )

            self.pipeline = WanFunInpaintAudioPipeline(
                transformer=transformer,
                vae=self.vae,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                scheduler=scheduler,
                clip_image_encoder=clip_image_encoder
            )

            self.pipeline.to(device=self.device)

            # 启用 TeaCache 加速
            if self.enable_teacache:
                coefficients = get_teacache_coefficients("Wan2.1-Fun-V1.1-1.3B-InP")
                if coefficients is not None:
                    logger.info(f"  启用 TeaCache (threshold={self.teacache_threshold})")
                    self.pipeline.transformer.enable_teacache(
                        coefficients,
                        self.num_inference_steps,
                        self.teacache_threshold,
                        num_skip_start_steps=5,
                        offload=False
                    )

            self.weight_dtype = weight_dtype
            self.is_loaded = True
            logger.info("✓ EchoMimic V3 模型加载成功")
            return True

        except Exception as e:
            logger.error(f"加载 EchoMimic V3 失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _get_audio_embed(self, audio_path: str, video_length: int):
        """提取音频特征"""
        import librosa
        import pyloudnorm as pyln
        from einops import rearrange

        # 加载音频
        mel_input, sr = librosa.load(audio_path, sr=16000)

        # 响度归一化
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(mel_input)
        if abs(loudness) <= 100:
            mel_input = pyln.normalize.loudness(mel_input, loudness, -23)

        # 截取到视频长度
        mel_input = mel_input[:int(video_length / self.fps * sr)]

        # 提取特征
        audio_feature = np.squeeze(
            self.wav2vec_feature_extractor(mel_input, sampling_rate=sr).input_values
        )
        audio_feature = torch.from_numpy(audio_feature).float().to('cpu')
        audio_feature = audio_feature.unsqueeze(0)

        with torch.no_grad():
            embeddings = self.audio_encoder(
                audio_feature, seq_len=int(video_length), output_hidden_states=True
            )

        audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
        audio_emb = rearrange(audio_emb, "b s d -> s b d")

        # 处理时间窗口
        indices = (torch.arange(2 * 2 + 1) - 2) * 1
        center_indices = torch.arange(0, video_length, 1).unsqueeze(1) + indices.unsqueeze(0)
        center_indices = torch.clamp(center_indices, min=0, max=audio_emb.shape[0] - 1)
        audio_embeds = audio_emb[center_indices]
        audio_embeds = audio_embeds.unsqueeze(0).to(device=self.device, dtype=self.weight_dtype)

        return audio_embeds

    def _get_sample_size(self, pil_img):
        """计算输出尺寸"""
        w, h = pil_img.size
        ori_a = w * h
        default_a = self.width * self.height

        if default_a < ori_a:
            ratio_a = math.sqrt(ori_a / self.width / self.height)
            w = w / ratio_a // 16 * 16
            h = h / ratio_a // 16 * 16
        else:
            w = w // 16 * 16
            h = h // 16 * 16

        return [int(h), int(w)]

    def generate(
        self,
        reference_image_path: str,
        audio_path: str,
        output_path: str,
        prompt: str = "A person is speaking.",
        max_frames: Optional[int] = None
    ) -> Optional[str]:
        """
        生成 Avatar 视频

        Args:
            reference_image_path: 参考图片路径
            audio_path: 驱动音频路径
            output_path: 输出视频路径
            prompt: 文本提示
            max_frames: 最大帧数

        Returns:
            生成的视频路径，失败返回 None
        """
        if not self.is_loaded:
            if not self.load_model():
                return None

        if max_frames is None:
            max_frames = self.max_frames

        try:
            from moviepy import VideoFileClip, AudioFileClip
            from src.utils import get_image_to_video_latent2, save_videos_grid

            logger.info(f"开始生成视频...")
            logger.info(f"  参考图片: {reference_image_path}")
            logger.info(f"  音频: {audio_path}")

            # 加载参考图片
            ref_image = Image.open(reference_image_path).convert("RGB")
            ref_start = np.array(ref_image)

            # 获取音频时长
            audio_clip = AudioFileClip(audio_path)
            video_length = min(int(audio_clip.duration * self.fps), max_frames)

            # 对齐到 VAE 时间压缩比
            temporal_ratio = self.vae.config.temporal_compression_ratio
            video_length = int((video_length - 1) // temporal_ratio * temporal_ratio) + 1 if video_length != 1 else 1

            logger.info(f"  视频长度: {video_length} 帧 ({video_length / self.fps:.1f} 秒)")

            # 提取音频特征
            logger.info("  提取音频特征...")
            audio_embeds = self._get_audio_embed(audio_path, video_length)

            # 准备输入
            validation_image_start = Image.fromarray(ref_start).convert("RGB")
            sample_size_0, sample_size_1 = self._get_sample_size(validation_image_start)

            logger.info(f"  输出尺寸: {sample_size_1}x{sample_size_0}")

            input_video, input_video_mask, clip_image = get_image_to_video_latent2(
                validation_image_start, None,
                video_length=video_length,
                sample_size=[sample_size_0, sample_size_1]
            )

            # 生成
            logger.info(f"  开始推理 ({self.num_inference_steps} 步)...")
            generator = torch.Generator(device=self.device).manual_seed(42)

            negative_prompt = "Gesture is bad. Gesture is unclear. Strange and twisted hands. Bad hands. Bad fingers."

            with torch.no_grad():
                sample = self.pipeline(
                    prompt,
                    num_frames=video_length,
                    negative_prompt=negative_prompt,
                    audio_embeds=audio_embeds,
                    audio_scale=1.0,
                    ip_mask=None,
                    use_un_ip_mask=False,
                    height=sample_size_0,
                    width=sample_size_1,
                    generator=generator,
                    neg_scale=1.0,
                    neg_steps=0,
                    use_dynamic_cfg=False,
                    use_dynamic_acfg=False,
                    guidance_scale=self.guidance_scale,
                    audio_guidance_scale=self.audio_guidance_scale,
                    num_inference_steps=self.num_inference_steps,
                    video=input_video,
                    mask_video=input_video_mask,
                    clip_image=clip_image,
                    cfg_skip_ratio=0.0,
                    shift=5.0,
                ).videos

            # 保存临时视频
            tmp_video_path = output_path.replace('.mp4', '_tmp.mp4')
            save_videos_grid(sample[:, :, :video_length], tmp_video_path, fps=self.fps)

            # 添加音频
            logger.info("  合并音频...")
            video_clip = VideoFileClip(tmp_video_path)
            audio_clip = audio_clip.subclipped(0, video_length / self.fps)
            video_clip = video_clip.with_audio(audio_clip)
            video_clip.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                threads=2,
                logger=None
            )

            # 清理
            os.remove(tmp_video_path)
            video_clip.close()
            audio_clip.close()

            logger.info(f"✓ 视频生成完成: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"生成视频失败: {e}")
            import traceback
            traceback.print_exc()
            return None


# ============================================================
# TTS 语音合成服务
# ============================================================
class TTSService:
    """TTS 语音合成服务 (使用 edge-tts)"""

    def __init__(self, config: dict):
        self.config = config
        self.default_voice = config.get('default_voice', 'zh-CN-XiaoxiaoNeural')
        self.is_loaded = True  # edge-tts 无需加载模型

    async def synthesize(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None
    ) -> Optional[str]:
        """
        合成语音

        Args:
            text: 要合成的文本
            output_path: 输出音频路径
            voice: 声音名称

        Returns:
            生成的音频路径，失败返回 None
        """
        try:
            import edge_tts

            voice = voice or self.default_voice
            logger.info(f"TTS 合成: {text[:30]}... (voice={voice})")

            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_path)

            logger.info(f"✓ TTS 合成完成: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"TTS 合成失败: {e}")
            return None


# ============================================================
# FastAPI 应用
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global config, avatar_service, tts_service

    logger.info("正在启动 EchoSelf GPU Server (EchoMimic V3)...")

    config = load_config()

    # 创建目录
    Path(config['paths']['temp_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['output_dir']).mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)

    # 初始化服务
    avatar_service = AvatarServiceV3(config['avatar'])
    tts_service = TTSService(config['tts'])

    logger.info("✓ EchoSelf GPU Server 启动完成")
    logger.info("  模型将在首次请求时加载（预热）")

    yield

    logger.info("正在关闭 EchoSelf GPU Server...")


app = FastAPI(
    title="EchoSelf GPU Server",
    description="提供 TTS 语音合成和 Avatar 视频生成 API (基于 EchoMimic V3)",
    version="2.0.0",
    lifespan=lifespan
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "version": "2.0.0 (EchoMimic V3)",
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
        "avatar_loaded": avatar_service.is_loaded if avatar_service else False,
        "tts_loaded": tts_service.is_loaded if tts_service else False,
    }


@app.post("/tts")
async def text_to_speech(
    text: str = Form(...),
    voice: str = Form("zh-CN-XiaoxiaoNeural"),
):
    """
    TTS 语音合成

    - text: 要合成的文本
    - voice: 声音名称 (edge-tts)

    常用中文声音:
    - zh-CN-XiaoxiaoNeural (女)
    - zh-CN-YunxiNeural (男)
    - zh-CN-YunjianNeural (男)
    """
    request_id = str(uuid.uuid4())[:8]
    temp_dir = Path(config['paths']['temp_dir']) / request_id
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        output_path = str(temp_dir / "output.wav")
        result = await tts_service.synthesize(
            text=text,
            output_path=output_path,
            voice=voice
        )

        if result and os.path.exists(result):
            return FileResponse(
                result,
                media_type="audio/wav",
                filename="tts_output.wav"
            )
        else:
            raise HTTPException(status_code=500, detail="TTS 合成失败")

    except Exception as e:
        logger.error(f"TTS 请求失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/avatar")
async def generate_avatar(
    audio: UploadFile = File(...),
    reference_image: UploadFile = File(...),
    prompt: str = Form("A person is speaking."),
    max_frames: int = Form(81)
):
    """
    Avatar 视频生成 (EchoMimic V3)

    - audio: 驱动音频文件
    - reference_image: 参考人像图片
    - prompt: 文本提示 (可选)
    - max_frames: 最大帧数（默认 81，约 3.2 秒）

    V3 优势:
    - 8 步推理，比 V2 快 4 倍
    - 无需 pose 数据
    - 12GB 显存即可运行
    """
    request_id = str(uuid.uuid4())[:8]
    temp_dir = Path(config['paths']['temp_dir']) / request_id
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 保存上传的文件
        audio_path = str(temp_dir / "audio.wav")
        image_path = str(temp_dir / "reference.png")

        with open(audio_path, "wb") as f:
            f.write(await audio.read())

        with open(image_path, "wb") as f:
            f.write(await reference_image.read())

        # 生成视频
        output_path = str(temp_dir / "output.mp4")
        result = avatar_service.generate(
            reference_image_path=image_path,
            audio_path=audio_path,
            output_path=output_path,
            prompt=prompt,
            max_frames=max_frames
        )

        if result and os.path.exists(result):
            return FileResponse(
                result,
                media_type="video/mp4",
                filename="avatar_output.mp4"
            )
        else:
            raise HTTPException(status_code=500, detail="Avatar 生成失败")

    except Exception as e:
        logger.error(f"Avatar 请求失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/warmup")
async def warmup():
    """
    预热模型（首次调用前建议先预热）
    """
    logger.info("开始预热模型...")

    if not avatar_service.is_loaded:
        success = avatar_service.load_model()
        if success:
            return {"status": "ok", "message": "模型预热完成"}
        else:
            raise HTTPException(status_code=500, detail="模型加载失败")
    else:
        return {"status": "ok", "message": "模型已加载"}


@app.get("/info")
async def get_info():
    """获取服务信息"""
    return {
        "name": "EchoSelf GPU Server",
        "version": "2.0.0",
        "avatar_model": "EchoMimic V3 flash-pro",
        "tts_engine": "edge-tts",
        "features": {
            "inference_steps": config['avatar']['num_inference_steps'],
            "max_resolution": f"{config['avatar']['width']}x{config['avatar']['height']}",
            "fps": config['avatar']['fps'],
            "teacache": config['avatar']['enable_teacache'],
        },
        "advantages": [
            "8 步推理（比 V2 快 4 倍）",
            "无需 pose 数据",
            "无需 face mask",
            "12GB 显存即可运行",
        ]
    }


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    import uvicorn

    cfg = load_config()
    host = cfg['server']['host']
    port = cfg['server']['port']

    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
