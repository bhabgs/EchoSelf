"""
EchoSelf GPU Server - 提供 TTS 和 Avatar 生成 API

运行: python api_server.py
文档: http://localhost:8000/docs
"""

import os
import sys
import uuid
import tempfile
import shutil
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
avatar_pipeline = None
tts_model = None


# ============================================================
# 配置加载
# ============================================================
def load_config(config_path: str = "config.yaml") -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# ============================================================
# EchoMimic V2 Avatar 生成器
# ============================================================
class AvatarService:
    """EchoMimic V2 Avatar 生成服务"""

    def __init__(self, config: dict):
        self.config = config
        self.device = config.get('device', 'cuda:1')
        self.model_dir = Path(config.get('model_dir', './models/echomimic_v2'))
        self.width = config.get('width', 768)
        self.height = config.get('height', 768)
        self.fps = config.get('fps', 24)
        self.steps = config.get('steps', 30)
        self.cfg_scale = config.get('cfg_scale', 2.5)
        self.pose_dir = config.get('default_pose_dir', './echomimic_v2_src/assets/halfbody_demo/pose/01')

        self.pipe = None
        self.audio_processor = None
        self.pose_net = None
        self.is_loaded = False

    def load_model(self):
        """加载 EchoMimic V2 模型"""
        if self.is_loaded:
            return True

        logger.info(f"加载 EchoMimic V2 模型到 {self.device}...")

        try:
            # 添加源码路径
            echomimic_src = Path("echomimic_v2_src")
            if str(echomimic_src) not in sys.path:
                sys.path.insert(0, str(echomimic_src))

            from diffusers import AutoencoderKL, DDIMScheduler
            from src.models.unet_2d_condition import UNet2DConditionModel
            from src.models.unet_3d_emo import EMOUNet3DConditionModel
            from src.models.whisper.audio2feature import load_audio_model
            from src.pipelines.pipeline_echomimicv2 import EchoMimicV2Pipeline
            from src.models.pose_encoder import PoseEncoder

            dtype = torch.float16

            # 加载 VAE
            vae = AutoencoderKL.from_pretrained(
                str(self.model_dir / "sd-vae-ft-mse")
            ).to(self.device, dtype=dtype)

            # 加载 Reference UNet
            reference_unet = UNet2DConditionModel.from_pretrained(
                str(self.model_dir / "sd-image-variations-diffusers"),
                subfolder="unet",
                use_safetensors=False
            ).to(dtype=dtype, device=self.device)
            reference_unet.load_state_dict(
                torch.load(str(self.model_dir / "reference_unet.pth"), weights_only=True)
            )

            # 加载 Denoising UNet
            denoising_unet = EMOUNet3DConditionModel.from_pretrained_2d(
                str(self.model_dir / "sd-image-variations-diffusers"),
                str(self.model_dir / "motion_module.pth"),
                subfolder="unet",
                unet_additional_kwargs={
                    "use_inflated_groupnorm": True,
                    "unet_use_cross_frame_attention": False,
                    "unet_use_temporal_attention": False,
                    "use_motion_module": True,
                    "cross_attention_dim": 384,
                    "motion_module_resolutions": [1, 2, 4, 8],
                    "motion_module_mid_block": True,
                    "motion_module_decoder_only": False,
                    "motion_module_type": "Vanilla",
                    "motion_module_kwargs": {
                        "num_attention_heads": 8,
                        "num_transformer_block": 1,
                        "attention_block_types": ['Temporal_Self', 'Temporal_Self'],
                        "temporal_position_encoding": True,
                        "temporal_position_encoding_max_len": 32,
                        "temporal_attention_dim_div": 1,
                    }
                },
            ).to(dtype=dtype, device=self.device)
            denoising_unet.load_state_dict(
                torch.load(str(self.model_dir / "denoising_unet.pth"), weights_only=True),
                strict=False
            )

            # 加载 Pose Encoder
            self.pose_net = PoseEncoder(
                320, conditioning_channels=3, block_out_channels=(16, 32, 96, 256)
            ).to(dtype=dtype, device=self.device)
            self.pose_net.load_state_dict(
                torch.load(str(self.model_dir / "pose_encoder.pth"), weights_only=True)
            )

            # 加载音频处理器
            self.audio_processor = load_audio_model(
                model_path=str(self.model_dir / "audio_processor" / "tiny.pt"),
                device=self.device
            )

            # 创建调度器
            scheduler = DDIMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="linear",
                clip_sample=False,
                steps_offset=1,
                prediction_type="v_prediction",
                rescale_betas_zero_snr=True,
                timestep_spacing="trailing"
            )

            # 创建 Pipeline
            self.pipe = EchoMimicV2Pipeline(
                vae=vae,
                reference_unet=reference_unet,
                denoising_unet=denoising_unet,
                audio_guider=self.audio_processor,
                pose_encoder=self.pose_net,
                scheduler=scheduler,
            ).to(self.device, dtype=dtype)

            self.is_loaded = True
            logger.info("✓ EchoMimic V2 模型加载成功")
            return True

        except Exception as e:
            logger.error(f"加载 EchoMimic V2 失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def generate(
        self,
        reference_image_path: str,
        audio_path: str,
        output_path: str,
        max_frames: int = 240
    ) -> Optional[str]:
        """
        生成 Avatar 视频

        Args:
            reference_image_path: 参考图片路径
            audio_path: 驱动音频路径
            output_path: 输出视频路径
            max_frames: 最大帧数

        Returns:
            生成的视频路径，失败返回 None
        """
        if not self.is_loaded:
            if not self.load_model():
                return None

        try:
            from moviepy.editor import VideoFileClip, AudioFileClip
            from src.utils.util import save_videos_grid
            from src.utils.dwpose_util import draw_pose_select_v2

            dtype = torch.float16

            # 加载参考图片
            ref_image = Image.open(reference_image_path).convert('RGB')
            ref_image = ref_image.resize((self.width, self.height))

            # 获取音频时长
            audio_clip = AudioFileClip(audio_path)
            audio_duration = audio_clip.duration

            # 计算帧数
            num_frames = min(int(audio_duration * self.fps), max_frames)
            num_frames = min(num_frames, len(os.listdir(self.pose_dir)))

            logger.info(f"生成参数: {self.width}x{self.height}, {num_frames}帧, {self.fps}fps")

            # 加载姿态数据
            pose_list = []
            for idx in range(num_frames):
                tgt_musk = np.zeros((self.width, self.height, 3)).astype('uint8')
                pose_path = os.path.join(self.pose_dir, f"{idx}.npy")

                if os.path.exists(pose_path):
                    detected_pose = np.load(pose_path, allow_pickle=True).tolist()
                    imh_new, imw_new, rb, re, cb, ce = detected_pose['draw_pose_params']
                    im = draw_pose_select_v2(detected_pose, imh_new, imw_new, ref_w=800)
                    im = np.transpose(np.array(im), (1, 2, 0))
                    tgt_musk[rb:re, cb:ce, :] = im

                tgt_musk_pil = Image.fromarray(tgt_musk).convert('RGB')
                pose_tensor = torch.Tensor(np.array(tgt_musk_pil)).to(
                    dtype=dtype, device=self.device
                ).permute(2, 0, 1) / 255.0
                pose_list.append(pose_tensor)

            poses_tensor = torch.stack(pose_list, dim=1).unsqueeze(0)

            # 设置音频时长
            audio_clip = audio_clip.set_duration(num_frames / self.fps)

            # 生成视频
            generator = torch.manual_seed(42)

            with torch.no_grad():
                video = self.pipe(
                    ref_image,
                    audio_path,
                    poses_tensor[:, :, :num_frames, ...],
                    self.width,
                    self.height,
                    num_frames,
                    self.steps,
                    self.cfg_scale,
                    generator=generator,
                    audio_sample_rate=16000,
                    context_frames=12,
                    fps=self.fps,
                    context_overlap=3,
                    start_idx=0,
                ).videos

            # 保存无音频视频
            temp_video = output_path.replace('.mp4', '_temp.mp4')
            final_length = min(video.shape[2], num_frames)
            video_sig = video[:, :, :final_length, :, :]

            save_videos_grid(video_sig, temp_video, n_rows=1, fps=self.fps)

            # 添加音频
            video_clip = VideoFileClip(temp_video)
            video_clip = video_clip.set_audio(audio_clip)
            video_clip.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                threads=2,
                logger=None
            )

            # 清理临时文件
            os.remove(temp_video)
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
    """TTS 语音合成服务"""

    def __init__(self, config: dict):
        self.config = config
        self.device = config.get('device', 'cuda:0')
        self.model_dir = Path(config.get('model_dir', './models/fish-speech-1.5'))
        self.sample_rate = config.get('sample_rate', 44100)

        self.model = None
        self.is_loaded = False
        self.use_edge_tts = False

    def load_model(self):
        """加载 TTS 模型"""
        if self.is_loaded:
            return True

        logger.info(f"加载 TTS 模型到 {self.device}...")

        # 尝试加载 Fish Speech
        try:
            # Fish Speech 加载逻辑
            # TODO: 实现 Fish Speech 加载
            logger.warning("Fish Speech 暂未实现，使用 edge-tts 备选方案")
            self.use_edge_tts = True
            self.is_loaded = True
            return True

        except Exception as e:
            logger.warning(f"加载 Fish Speech 失败: {e}，使用 edge-tts")
            self.use_edge_tts = True
            self.is_loaded = True
            return True

    async def synthesize(
        self,
        text: str,
        output_path: str,
        reference_audio: Optional[str] = None,
        voice: str = "zh-CN-XiaoxiaoNeural"
    ) -> Optional[str]:
        """
        合成语音

        Args:
            text: 要合成的文本
            output_path: 输出音频路径
            reference_audio: 参考音频（用于声音克隆）
            voice: edge-tts 声音名称

        Returns:
            生成的音频路径，失败返回 None
        """
        if not self.is_loaded:
            self.load_model()

        try:
            if self.use_edge_tts:
                import edge_tts

                communicate = edge_tts.Communicate(text, voice)
                await communicate.save(output_path)

                logger.info(f"✓ TTS 合成完成: {output_path}")
                return output_path
            else:
                # Fish Speech 合成逻辑
                # TODO: 实现 Fish Speech 合成
                pass

        except Exception as e:
            logger.error(f"TTS 合成失败: {e}")
            return None


# ============================================================
# FastAPI 应用
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global config, avatar_pipeline, tts_model

    # 启动时加载配置和模型
    logger.info("正在启动 EchoSelf GPU Server...")

    config = load_config()

    # 创建目录
    Path(config['paths']['temp_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['output_dir']).mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)

    # 初始化服务
    avatar_pipeline = AvatarService(config['avatar'])
    tts_model = TTSService(config['tts'])

    # 预加载模型（可选，启动时加载会慢但首次请求快）
    # avatar_pipeline.load_model()
    # tts_model.load_model()

    logger.info("✓ EchoSelf GPU Server 启动完成")

    yield

    # 关闭时清理
    logger.info("正在关闭 EchoSelf GPU Server...")


app = FastAPI(
    title="EchoSelf GPU Server",
    description="提供 TTS 语音合成和 Avatar 视频生成 API",
    version="1.0.0",
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
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "avatar_loaded": avatar_pipeline.is_loaded if avatar_pipeline else False,
        "tts_loaded": tts_model.is_loaded if tts_model else False,
    }


@app.post("/tts")
async def text_to_speech(
    text: str = Form(...),
    voice: str = Form("zh-CN-XiaoxiaoNeural"),
    reference_audio: Optional[UploadFile] = File(None)
):
    """
    TTS 语音合成

    - text: 要合成的文本
    - voice: 声音名称 (edge-tts)
    - reference_audio: 参考音频文件（用于声音克隆，可选）
    """
    request_id = str(uuid.uuid4())[:8]
    temp_dir = Path(config['paths']['temp_dir']) / request_id
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 保存参考音频（如果有）
        ref_audio_path = None
        if reference_audio:
            ref_audio_path = str(temp_dir / "reference.wav")
            with open(ref_audio_path, "wb") as f:
                f.write(await reference_audio.read())

        # 合成
        output_path = str(temp_dir / "output.wav")
        result = await tts_model.synthesize(
            text=text,
            output_path=output_path,
            reference_audio=ref_audio_path,
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
    finally:
        # 延迟清理（让文件有时间被读取）
        # shutil.rmtree(temp_dir, ignore_errors=True)
        pass


@app.post("/avatar")
async def generate_avatar(
    audio: UploadFile = File(...),
    reference_image: UploadFile = File(...),
    max_frames: int = Form(240)
):
    """
    Avatar 视频生成

    - audio: 驱动音频文件
    - reference_image: 参考人像图片
    - max_frames: 最大帧数（默认 240，约 10 秒）
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
        result = avatar_pipeline.generate(
            reference_image_path=image_path,
            audio_path=audio_path,
            output_path=output_path,
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


@app.get("/models")
async def list_models():
    """列出可用模型"""
    return {
        "tts": {
            "fish_speech": os.path.exists(config['tts']['model_dir']),
            "edge_tts": True,
        },
        "avatar": {
            "echomimic_v2": os.path.exists(config['avatar']['model_dir']),
        }
    }


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    import uvicorn

    # 加载配置获取端口
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
