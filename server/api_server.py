"""
EchoSelf GPU Server - API 网关 (基于 Duix-Avatar)

Duix-Avatar 提供三个核心服务:
- TTS (端口 18180): Fish Speech 语音合成/克隆
- ASR (端口 10095): FunASR 语音识别
- Video (端口 8383): 数字人视频生成

本服务作为 API 网关，提供统一的接口。

运行: python api_server.py
文档: http://localhost:8000/docs
"""

import os
import sys
import uuid
import time
import asyncio
import shutil
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import yaml
import httpx
import aiofiles
from loguru import logger
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
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
http_client = None


# ============================================================
# 配置加载
# ============================================================
def load_config(config_path: str = "config.yaml") -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 支持环境变量覆盖
    cfg['duix']['tts_url'] = os.getenv('DUIX_TTS_URL', cfg['duix']['tts_url'])
    cfg['duix']['video_url'] = os.getenv('DUIX_VIDEO_URL', cfg['duix']['video_url'])
    cfg['duix']['asr_url'] = os.getenv('DUIX_ASR_URL', cfg['duix']['asr_url'])

    return cfg


# ============================================================
# Duix-Avatar 客户端
# ============================================================
class DuixAvatarClient:
    """Duix-Avatar API 客户端"""

    def __init__(self, config: dict):
        self.tts_url = config['duix']['tts_url']
        self.video_url = config['duix']['video_url']
        self.asr_url = config['duix']['asr_url']
        self.tts_config = config.get('tts', {})
        self.video_config = config.get('video', {})
        self.paths = config['paths']

        # 说话人信息缓存 (用于声音克隆)
        self.speakers = {}

    async def check_services(self) -> dict:
        """检查各服务状态"""
        status = {
            "tts": False,
            "video": False,
            "asr": False
        }

        async with httpx.AsyncClient(timeout=5.0) as client:
            # 检查 TTS
            try:
                resp = await client.get(f"{self.tts_url}/health")
                status["tts"] = resp.status_code == 200
            except:
                pass

            # 检查 Video
            try:
                resp = await client.get(f"{self.video_url}/health")
                status["video"] = resp.status_code == 200
            except:
                pass

            # 检查 ASR
            try:
                resp = await client.get(f"{self.asr_url}/health")
                status["asr"] = resp.status_code == 200
            except:
                pass

        return status

    async def register_speaker(
        self,
        speaker_id: str,
        reference_audio_path: str,
        reference_text: str
    ) -> dict:
        """
        注册说话人 (用于声音克隆)

        Args:
            speaker_id: 说话人唯一标识
            reference_audio_path: 参考音频路径
            reference_text: 参考音频对应的文本

        Returns:
            注册结果
        """
        # 将音频复制到 Duix 数据目录
        voice_data_dir = Path(self.paths['voice_data'])
        voice_data_dir.mkdir(parents=True, exist_ok=True)

        target_audio = voice_data_dir / f"{speaker_id}.wav"
        shutil.copy(reference_audio_path, target_audio)

        # 保存说话人信息
        self.speakers[speaker_id] = {
            "reference_audio": str(target_audio),
            "reference_text": reference_text,
            "asr_format_audio_url": f"/code/data/{speaker_id}.wav"
        }

        logger.info(f"✓ 注册说话人: {speaker_id}")
        return self.speakers[speaker_id]

    async def synthesize_speech(
        self,
        text: str,
        speaker_id: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        TTS 语音合成

        Args:
            text: 要合成的文本
            speaker_id: 说话人 ID (用于声音克隆)
            output_path: 输出路径

        Returns:
            生成的音频路径
        """
        if not text:
            return None

        if output_path is None:
            output_path = f"temp/{uuid.uuid4()}.wav"

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # 构建请求参数
        params = {
            "speaker": speaker_id or str(uuid.uuid4()),
            "text": text,
            "format": self.tts_config.get("format", "wav"),
            "topP": self.tts_config.get("topP", 0.7),
            "max_new_tokens": self.tts_config.get("max_new_tokens", 1024),
            "chunk_length": self.tts_config.get("chunk_length", 100),
            "repetition_penalty": self.tts_config.get("repetition_penalty", 1.2),
            "temperature": self.tts_config.get("temperature", 0.7),
            "need_asr": False,
            "streaming": False,
            "is_fixed_seed": 0,
            "is_norm": 0,
        }

        # 如果有说话人信息，添加参考音频
        if speaker_id and speaker_id in self.speakers:
            speaker_info = self.speakers[speaker_id]
            params["reference_audio"] = speaker_info["asr_format_audio_url"]
            params["reference_text"] = speaker_info["reference_text"]

        logger.info(f"TTS 合成: {text[:30]}...")

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self.tts_url}/v1/invoke",
                    json=params
                )

                if resp.status_code == 200:
                    # 保存音频
                    async with aiofiles.open(output_path, 'wb') as f:
                        await f.write(resp.content)

                    logger.info(f"✓ TTS 合成完成: {output_path}")
                    return output_path
                else:
                    logger.error(f"TTS 失败: {resp.status_code} - {resp.text}")
                    return None

        except Exception as e:
            logger.error(f"TTS 请求异常: {e}")
            return None

    async def generate_video(
        self,
        audio_path: str,
        video_path: str,
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        生成数字人视频

        Args:
            audio_path: 驱动音频路径
            video_path: 参考视频/图片路径 (需要在 Duix 数据目录中)
            output_path: 输出路径

        Returns:
            生成的视频路径
        """
        task_code = str(uuid.uuid4())

        if output_path is None:
            output_path = f"outputs/{task_code}.mp4"

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # 构建请求参数
        params = {
            "audio_url": audio_path,
            "video_url": video_path,
            "code": task_code,
            "chaofen": self.video_config.get("chaofen", 0),
            "watermark_switch": self.video_config.get("watermark_switch", 0),
            "pn": self.video_config.get("pn", 1)
        }

        logger.info(f"开始生成视频 (task: {task_code})...")

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                # 提交任务
                resp = await client.post(
                    f"{self.video_url}/easy/submit",
                    json=params
                )

                if resp.status_code != 200:
                    logger.error(f"提交任务失败: {resp.status_code} - {resp.text}")
                    return None

                # 轮询查询进度
                max_wait = 300  # 最大等待 5 分钟
                start_time = time.time()

                while time.time() - start_time < max_wait:
                    await asyncio.sleep(2)  # 每 2 秒查询一次

                    query_resp = await client.get(
                        f"{self.video_url}/easy/query",
                        params={"code": task_code}
                    )

                    if query_resp.status_code == 200:
                        result = query_resp.json()

                        # 检查任务状态
                        status = result.get("status", "")
                        progress = result.get("progress", 0)

                        logger.info(f"  进度: {progress}% - {status}")

                        if status == "completed" or progress >= 100:
                            # 下载生成的视频
                            video_url = result.get("video_url")
                            if video_url:
                                video_resp = await client.get(video_url)
                                if video_resp.status_code == 200:
                                    async with aiofiles.open(output_path, 'wb') as f:
                                        await f.write(video_resp.content)

                                    logger.info(f"✓ 视频生成完成: {output_path}")
                                    return output_path

                            # 如果没有 video_url，可能结果在本地
                            local_output = result.get("output_path")
                            if local_output and os.path.exists(local_output):
                                shutil.copy(local_output, output_path)
                                logger.info(f"✓ 视频生成完成: {output_path}")
                                return output_path

                            return None

                        elif status == "failed":
                            logger.error(f"视频生成失败: {result.get('error', 'Unknown error')}")
                            return None

                logger.error("视频生成超时")
                return None

        except Exception as e:
            logger.error(f"视频生成异常: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def clone_voice(
        self,
        speaker_id: str,
        reference_audio_path: str
    ) -> dict:
        """
        克隆声音

        Args:
            speaker_id: 说话人 ID
            reference_audio_path: 参考音频路径

        Returns:
            克隆结果
        """
        # 先进行 ASR 识别参考音频的文本
        reference_text = await self.transcribe(reference_audio_path)

        if not reference_text:
            reference_text = "这是一段参考音频。"  # 默认文本

        # 注册说话人
        return await self.register_speaker(
            speaker_id=speaker_id,
            reference_audio_path=reference_audio_path,
            reference_text=reference_text
        )

    async def transcribe(self, audio_path: str) -> Optional[str]:
        """
        语音识别 (ASR)

        Args:
            audio_path: 音频文件路径

        Returns:
            识别的文本
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                with open(audio_path, 'rb') as f:
                    files = {"audio": f}
                    resp = await client.post(
                        f"{self.asr_url}/transcribe",
                        files=files
                    )

                if resp.status_code == 200:
                    result = resp.json()
                    text = result.get("text", "")
                    logger.info(f"✓ ASR 识别: {text[:50]}...")
                    return text
                else:
                    logger.warning(f"ASR 失败: {resp.status_code}")
                    return None

        except Exception as e:
            logger.warning(f"ASR 请求异常: {e}")
            return None


# ============================================================
# FastAPI 应用
# ============================================================
duix_client: Optional[DuixAvatarClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global config, duix_client

    logger.info("正在启动 EchoSelf GPU Server (Duix-Avatar)...")

    config = load_config()

    # 创建目录
    Path(config['paths']['temp_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['output_dir']).mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)

    # 初始化 Duix 客户端
    duix_client = DuixAvatarClient(config)

    # 检查服务状态
    status = await duix_client.check_services()
    logger.info(f"Duix 服务状态: TTS={status['tts']}, Video={status['video']}, ASR={status['asr']}")

    logger.info("✓ EchoSelf GPU Server 启动完成")

    yield

    logger.info("正在关闭 EchoSelf GPU Server...")


app = FastAPI(
    title="EchoSelf GPU Server",
    description="基于 Duix-Avatar 的数字人视频生成 API",
    version="3.0.0",
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
    status = await duix_client.check_services()
    return {
        "status": "ok" if all(status.values()) else "degraded",
        "version": "3.0.0 (Duix-Avatar)",
        "services": status
    }


@app.post("/tts")
async def text_to_speech(
    text: str = Form(...),
    speaker_id: Optional[str] = Form(None),
):
    """
    TTS 语音合成 (Fish Speech)

    - text: 要合成的文本
    - speaker_id: 说话人 ID (可选，用于声音克隆)
    """
    request_id = str(uuid.uuid4())[:8]
    output_path = f"temp/{request_id}.wav"

    try:
        result = await duix_client.synthesize_speech(
            text=text,
            speaker_id=speaker_id,
            output_path=output_path
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
    reference_video: UploadFile = File(...),
):
    """
    Avatar 视频生成

    - audio: 驱动音频文件
    - reference_video: 参考视频/图片文件

    注意: 参考视频需要先通过 /train 接口训练模型
    """
    request_id = str(uuid.uuid4())[:8]
    temp_dir = Path(config['paths']['temp_dir']) / request_id
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 保存上传的文件
        audio_path = str(temp_dir / "audio.wav")
        video_path = str(temp_dir / "reference.mp4")

        async with aiofiles.open(audio_path, 'wb') as f:
            await f.write(await audio.read())

        async with aiofiles.open(video_path, 'wb') as f:
            await f.write(await reference_video.read())

        # 复制到 Duix 数据目录
        duix_data_dir = Path(config['paths']['video_data'])
        duix_data_dir.mkdir(parents=True, exist_ok=True)

        duix_audio = duix_data_dir / f"{request_id}_audio.wav"
        duix_video = duix_data_dir / f"{request_id}_video.mp4"

        shutil.copy(audio_path, duix_audio)
        shutil.copy(video_path, duix_video)

        # 生成视频
        output_path = f"outputs/{request_id}.mp4"
        result = await duix_client.generate_video(
            audio_path=str(duix_audio),
            video_path=str(duix_video),
            output_path=output_path
        )

        if result and os.path.exists(result):
            return FileResponse(
                result,
                media_type="video/mp4",
                filename="avatar_output.mp4"
            )
        else:
            raise HTTPException(status_code=500, detail="视频生成失败")

    except Exception as e:
        logger.error(f"Avatar 请求失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clone-voice")
async def clone_voice(
    speaker_id: str = Form(...),
    reference_audio: UploadFile = File(...),
):
    """
    声音克隆

    - speaker_id: 说话人唯一标识
    - reference_audio: 参考音频文件 (建议 5-30 秒)
    """
    temp_path = f"temp/{speaker_id}_reference.wav"

    try:
        # 保存参考音频
        async with aiofiles.open(temp_path, 'wb') as f:
            await f.write(await reference_audio.read())

        # 克隆声音
        result = await duix_client.clone_voice(
            speaker_id=speaker_id,
            reference_audio_path=temp_path
        )

        return {
            "status": "ok",
            "speaker_id": speaker_id,
            "message": "声音克隆成功，可在 /tts 接口中使用此 speaker_id"
        }

    except Exception as e:
        logger.error(f"声音克隆失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/asr")
async def speech_to_text(
    audio: UploadFile = File(...),
):
    """
    语音识别 (ASR)

    - audio: 音频文件
    """
    temp_path = f"temp/{uuid.uuid4()}.wav"

    try:
        # 保存音频
        async with aiofiles.open(temp_path, 'wb') as f:
            await f.write(await audio.read())

        # 识别
        text = await duix_client.transcribe(temp_path)

        if text:
            return {"status": "ok", "text": text}
        else:
            raise HTTPException(status_code=500, detail="语音识别失败")

    except Exception as e:
        logger.error(f"ASR 请求失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/info")
async def get_info():
    """获取服务信息"""
    status = await duix_client.check_services()
    return {
        "name": "EchoSelf GPU Server",
        "version": "3.0.0",
        "backend": "Duix-Avatar",
        "services": {
            "tts": {
                "name": "Fish Speech",
                "url": config['duix']['tts_url'],
                "available": status['tts']
            },
            "video": {
                "name": "Duix Avatar",
                "url": config['duix']['video_url'],
                "available": status['video']
            },
            "asr": {
                "name": "FunASR",
                "url": config['duix']['asr_url'],
                "available": status['asr']
            }
        },
        "features": [
            "TTS 语音合成",
            "声音克隆",
            "数字人视频生成",
            "语音识别 (ASR)"
        ]
    }


@app.get("/speakers")
async def list_speakers():
    """列出已注册的说话人"""
    return {
        "speakers": list(duix_client.speakers.keys())
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
