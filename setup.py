#!/usr/bin/env python3
"""
EchoSelf - 安装和设置脚本
Setup Script for EchoSelf Digital Avatar System

功能：
1. 检查系统依赖
2. 下载必要的模型
3. 验证配置
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.absolute()


def print_header(text):
    """打印格式化标题"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def check_python_version():
    """检查 Python 版本"""
    print_header("检查 Python 版本")
    version = sys.version_info
    print(f"当前版本: Python {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("错误: 需要 Python 3.10 或更高版本")
        return False

    print("Python 版本检查通过")
    return True


def check_cuda():
    """检查 CUDA 是否可用"""
    print_header("检查 CUDA/GPU")

    try:
        import torch
        if torch.cuda.is_available():
            print(f"CUDA 可用")
            print(f"CUDA 版本: {torch.version.cuda}")
            print(f"GPU 设备: {torch.cuda.get_device_name(0)}")
            print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return True
        else:
            print("警告: CUDA 不可用，将使用 CPU（速度较慢）")
            return False
    except ImportError:
        print("PyTorch 未安装")
        return False


def check_ffmpeg():
    """检查 ffmpeg 是否安装"""
    print_header("检查 FFmpeg")

    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"FFmpeg 已安装: {version_line}")
            return True
    except FileNotFoundError:
        pass

    print("警告: FFmpeg 未安装")
    print("请安装 FFmpeg:")
    if platform.system() == "Darwin":
        print("  brew install ffmpeg")
    elif platform.system() == "Linux":
        print("  sudo apt install ffmpeg")
    elif platform.system() == "Windows":
        print("  choco install ffmpeg 或下载安装包")

    return False


def check_ollama_connection():
    """检查 Ollama 服务器连接"""
    print_header("检查 Ollama 连接")

    ollama_host = os.getenv("OLLAMA_HOST", "http://192.168.50.218:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")

    print(f"服务器地址: {ollama_host}")
    print(f"目标模型: {ollama_model}")

    try:
        import requests
        response = requests.get(f"{ollama_host}/api/tags", timeout=10)

        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            print(f"连接成功！可用模型: {model_names}")

            if not any(ollama_model in name for name in model_names):
                print(f"警告: 目标模型 '{ollama_model}' 未找到")
                print("请在 Ollama 服务器上运行: ollama pull " + ollama_model)

            return True
        else:
            print(f"连接失败，状态码: {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"无法连接到 {ollama_host}")
        print("请确保:")
        print("  1. Ollama 服务器正在运行")
        print("  2. 网络连接正常（检查防火墙）")
        print("  3. 服务器地址正确")
        return False
    except Exception as e:
        print(f"检查连接时出错: {e}")
        return False


def install_dependencies():
    """安装 Python 依赖"""
    print_header("安装 Python 依赖")

    requirements_file = PROJECT_ROOT / "requirements.txt"

    if not requirements_file.exists():
        print("错误: requirements.txt 不存在")
        return False

    print("正在安装依赖...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
        capture_output=False
    )

    if result.returncode == 0:
        print("依赖安装完成")
        return True
    else:
        print("依赖安装失败")
        return False


def download_models():
    """下载必要的模型"""
    print_header("下载模型")

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    # 模型列表
    models = [
        {
            "name": "sentence-transformers (RAG 嵌入模型)",
            "type": "huggingface",
            "repo": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "auto_download": True  # sentence-transformers 会自动下载
        },
        {
            "name": "Fish Speech V1.5 (TTS)",
            "type": "huggingface",
            "repo": "fishaudio/fish-speech-1.5",
            "local_dir": models_dir / "fish-speech-1.5",
            "auto_download": False  # 需要手动确认
        },
        {
            "name": "EchoMimic V2 (Avatar)",
            "type": "huggingface",
            "repo": "antgroup/echomimic_v2",
            "local_dir": models_dir / "echomimic_v2",
            "auto_download": False
        }
    ]

    print("\n需要下载的模型:")
    for i, model in enumerate(models, 1):
        status = "自动下载" if model.get("auto_download") else "需要手动下载"
        print(f"  {i}. {model['name']} [{status}]")

    print("\n说明:")
    print("- sentence-transformers 模型会在首次使用时自动下载（约 500MB）")
    print("- Fish Speech 和 EchoMimic 模型较大，建议手动下载")

    # 询问是否下载大模型
    print("\n是否尝试下载 Fish Speech 和 EchoMimic 模型？")
    print("（这些模型较大，可能需要较长时间）")

    try:
        choice = input("输入 y 下载，n 跳过 [n]: ").strip().lower()
    except EOFError:
        choice = "n"

    if choice == "y":
        try:
            from huggingface_hub import snapshot_download

            for model in models:
                if not model.get("auto_download") and "local_dir" in model:
                    print(f"\n下载 {model['name']}...")
                    try:
                        snapshot_download(
                            repo_id=model["repo"],
                            local_dir=str(model["local_dir"]),
                            local_dir_use_symlinks=False
                        )
                        print(f"下载完成: {model['name']}")
                    except Exception as e:
                        print(f"下载失败: {e}")
                        print(f"请手动下载:")
                        print(f"  huggingface-cli download {model['repo']} --local-dir {model['local_dir']}")

        except ImportError:
            print("huggingface_hub 未安装，请先安装依赖")
    else:
        print("\n跳过模型下载。如需手动下载，请运行:")
        for model in models:
            if not model.get("auto_download") and "local_dir" in model:
                print(f"  huggingface-cli download {model['repo']} --local-dir {model['local_dir']}")

    return True


def create_directories():
    """创建必要的目录"""
    print_header("创建目录结构")

    dirs = [
        "models",
        "uploads/photos",
        "uploads/voice_samples",
        "uploads/chat_history",
        "outputs",
        "data/chroma_db",
        "static"
    ]

    for dir_path in dirs:
        full_path = PROJECT_ROOT / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"  已创建: {dir_path}/")

    return True


def main():
    """主函数"""
    print_header("EchoSelf 安装和设置")
    print("此脚本将帮助您设置 EchoSelf 数字分身系统")

    results = {}

    # 1. 检查 Python 版本
    results["python"] = check_python_version()

    # 2. 创建目录
    results["dirs"] = create_directories()

    # 3. 安装依赖
    print("\n是否安装/更新 Python 依赖？")
    try:
        choice = input("输入 y 安装，n 跳过 [y]: ").strip().lower()
    except EOFError:
        choice = "y"

    if choice != "n":
        results["deps"] = install_dependencies()
    else:
        results["deps"] = None

    # 4. 检查 CUDA
    results["cuda"] = check_cuda()

    # 5. 检查 FFmpeg
    results["ffmpeg"] = check_ffmpeg()

    # 6. 检查 Ollama
    results["ollama"] = check_ollama_connection()

    # 7. 下载模型
    results["models"] = download_models()

    # 总结
    print_header("设置完成 - 总结")

    for name, status in results.items():
        if status is None:
            status_str = "跳过"
        elif status:
            status_str = "通过"
        else:
            status_str = "需要注意"
        print(f"  {name}: {status_str}")

    print("\n" + "=" * 60)
    print("如果所有检查都通过，可以运行以下命令启动系统:")
    print("  python app.py")
    print("或")
    print("  ./run.sh")
    print("=" * 60)


if __name__ == "__main__":
    main()
