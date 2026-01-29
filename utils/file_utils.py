"""
文件处理工具模块
File Utilities Module

提供文件操作相关的工具函数
"""

import os
import shutil
import hashlib
import time
from pathlib import Path
from typing import Optional, Union, List, Tuple
from loguru import logger


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    确保目录存在，如果不存在则创建

    Args:
        path: 目录路径

    Returns:
        Path: 目录路径对象
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_hash(file_path: Union[str, Path], algorithm: str = "md5") -> str:
    """
    计算文件哈希值

    Args:
        file_path: 文件路径
        algorithm: 哈希算法 ("md5", "sha256", etc.)

    Returns:
        str: 文件哈希值
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    hasher = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            hasher.update(chunk)

    return hasher.hexdigest()


def clean_old_files(
    directory: Union[str, Path],
    max_age_hours: int = 24,
    extensions: Optional[List[str]] = None
) -> int:
    """
    清理指定目录中的旧文件

    Args:
        directory: 目录路径
        max_age_hours: 最大保留时间（小时）
        extensions: 要清理的文件扩展名列表（None 表示所有文件）

    Returns:
        int: 删除的文件数量
    """
    directory = Path(directory)
    if not directory.exists():
        return 0

    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    deleted_count = 0

    for file_path in directory.iterdir():
        if not file_path.is_file():
            continue

        # 检查扩展名
        if extensions and file_path.suffix.lower() not in extensions:
            continue

        # 检查文件年龄
        file_age = current_time - file_path.stat().st_mtime
        if file_age > max_age_seconds:
            try:
                file_path.unlink()
                deleted_count += 1
                logger.debug(f"已删除旧文件: {file_path.name}")
            except Exception as e:
                logger.warning(f"删除文件失败: {file_path.name}, 错误: {e}")

    if deleted_count > 0:
        logger.info(f"清理完成，共删除 {deleted_count} 个文件")

    return deleted_count


def save_uploaded_file(
    file_data,
    save_dir: Union[str, Path],
    filename: Optional[str] = None,
    overwrite: bool = False
) -> Tuple[Path, str]:
    """
    保存上传的文件

    Args:
        file_data: 文件数据（可以是 Gradio file 对象、bytes 或文件路径）
        save_dir: 保存目录
        filename: 文件名（可选，自动生成）
        overwrite: 是否覆盖同名文件

    Returns:
        Tuple[Path, str]: (保存路径, 文件哈希)
    """
    save_dir = ensure_dir(save_dir)

    # 处理不同类型的文件数据
    if isinstance(file_data, (str, Path)):
        # 文件路径
        src_path = Path(file_data)
        if not src_path.exists():
            raise FileNotFoundError(f"源文件不存在: {src_path}")

        if filename is None:
            filename = src_path.name

        dest_path = save_dir / filename

        # 避免覆盖
        if not overwrite and dest_path.exists():
            base, ext = os.path.splitext(filename)
            counter = 1
            while dest_path.exists():
                filename = f"{base}_{counter}{ext}"
                dest_path = save_dir / filename
                counter += 1

        # 复制文件
        shutil.copy2(src_path, dest_path)

    elif hasattr(file_data, 'name'):
        # Gradio file 对象
        src_path = Path(file_data.name)
        if filename is None:
            filename = src_path.name

        dest_path = save_dir / filename

        if not overwrite and dest_path.exists():
            base, ext = os.path.splitext(filename)
            counter = 1
            while dest_path.exists():
                filename = f"{base}_{counter}{ext}"
                dest_path = save_dir / filename
                counter += 1

        shutil.copy2(src_path, dest_path)

    elif isinstance(file_data, bytes):
        # 字节数据
        if filename is None:
            filename = f"uploaded_{int(time.time())}"

        dest_path = save_dir / filename

        if not overwrite and dest_path.exists():
            base, ext = os.path.splitext(filename)
            counter = 1
            while dest_path.exists():
                filename = f"{base}_{counter}{ext}"
                dest_path = save_dir / filename
                counter += 1

        with open(dest_path, 'wb') as f:
            f.write(file_data)

    else:
        raise ValueError(f"不支持的文件数据类型: {type(file_data)}")

    # 计算哈希
    file_hash = get_file_hash(dest_path)

    logger.info(f"文件已保存: {dest_path.name}")
    return dest_path, file_hash


def validate_file_type(
    file_path: Union[str, Path],
    allowed_extensions: List[str],
    allowed_mimetypes: Optional[List[str]] = None
) -> bool:
    """
    验证文件类型

    Args:
        file_path: 文件路径
        allowed_extensions: 允许的扩展名列表（如 ['.jpg', '.png']）
        allowed_mimetypes: 允许的 MIME 类型列表（可选）

    Returns:
        bool: 是否有效
    """
    file_path = Path(file_path)

    # 检查扩展名
    ext = file_path.suffix.lower()
    if ext not in [e.lower() for e in allowed_extensions]:
        return False

    # 检查 MIME 类型（如果指定）
    if allowed_mimetypes:
        try:
            import magic
            mime = magic.Magic(mime=True)
            file_mime = mime.from_file(str(file_path))
            if file_mime not in allowed_mimetypes:
                return False
        except ImportError:
            # 如果没有安装 python-magic，跳过 MIME 检查
            pass

    return True


def get_file_size(file_path: Union[str, Path], human_readable: bool = True) -> Union[int, str]:
    """
    获取文件大小

    Args:
        file_path: 文件路径
        human_readable: 是否返回人类可读格式

    Returns:
        Union[int, str]: 文件大小
    """
    file_path = Path(file_path)
    size_bytes = file_path.stat().st_size

    if not human_readable:
        return size_bytes

    # 转换为人类可读格式
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024

    return f"{size_bytes:.1f} TB"


def create_temp_file(
    suffix: str = "",
    prefix: str = "echoself_",
    directory: Optional[Union[str, Path]] = None
) -> Path:
    """
    创建临时文件路径（不创建实际文件）

    Args:
        suffix: 文件后缀
        prefix: 文件前缀
        directory: 临时目录（默认使用系统临时目录）

    Returns:
        Path: 临时文件路径
    """
    import tempfile

    if directory:
        directory = ensure_dir(directory)
        fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=str(directory))
    else:
        fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix)

    os.close(fd)
    return Path(path)


def copy_with_progress(
    src: Union[str, Path],
    dst: Union[str, Path],
    callback=None
) -> Path:
    """
    带进度回调的文件复制

    Args:
        src: 源文件路径
        dst: 目标文件路径
        callback: 进度回调函数 callback(copied_bytes, total_bytes)

    Returns:
        Path: 目标文件路径
    """
    src = Path(src)
    dst = Path(dst)

    total_size = src.stat().st_size
    copied = 0

    with open(src, 'rb') as fsrc:
        with open(dst, 'wb') as fdst:
            while True:
                chunk = fsrc.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                fdst.write(chunk)
                copied += len(chunk)
                if callback:
                    callback(copied, total_size)

    return dst


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("File Utils 测试")
    print("=" * 60)

    # 测试 ensure_dir
    print("\n1. 测试 ensure_dir...")
    test_dir = Path(tempfile.gettempdir()) / "echoself_test"
    result = ensure_dir(test_dir)
    print(f"✓ 目录已创建: {result}")

    # 测试文件保存和哈希
    print("\n2. 测试文件保存和哈希...")
    test_content = b"Hello, EchoSelf!"
    saved_path, file_hash = save_uploaded_file(
        test_content,
        test_dir,
        filename="test.txt"
    )
    print(f"✓ 文件已保存: {saved_path}")
    print(f"  哈希值: {file_hash}")

    # 测试文件大小
    print("\n3. 测试文件大小...")
    size = get_file_size(saved_path)
    print(f"✓ 文件大小: {size}")

    # 测试文件类型验证
    print("\n4. 测试文件类型验证...")
    is_valid = validate_file_type(saved_path, ['.txt', '.md'])
    print(f"✓ 验证结果: {is_valid}")

    # 清理
    print("\n5. 清理测试文件...")
    shutil.rmtree(test_dir)
    print("✓ 清理完成")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
