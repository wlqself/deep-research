import hashlib
import tempfile

from pathlib import Path
from uuid import UUID
from fastapi import UploadFile


ALLOWED_EXTENSIONS = {
    ".pdf",
    ".md",
    ".markdown",
    ".txt",
}

ALLOWED_MIME_TYPES = {
    ".pdf": "application/pdf",
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".markdown": "text/markdown",
}

# 创建文件的安全文件名和路径
def build_document_path(
    documents_root: str,
    document_id: str,
    extension: str,
) -> tuple[str, Path]:
    try:
        normalized_document_id = str(
            UUID(document_id)
        )
    except ValueError as error:
        raise ValueError(
            "document_id must be a valid UUID"
        ) from error

    normalized_extension = extension.lower()

    if normalized_extension not in ALLOWED_EXTENSIONS:
        raise ValueError(
            "unsupported document extension"
        )

    root = Path(documents_root).resolve()
    document_dir = (
        root / normalized_document_id
    ).resolve()

    safe_filename = (
        f"{normalized_document_id}"
        f"{normalized_extension}"
    )

    document_path = (
        document_dir / safe_filename
    ).resolve()

    if not document_dir.is_relative_to(root):
        raise ValueError(
            "document directory escapes documents root"
        )

    if not document_path.is_relative_to(root):
        raise ValueError(
            "document path escapes documents root"
        )
    return safe_filename, document_path

# 识别上传文件类型
def validate_upload_type(
        filename: str,
        content_type: str,
) -> str:
    # 提取文件扩展名
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_MIME_TYPES:
        raise ValueError(
            "unsupposrted file entension"
        )
    
    normalized_content_type = (
        content_type
        .split(";", 1)[0]
        .strip()
        .lower()
    )

    expected_content_type = (
        ALLOWED_MIME_TYPES[extension]
    )

    if normalized_content_type != expected_content_type:
        raise ValueError(
            "file extension and MIME type do not match"
        )

    return extension

# 原子写入
async def write_upload_to_temp(
    upload: UploadFile,
    temp_dir: str,
    max_bytes: int,
) -> tuple[Path, int, str]:
    temporary_root = Path(temp_dir)
    temporary_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path: Path | None = None
    digest = hashlib.sha256()
    total_bytes = 0

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", # 二进制写入模式
            prefix="upload-", # 文件名前缀，方便识别
            suffix=".part", # 后缀标记这是“半成品”，不是最终文件
            dir=temporary_root, # 指定临时目录（你可控，不是系统 /tmp）
            delete=False, # 	退出 with 块时不自动删除文件,但如果中途出错，必须手动清理
        ) as target:
            temporary_path = Path(target.name)

            while True:
                chunk = await upload.read(1024 * 1024) # 每次读 1MB

                if not chunk:
                    break

                total_bytes += len(chunk)

                if total_bytes > max_bytes:
                    raise ValueError(
                        "uploaded file exceeds size limit"
                    )

                digest.update(chunk) # 边写边算哈希
                target.write(chunk)

        return (
            temporary_path,  # 临时文件路径（后面用来解析/移动）
            total_bytes, # 实际文件大小（写入 DocumentRecord.size_bytes）
            digest.hexdigest(),  # SHA256（用于去重查询）
        )

    except Exception: # 自动清理垃圾文件（最关键的防御）
        if temporary_path is not None:
            temporary_path.unlink(
                missing_ok=True
            )
        raise

# 文件路径处理
def finalize_document_file(
    temporary_path: Path,
    document_path: Path,
) -> Path:
    # 拒绝符号链接
    if temporary_path.is_symlink():
        raise ValueError(
            "temporary upload must not be a symlink"
        )
    # 确保临时文件是普通文件且存在
    if not temporary_path.is_file():
        raise FileNotFoundError(
            "temporary upload does not exist"
        )
    # 若没有父目录，创建父目录，防止失败
    document_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    #若路径已经存在，则向上级返回错误
    if document_path.exists():
        raise FileExistsError(
            "document path already exists"
        )
    
    temporary_path.rename(document_path)

    return document_path

# 生成期望存储路径
def resolve_document_path(
    documents_root: str,
    document_id: str,
    safe_filename: str,
) -> Path:
    # 统一格式，后缀转为小写
    extension = Path(
        safe_filename
    ).suffix.lower()
    # 获得期望路径
    expected_filename, expected_path = (
        build_document_path(
            documents_root,
            document_id,
            extension,
        )
    )
    # 查看安全路径是否等于期望路径
    if safe_filename != expected_filename:
        raise ValueError(
            "safe_filename does not match document_id"
        )

    return expected_path