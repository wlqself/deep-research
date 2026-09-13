"""Safe access to research Artifacts stored in LangGraph thread state."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Awaitable, Callable
from typing import Any

from ..state.access import thread_values
from .models import ArtifactSnapshot, PublishingDomainError

#  正则模式定义
ARTIFACT_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

#  制品相关异常类
class ArtifactNotFoundError(PublishingDomainError):
    """Raised when an Artifact is not available in the requested thread."""


class ArtifactIntegrityError(PublishingDomainError):
    """Raised when stored Artifact content does not match its metadata."""


ThreadValuesReader = Callable[
    [Any, str],
    Awaitable[dict[str, object]],
]

# 统一错误消息、减少重复、方便后续修改文案。
def _not_found() -> ArtifactNotFoundError:
    return ArtifactNotFoundError("artifact not found")

#  制品 ID 校验函数
def _validate_artifact_id(artifact_id: str) -> None:
    if not isinstance(artifact_id, str) or not ARTIFACT_ID_PATTERN.fullmatch(
        artifact_id
    ):
        raise _not_found()

# 元数据校验函数
def _validated_metadata(
    *,
    artifact_id: str,
    metadata: object,
) -> tuple[str, str, int, str]:
    if not isinstance(metadata, dict):
        raise _not_found()
    # 从字典中提取四个关键字段
    workspace_path = metadata.get("workspace_path")
    filename = metadata.get("filename")
    size_bytes = metadata.get("size_bytes")
    sha256 = metadata.get("sha256")
    # 对提取的字段做一系列严格的格式校验
    if (
        not isinstance(workspace_path, str)
        or not workspace_path.startswith("/final/")
        or "\\" in workspace_path
        or workspace_path.count("/") != 2
        or not isinstance(filename, str)
        or not filename
        or "/" in filename
        or "\\" in filename
        or workspace_path != f"/final/{filename}"
        or isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes < 0
        or not isinstance(sha256, str)
        or not SHA256_PATTERN.fullmatch(sha256)
    ):
        raise _not_found()

    metadata_artifact_id = metadata.get("artifact_id", artifact_id)
    if metadata_artifact_id != artifact_id:
        raise _not_found()

    return workspace_path, filename, size_bytes, sha256

# 读取并验证制品，同时不接受任意文件路径——所有路径都受限于元数据中的 /final/ 前缀约束，防止路径遍历攻击；
class ArtifactService:
    """Read and validate an Artifact without accepting arbitrary file paths."""

    def __init__(
        self,
        agent: Any,
        *,
        thread_values_reader: ThreadValuesReader = thread_values,
    ) -> None:
        self._agent = agent
        self._thread_values_reader = thread_values_reader

    async def read(
        self,
        *,
        thread_id: str,
        artifact_id: str,
    ) -> ArtifactSnapshot:
        """Return a verified snapshot scoped to exactly one thread."""

        if not isinstance(thread_id, str) or not thread_id.strip():
            raise _not_found()

        _validate_artifact_id(artifact_id)
        # 读取线程数据
        values = await self._thread_values_reader(
            self._agent,
            thread_id,
        )
        artifacts = values.get("artifacts", {})

        if not isinstance(artifacts, dict):
            raise _not_found()
        # 元数据校验
        metadata = artifacts.get(artifact_id)
        workspace_path, filename, size_bytes, sha256 = (
            _validated_metadata(
                artifact_id=artifact_id,
                metadata=metadata,
            )
        )

        files = values.get("files", {})
        if not isinstance(files, dict):
            raise _not_found()

        file_data = files.get(workspace_path)
        if not isinstance(file_data, dict):
            raise _not_found()

        content = file_data.get("content")
        if not isinstance(content, str):
            raise _not_found()

        content_bytes = content.encode("utf-8")
        if len(content_bytes) != size_bytes:
            raise ArtifactIntegrityError("artifact size does not match metadata")

        actual_sha256 = hashlib.sha256(content_bytes).hexdigest()
        if actual_sha256 != sha256:
            raise ArtifactIntegrityError(
                "artifact hash does not match metadata"
            )

        return ArtifactSnapshot(
            source_thread_id=thread_id,
            source_artifact_id=artifact_id,
            workspace_path=workspace_path,
            filename=filename,
            size_bytes=size_bytes,
            sha256=sha256,
            markdown_content=content,
        )


