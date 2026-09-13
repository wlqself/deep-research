import re
import logging

from ..log.logging_utils import log_event
from urllib.parse import quote
from uuid import UUID

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from .. import agent as agent_module
from ..publishing.artifacts import (
    ArtifactIntegrityError,
    ArtifactNotFoundError,
    ArtifactService,
)
from ..publishing.models import ArtifactSnapshot
from ..state.access import thread_values

# 基础设施定义
logger = logging.getLogger(
    "deep_research.artifact"
)
router = APIRouter()

ARTIFACT_ID_PATTERN = re.compile(
    r"^[0-9a-f]{32}$"
)

# _not_found 辅助函数
def _not_found() -> HTTPException:
    return HTTPException(
        status_code=404,
        detail="Artifact not found.",
    )


async def _get_thread_artifact(
    thread_id: UUID,
    artifact_id: str,
) -> ArtifactSnapshot:
    try:
        # 临时构造 ArtifactService，调用其 read 方法
        return await ArtifactService(
            agent_module.agent,
        ).read(
            thread_id=str(thread_id),
            artifact_id=artifact_id,
        )
    except ArtifactNotFoundError as error:
        raise _not_found() from error
    except ArtifactIntegrityError as error:
        log_event(
            logger,
            logging.ERROR,
            "artifact.download.failed",
            thread_id=str(thread_id),
            artifact_id=artifact_id,
            status="failed",
            error_code="artifact_integrity_failed",
        )
        raise HTTPException(
            status_code=409,
            detail="Artifact integrity check failed.",
        ) from error

# 注册 GET 路由，路径参数 thread_id（UUID 类型自动校验格式）和 artifact_id（字符串）
@router.get(
    "/artifacts/{thread_id}/{artifact_id}",
    name="download_artifact",
)
async def download_artifact(
    thread_id: UUID,
    artifact_id: str,
) -> Response:
    # 调用辅助函数获取已验证的制品快照
    snapshot = await _get_thread_artifact(
        thread_id,
        artifact_id,
    )
    # 取快照中的 Markdown 内容作为响应体
    content = snapshot.markdown_content
    # 用 urllib.parse.quote 对文件名做 URL 编码
    encoded_filename = quote(
        snapshot.filename,
        safe="",
    )
    # 记录 INFO 级别的审计日志，标记操作完成
    log_event(
        logger,
        logging.INFO,
        "artifact.downloaded",
        thread_id=str(thread_id),
        artifact_id=artifact_id,
        status="completed",
    )
    return Response(
        content=content,
        media_type="text/markdown",
        headers={
            "Content-Disposition": (
                "attachment; "
                f"filename*=UTF-8''{encoded_filename}"
            ),
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get(
    "/threads/{thread_id}/artifacts",
)
# 列出某个线程下的所有制品
async def list_artifacts(
    thread_id: UUID,
) -> dict[str, object]:
    #  从 agent 读取线程的原始数据
    values = await thread_values(
        agent_module.agent,
        str(thread_id),
    )
    # 如果 messages 不存在或为空，认为线程不存在，返回 404
    raw_messages = values.get(
        "messages",
        [],
    )

    if (
        not isinstance(raw_messages, list)
        or not raw_messages
    ):
        raise HTTPException(
            status_code=404,
            detail="Thread not found.",
        )
    # 从线程数据中取 artifacts 字典
    raw_artifacts = values.get(
        "artifacts",
        {},
    )

    if not isinstance(raw_artifacts, dict):
        raw_artifacts = {}

    artifacts: list[dict[str, object]] = []
    # 校验 artifact_id 格式（32 位十六进制）和 metadata 类型
    for artifact_id, metadata in raw_artifacts.items():
        if (
            not isinstance(artifact_id, str)
            or not ARTIFACT_ID_PATTERN.fullmatch(
                artifact_id
            )
            or not isinstance(metadata, dict)
        ):
            continue
        # 从 metadata 中提取各个字段
        filename = metadata.get("filename")
        workspace_path = metadata.get(
            "workspace_path"
        )
        created_at = metadata.get("created_at")
        size_bytes = metadata.get("size_bytes")
        sha256 = metadata.get("sha256")

        if (
            not isinstance(filename, str)
            or not filename
            or not isinstance(workspace_path, str)
            or not workspace_path.startswith("/final/")
            or not isinstance(created_at, str)
            or not isinstance(size_bytes, int)
            or not isinstance(sha256, str)
        ):
            continue
        # 构造公开的制品信息字典，包含 download_url 字段
        artifacts.append(
            {
                "artifact_id": artifact_id,
                "filename": filename,
                "workspace_path": workspace_path,
                "created_at": created_at,
                "size_bytes": size_bytes,
                "sha256": sha256,
                "download_url": (
                    f"/artifacts/{thread_id}/"
                    f"{artifact_id}"
                ),
            }
        )

    artifacts.sort(
        key=lambda item: item["created_at"],
        reverse=True,
    )
    log_event(
        logger,
        logging.INFO,
        "artifact.list.completed",
        thread_id=str(thread_id),
        status="completed",
    )
    return {
        "thread_id": str(thread_id),
        "artifacts": artifacts,
    }

# values 里面的内容
# content = "# Test report"
# artifact_id = "abcdef1234567890abcdef1234567890"
# filename = "Test-report-abcdef123456.md"
# workspace_path = f"/final/{filename}"

# values = {
#     "messages": [
#         {
#             "role": "user",
#             "content": "test",
#         }
#     ],
#     "artifacts": {
#         artifact_id: {
#             "artifact_id": artifact_id,
#             "filename": filename,
#             "workspace_path": workspace_path,
#             "created_at": "2026-08-20T12:00:00+00:00",
#             "size_bytes": len(content.encode("utf-8")),
#             "sha256": hashlib.sha256(
#                 content.encode("utf-8")
#             ).hexdigest(),
#         }
#     },
#     "files": {
#         workspace_path: {
#             "content": content,
#             "encoding": "utf-8",
#         }
#     },
# }
