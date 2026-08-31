import hashlib
import re
from urllib.parse import quote
from uuid import UUID

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from .. import agent as agent_module
from ..state.access import thread_values


router = APIRouter()

ARTIFACT_ID_PATTERN = re.compile(
    r"^[0-9a-f]{32}$"
)


def _not_found() -> HTTPException:
    return HTTPException(
        status_code=404,
        detail="Artifact not found.",
    )


def _validate_artifact_id(
    artifact_id: str,
) -> None:
    if not ARTIFACT_ID_PATTERN.fullmatch(
        artifact_id
    ):
        raise _not_found()


async def _get_thread_artifact(
    thread_id: UUID,
    artifact_id: str,
) -> tuple[dict[str, object], dict[str, object]]:
    _validate_artifact_id(artifact_id)

    values = await thread_values(
        agent_module.agent,
        str(thread_id),
    )

    artifacts = values.get(
        "artifacts",
        {},
    )

    if not isinstance(artifacts, dict):
        raise _not_found()

    metadata = artifacts.get(artifact_id)

    if not isinstance(metadata, dict):
        raise _not_found()

    workspace_path = metadata.get(
        "workspace_path"
    )

    filename = metadata.get("filename")
    size_bytes = metadata.get("size_bytes")
    sha256 = metadata.get("sha256")

    if (
        not isinstance(workspace_path, str)
        or not workspace_path.startswith("/final/")
        or not isinstance(filename, str)
        or not filename
        or not isinstance(size_bytes, int)
        or size_bytes < 0
        or not isinstance(sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", sha256)
    ):
        raise _not_found()

    files = values.get(
        "files",
        {},
    )

    if not isinstance(files, dict):
        raise _not_found()

    file_data = files.get(workspace_path)

    if not isinstance(file_data, dict):
        raise _not_found()

    content = file_data.get("content")

    if not isinstance(content, str):
        raise _not_found()

    return (
        metadata,
        {
            "workspace_path": workspace_path,
            "filename": filename,
            "size_bytes": size_bytes,
            "sha256": sha256,
            "content": content,
        },
    )


@router.get(
    "/artifacts/{thread_id}/{artifact_id}",
    name="download_artifact",
)
async def download_artifact(
    thread_id: UUID,
    artifact_id: str,
) -> Response:
    _, file_info = await _get_thread_artifact(
        thread_id,
        artifact_id,
    )

    content = file_info["content"]
    content_bytes = content.encode("utf-8")

    if len(content_bytes) != file_info["size_bytes"]:
        raise HTTPException(
            status_code=409,
            detail="Artifact integrity check failed.",
        )

    actual_sha256 = hashlib.sha256(
        content_bytes,
    ).hexdigest()

    if actual_sha256 != file_info["sha256"]:
        raise HTTPException(
            status_code=409,
            detail="Artifact integrity check failed.",
        )

    encoded_filename = quote(
        file_info["filename"],
        safe="",
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
async def list_artifacts(
    thread_id: UUID,
) -> dict[str, object]:
    values = await thread_values(
        agent_module.agent,
        str(thread_id),
    )

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

    raw_artifacts = values.get(
        "artifacts",
        {},
    )

    if not isinstance(raw_artifacts, dict):
        raw_artifacts = {}

    artifacts: list[dict[str, object]] = []

    for artifact_id, metadata in raw_artifacts.items():
        if (
            not isinstance(artifact_id, str)
            or not ARTIFACT_ID_PATTERN.fullmatch(
                artifact_id
            )
            or not isinstance(metadata, dict)
        ):
            continue

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