import hashlib
import json
import logging

from ..log.logging_utils import log_event
from datetime import datetime, timezone
from urllib.parse import quote

from deepagents.backends import StateBackend
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command

from ..artifacts.metadata import (
    build_artifact_metadata,
)
from ..context import ResearchContext
from ..citations import (
    append_verified_sources,
    invalid_source_ids,
)
from ..log.log_audit import audit_event

logger = logging.getLogger(
    "deep_research.artifact"
)
@tool
def save_report(
    title: str,
    content: str,
    runtime: ToolRuntime[ResearchContext],
) -> dict[str, object] | Command:
    """Save Markdown explicitly requested by the user or required for publishing.

    When the user asks to research and then publish in one request, saving the
    cited findings as an Artifact is an authorized prerequisite. Return the
    actual ``artifact_id`` for the next publishing tool; never invent one.
    """

    normalized_title = title.strip()

    if not normalized_title:
        return {
            "ok": False,
            "error": "invalid_title",
            "message": "Report title must not be empty.",
        }

    if not content.strip():
        return {
            "ok": False,
            "error": "invalid_content",
            "message": "Report content must not be empty.",
        }

    configurable = runtime.config.get(
        "configurable",
        {},
    )

    thread_id = configurable.get("thread_id")

    if not thread_id:
        return {
            "ok": False,
            "error": "missing_thread_id",
            "message": "A thread_id is required to save a report.",
        }
    
    raw_sources = runtime.state.get(
        "sources",
        {},
    )

    sources = (
        raw_sources
        if isinstance(raw_sources, dict)
        else {}
    )

    invalid_references = invalid_source_ids(
        content,
        sources,
    )

    if invalid_references:
        return {
            "ok": False,
            "error": "invalid_source_reference",
            "invalid_references": invalid_references,
            "message": (
                "The report contains invalid source references: "
                + ", ".join(invalid_references)
                + ". Use only sources from the current thread."
            ),
        }

    content_to_save = append_verified_sources(
        content,
        sources,
    )
    content_bytes = content_to_save.encode("utf-8")

    metadata = build_artifact_metadata(
        normalized_title,
    )
    
    write_result = StateBackend().write(
        metadata["workspace_path"],
        content_to_save,
    )
    # 把文件内容作为 State 的一部分，存进了 LangGraph 的 Checkpointer
    if write_result.error:
        log_event(
            logger,
            logging.ERROR,
            "artifact.save.failed",
            thread_id=str(thread_id),
            status="failed",
            error_code="artifact_write_failed",
        )
        return {
            "ok": False,
            "error": "save_failed",
            "message": write_result.error,
        }

    created_at = datetime.now(
        timezone.utc,
    ).isoformat()

    artifact_metadata = {
        "artifact_id": metadata["artifact_id"],
        "filename": metadata["filename"],
        "workspace_path": metadata["workspace_path"],
        "created_at": created_at,
        "size_bytes": len(content_bytes),
        "sha256": hashlib.sha256(
            content_bytes,
        ).hexdigest(),
    }

    encoded_thread_id = quote(
        str(thread_id),
        safe="",
    )

    payload = {
        "ok": True,
        **artifact_metadata,
        "download_url": (
            f"/artifacts/{encoded_thread_id}/"
            f"{metadata['artifact_id']}"
        ),
    }
    log_event(
        logger,
        logging.INFO,
        "artifact.saved",
        thread_id=str(thread_id),
        artifact_id=metadata["artifact_id"],
        status="completed",
    )
    audit_event(
        "report.save",
        "completed",
        thread_id=str(thread_id),
        artifact_id=metadata["artifact_id"],
    )
    return Command(
        update={
            "artifacts": {
                metadata["artifact_id"]: artifact_metadata,
            },
            "messages": [
                ToolMessage(
                    content=json.dumps(
                        payload,
                        ensure_ascii=False,
                    ),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )
