import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import(
    AIMessage,
    HumanMessage,
    ToolMessage,
)

from .review_types import (
    MemoryReviewInput,
    ReviewArtifactMetadata,
    ReviewBatch,
    ReviewMessage,
)
from .type import MemoryEntry
from ..config import settings

_SUMMARY_PREFIX = (
    "Here is a summary of the conversation to date:\n\n"
)

def _is_summary_message(message: object) -> bool:
    return (
        isinstance(message, HumanMessage)
        and message.additional_kwargs.get("lc_source")
        == "summarization"
    )

def _summary_text(message: object) -> str | None:
    if not _is_summary_message(message):
        return None

    content = getattr(message, "content", "")
    if not isinstance(content, str):
        return None

    if content.startswith(_SUMMARY_PREFIX):
        content = content[len(_SUMMARY_PREFIX):]

    return content.strip() or None

def _review_message(message: object) -> ReviewMessage | None:
    if _is_summary_message(message):
        return None

    if isinstance(message, ToolMessage):
        return None

    if isinstance(message, HumanMessage):
        role = "user"
    elif isinstance(message, AIMessage):
        if message.tool_calls:
            return None
        role = "assistant"
    else:
        return None

    content = getattr(message, "content", "")
    if not isinstance(content, str) or not content.strip():
        return None

    return ReviewMessage(
        role=role,
        content=content.strip(),
    )

def _artifact_metadata(
    values: Mapping[str, Any], # 真正交给 extractor 的 artifact metadata
    *,
    max_artifacts: int, # -> 是否还有 artifact 因上限未纳入
) -> tuple[
    list[ReviewArtifactMetadata],
    bool,
]:
    if max_artifacts < 1:
        raise ValueError(
            "max_artifacts must be positive"
        )
    
    raw_artifacts = values.get("artifacts", {}) 

    if not isinstance(raw_artifacts, dict):
        return [], False

    artifacts: list[ReviewArtifactMetadata] = []

    for artifact_id, raw in raw_artifacts.items():
        if not isinstance(artifact_id, str):
            continue

        if not isinstance(raw, dict):
            continue

        try:
            artifacts.append(
                ReviewArtifactMetadata.model_validate(
                    {
                        "artifact_id": raw.get(
                            "artifact_id",
                            artifact_id,
                        ), # raw 里有就用 raw 的,保证 artifact_id 字段永不缺失。
                        "filename": raw["filename"],
                        "workspace_path": raw["workspace_path"],
                        "created_at": raw["created_at"],
                        "size_bytes": raw["size_bytes"],
                        "sha256": raw["sha256"],
                    }
                )
            )
        except Exception: # 任何一条校验失败 跳过这一条，继续处理下一条
            continue

    truncated = len(artifacts) > max_artifacts

    return (
        artifacts[:max_artifacts],
        truncated,
    )
"""
所有合法 artifact metadata
  -> 按 max_artifacts 截断
  -> artifact_metadata 只包含前 N 条
  -> artifacts_truncated=True
  -> ReviewBatch.truncated_fields 增加 artifact_metadata
"""
def build_review_input(
    values: Mapping[str, Any],
    *,
    last_reviewed_message_id: str | None,
    similar_old_memories: Sequence[MemoryEntry],

) -> tuple[MemoryReviewInput, str | None]:
    raw_messages = values.get("messages", [])

    if not isinstance(raw_messages, list):
        raw_messages = [] # 不是列表就当空处理——防止状态被写坏导致后面遍历崩溃。

    current_summary: str | None = None
    # 从最新消息往回找
    for message in reversed(raw_messages):
        current_summary = _summary_text(message)
        if current_summary is not None:
            break # 拿到第一个（也就是最新的）摘要就 break

    current_summary, summary_truncated = (
        _bounded_summary(
            current_summary,
            max_chars=(
                settings.memory_extraction_max_summary_chars
            ),
        )
    )
    
    cursor_exists = (
        last_reviewed_message_id is None
        or any(
            _message_id(message)
            == last_reviewed_message_id
            for message in raw_messages
        )
    )

    collect_messages = last_reviewed_message_id is None
    fallback_after_summary = (
        last_reviewed_message_id is not None
        and not cursor_exists
    )
    summary_seen = False

    candidate_messages: list[
        tuple[str | None, ReviewMessage]
    ] = []

    for message in raw_messages:
        message_id = _message_id(message)

        if (
            cursor_exists
            and last_reviewed_message_id is not None
            and not collect_messages
        ):
            if message_id == last_reviewed_message_id:
                collect_messages = True
                continue

            continue

        if _is_summary_message(message):
            if fallback_after_summary:
                summary_seen = True
            continue

        if fallback_after_summary and not summary_seen:
            continue

        review_message = _review_message(message)

        if review_message is None:
            continue

        candidate_messages.append(
            (
                message_id,
                review_message,
            )
        )
    """
    candidate_messages
    -> _build_review_batch()
    -> 得到消息限制结果

    values["artifacts"]
    -> _artifact_metadata()
    -> 得到 artifact 限制结果

    消息限制状态 + artifact 限制状态
    -> 合并到 review_batch.truncated_fields

    review_input
    -> 使用最终 batch 和最终 artifact_metadata
    """
    review_batch = _build_review_batch(
        candidate_messages,
        max_messages=(
            settings.memory_extraction_max_messages
        ),
        max_chars=(
            settings.memory_extraction_max_chars
        ),
    )

    artifact_metadata, artifacts_truncated = (
        _artifact_metadata(
            values,
            max_artifacts=(
                settings.memory_extraction_max_artifacts
            ),
        )
    )

    truncated_fields = list(
        review_batch.truncated_fields
    )

    if summary_truncated:
        truncated_fields.append(
            "current_summary"
        )
        
    if artifacts_truncated:
        truncated_fields.append(
            "artifact_metadata"
        )


    """
    游标后的原始消息
    -> _review_message()
    -> candidate_messages
        [(message_id, ReviewMessage), ...]
    -> _build_review_batch()
    -> review_batch.included_messages
    """
    review_batch = review_batch.model_copy(
        update={
            "truncated_fields": truncated_fields,
        }
    )

    review_messages = review_batch.included_messages
    latest_message_id = (
        review_batch.last_included_message_id
    )

    latest_user_message = next(
        (
            message.content
            for message in reversed(review_messages)
            if message.role == "user"
        ),
        None,
    )

    latest_main_message = next(
        (
            message.content
            for message in reversed(review_messages)
            if message.role == "assistant"
        ),
        None,
    )

    review_input = MemoryReviewInput(
        current_summary=current_summary,
        unreviewed_messages=review_messages,
        review_batch=review_batch,
        latest_user_message=latest_user_message,
        latest_main_message=latest_main_message,
        artifact_metadata=artifact_metadata,
        similar_old_memories=list(similar_old_memories),
    )

    return review_input, latest_message_id


"""
有真实 message.id
  -> 使用真实 ID

没有 ID，但有文本内容
  -> 使用内容指纹

没有 ID，也没有文本
  -> 返回 None
"""
def _message_id(message: object) -> str | None:
    raw_id = getattr(message, "id", None)

    if isinstance(raw_id, str) and raw_id.strip():
        return raw_id

    content = getattr(message, "content", "")

    if not isinstance(content, str):
        return None

    content = content.strip()

    if not content:
        return None

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()

"""
从 candidates 的第一个开始
  -> oldest-first

消息数量达到 max_messages
  -> 停止
  -> has_more_unreviewed=True

总字符达到 max_chars
  -> 停止当前消息之后的内容
  -> 未纳入消息留给下一批

第一条消息本身超过 max_chars
  -> 截断到 max_chars
  -> 标记 message_content
  -> 明确记录发生了截断
"""
def _build_review_batch(
    candidates: Sequence[tuple[str | None, ReviewMessage]],
    *,
    max_messages: int,
    max_chars: int,
) -> ReviewBatch:
    
    included_messages: list[ReviewMessage] = []
    truncated_fields: list[str] = []
    total_chars = 0
    last_included_message_id: str | None = None
    has_more_unreviewed = False

    for index, (message_id, message) in enumerate(candidates):
        if len(included_messages) >= max_messages:
            truncated_fields.append("message_count")
            has_more_unreviewed = True
            break

        content = message.content

        if len(content) > max_chars:
            if not included_messages:
                truncated_content = (
                    content[:max_chars].rstrip()
                )

                included_messages.append(
                    ReviewMessage(
                        role=message.role,
                        content=truncated_content,
                    )
                )

                total_chars += len(truncated_content)
                last_included_message_id = message_id

                truncated_fields.append(
                    "message_content"
                )

                has_more_unreviewed = (
                    index + 1 < len(candidates)
                )
                break

            truncated_fields.append("message_chars")
            has_more_unreviewed = True
            break

        if total_chars + len(content) > max_chars:
            truncated_fields.append("message_chars")
            has_more_unreviewed = True
            break

        included_messages.append(message)
        total_chars += len(content)

        if message_id is not None:
            last_included_message_id = message_id

        if index + 1 < len(candidates):
            has_more_unreviewed = True

    return ReviewBatch(
        included_messages=included_messages,
        last_included_message_id=last_included_message_id,
        has_more_unreviewed=has_more_unreviewed,
        total_chars=total_chars,
        truncated_fields=truncated_fields,
    )

def _bounded_summary(
    summary: str | None,
    *,
    max_chars: int,
) -> tuple[str | None, bool]:
    if max_chars < 1:
        raise ValueError(
            "max_summary_chars must be positive"
        )

    if summary is None:
        return None, False

    normalized = summary.strip()

    if not normalized:
        return None, False

    if len(normalized) <= max_chars:
        return normalized, False

    return (
        normalized[:max_chars].rstrip(),
        True,
    )