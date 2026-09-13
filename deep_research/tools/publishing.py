"""Agent-facing preparation tool for the publishing workflow."""

from __future__ import annotations

import asyncio
from difflib import SequenceMatcher
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from langgraph.types import interrupt

from ..context import ResearchContext
from ..hitl.models import HITLAction, HITLStatus
from ..hitl.publication_state import resolve_publication_hitl_state
from ..hitl.service import HITLService, HITLThreadBusyError
from ..log.logging_utils import log_event
from ..publishing.artifacts import ArtifactIntegrityError, ArtifactNotFoundError
from ..publishing.intent_resolver import PublishingIntentResolver
from ..publishing.intents import (
    PublishingIntent,
    PublishingIntentError,
)
from ..publishing.models import ArticleStatus, InvalidStateTransitionError
from ..publishing.service import (
    InvalidSlugError,
    PublishingService,
    PublishingServiceError,
)
from ..publishing.attachments import ImageAttachmentService
from ..publishing.image_analysis import ImageAnalysisService
from ..publishing.image_generation import (
    ImageGenerationError,
    SiliconFlowImageGenerationService,
)
from ..publishing.wechat_covers import WeChatCoverService


logger = logging.getLogger("deep_research.publishing.tool")

_PUBLICATION_REPAIR_RULES: dict[str, dict[str, Any]] = {
    "xiaohongshu_title_too_long": {
        "field": "title",
        "max_length": 20,
        "instruction": "Shorten the title to at most 20 characters.",
    },
    "xiaohongshu_content_too_long": {
        "field": "markdown_content",
        "max_length": 1000,
        "instruction": "Shorten the cleaned note content to at most 1000 characters.",
    },
    "douyin_title_too_long": {
        "field": "title",
        "max_length": 30,
        "instruction": "Shorten the title to at most 30 characters.",
    },
    "wechat_title_too_long": {
        "field": "title",
        "max_length": 32,
        "instruction": "Shorten the title to at most 32 characters.",
    },
}


def _publication_repair_metadata(error_code: str) -> dict[str, Any]:
    rule = _PUBLICATION_REPAIR_RULES.get(error_code)
    if rule is None:
        return {"repairable": False}
    return {
        "repairable": True,
        "repair_action": "revise_article_for_publication",
        **rule,
    }


def _image_library_preview(attachment_service: ImageAttachmentService) -> list[dict[str, Any]]:
    """Return safe metadata for the shared image library, never storage paths."""
    return [
        {
            "attachment_id": item.attachment_id,
            "thread_id": item.thread_id,
            "filename": item.filename,
            "content_type": item.content_type,
            "size_bytes": item.size_bytes,
            "created_at": item.created_at.isoformat()
            if hasattr(item.created_at, "isoformat")
            else str(item.created_at),
        }
        for item in attachment_service.repository.list_image_attachments()
    ]


def build_set_wechat_cover_tool(
    attachment_service: ImageAttachmentService,
    cover_service: WeChatCoverService,
):
    """Build an Agent tool that promotes any uploaded image to the cover."""

    @tool
    async def use_uploaded_image_as_wechat_cover(
        attachment_id: str,
        runtime: ToolRuntime[ResearchContext],
    ) -> dict[str, Any]:
        """Use one uploaded image from the shared library as the active WeChat cover."""
        thread_id = _thread_id_from_runtime(runtime)
        try:
            attachment = attachment_service.get(attachment_id.strip())
            content = await asyncio.to_thread(
                lambda: Path(attachment.storage_path).read_bytes()
            )
            asset = await asyncio.to_thread(
                cover_service.upload_cover,
                content,
                filename=attachment.filename,
                content_type=attachment.content_type,
                make_active=True,
            )
            return {
                "ok": True,
                "attachment_id": attachment.attachment_id,
                "cover_asset_id": asset.asset_id,
                "status": "active",
                "message": "The uploaded image is now the active WeChat cover.",
            }
        except ValueError as error:
            code = str(error).strip() or "wechat_cover_attachment_failed"
            return {"ok": False, "error_code": code}
        except Exception:
            log_event(
                logger,
                logging.ERROR,
                "publishing.tool.failed",
                thread_id=thread_id,
                status="failed",
                error_code="wechat_cover_attachment_failed",
            )
            return {
                "ok": False,
                "error_code": "wechat_cover_attachment_failed",
            }

    return use_uploaded_image_as_wechat_cover


def build_analyze_uploaded_image_tool(
    attachment_service: ImageAttachmentService,
    analysis_service: ImageAnalysisService,
):
    """Build the Main-only tool that connects images to the vision model."""

    @tool
    async def analyze_uploaded_image(
        runtime: ToolRuntime[ResearchContext],
        attachment_ids: list[str] | str | None = None,
        mode: str = "cached",
        focus: str = "",
    ) -> dict[str, Any]:
        """Analyze uploaded images with the vision model.

        Use exact IDs from the current request or this conversation's image
        history. ``cached`` reads the analysis bound to the image and only
        analyzes it if no completed result exists; ``fresh`` forces a new
        visual analysis after the user disputes an earlier result.
        """
        if mode not in {"cached", "fresh"}:
            return {
                "ok": False,
                "error_code": "image_analysis_mode_invalid",
                "results": [],
            }
        runtime_context = getattr(runtime, "context", None)
        current_ids = tuple(
            getattr(runtime_context, "selected_attachment_ids", ()) or ()
        )
        conversation_ids = tuple(
            getattr(runtime_context, "conversation_attachment_ids", ()) or ()
        )
        runtime_state = getattr(runtime, "state", None)
        state_image_ids = (
            runtime_state.get("image_attachment_ids", [])
            if isinstance(runtime_state, dict)
            else []
        )
        state_ids = (
            tuple(state_image_ids)
            if isinstance(state_image_ids, (list, tuple))
            else ()
        )
        ordered_allowed_ids = list(dict.fromkeys(
            attachment_id
            for attachment_id in (*current_ids, *conversation_ids, *state_ids)
            if isinstance(attachment_id, str) and attachment_id.strip()
        ))
        allowed_ids = set(ordered_allowed_ids)

        requested_ids = (
            [attachment_ids]
            if isinstance(attachment_ids, str)
            else list(attachment_ids or [])
        )
        # Be tolerant of a model emitting an empty argument: use the current
        # request, or the latest historical image, but never a global library
        # image outside this thread's durable state.
        if not requested_ids:
            requested_ids = (
                list(current_ids)
                if current_ids
                else ordered_allowed_ids[-1:]
            )

        normalized_ids = []
        for attachment_id in requested_ids:
            if not isinstance(attachment_id, str):
                continue
            normalized = attachment_id.strip()
            if normalized and normalized not in normalized_ids:
                normalized_ids.append(normalized)
        if not normalized_ids:
            return {
                "ok": False,
                "error_code": "image_attachment_id_required",
                "results": [],
            }

        # The tool may only consume images explicitly attached to this
        # request or already registered in this conversation. This prevents
        # a model hallucination or an unrelated library ID from activating
        # the vision model.
        if not allowed_ids:
            return {
                "ok": False,
                "error_code": "image_analysis_not_requested",
                "results": [],
            }

        unauthorized_ids = [
            attachment_id
            for attachment_id in normalized_ids
            if attachment_id not in allowed_ids
        ]
        normalized_ids = [
            attachment_id
            for attachment_id in normalized_ids
            if attachment_id in allowed_ids
        ]
        if not normalized_ids:
            return {
                "ok": False,
                "error_code": "image_attachment_not_in_request",
                "results": [
                    {
                        "ok": False,
                        "attachment_id": attachment_id,
                        "error_code": "image_attachment_not_in_request",
                    }
                    for attachment_id in unauthorized_ids
                ],
            }

        results: list[dict[str, Any]] = []
        results.extend(
            {
                "ok": False,
                "attachment_id": attachment_id,
                "error_code": "image_attachment_not_in_request",
            }
            for attachment_id in unauthorized_ids
        )
        for attachment_id in normalized_ids[:30]:
            try:
                attachment = await asyncio.to_thread(
                    attachment_service.get,
                    attachment_id,
                )
                analysis = await analysis_service.analyze_now(
                    attachment_id,
                    force=mode == "fresh",
                    focus=focus,
                )
                if analysis is None:
                    results.append({
                        "ok": False,
                        "attachment_id": attachment_id,
                        "filename": attachment.filename,
                        "error_code": "image_analysis_not_ready",
                    })
                    continue

                structured_result: Any = analysis.structured_result
                if isinstance(structured_result, str):
                    try:
                        structured_result = json.loads(structured_result)
                    except json.JSONDecodeError:
                        pass
                results.append({
                    "ok": analysis.status == "completed",
                    "attachment_id": attachment.attachment_id,
                    "filename": attachment.filename,
                    "analysis_status": analysis.status,
                    "image_type": analysis.image_type,
                    "confidence": analysis.confidence,
                    "summary": analysis.summary,
                    "ocr_text": analysis.ocr_text,
                    "key_points": (
                        structured_result.get("key_points", [])
                        if isinstance(structured_result, dict)
                        else []
                    ),
                    "structured_result": structured_result,
                    "error_code": analysis.error_code,
                    "analysis_source": mode,
                })
            except Exception:
                results.append({
                    "ok": False,
                    "attachment_id": attachment_id,
                    "error_code": "image_analysis_failed",
                })

        return {
            "ok": bool(results) and all(item.get("ok") for item in results),
            "results": results,
        }

    return analyze_uploaded_image


def build_read_image_analysis_tool(
    attachment_service: ImageAttachmentService,
):
    """Build a Main-only tool for reading derived image understanding."""

    @tool
    async def read_uploaded_image_analysis(
        attachment_id: str,
        runtime: ToolRuntime[ResearchContext],
    ) -> dict[str, Any]:
        """Read safe OCR and visual analysis for one shared image attachment."""
        normalized = attachment_id.strip()
        if not normalized:
            return {
                "ok": False,
                "error_code": "image_attachment_id_required",
            }
        runtime_context = getattr(runtime, "context", None)
        allowed_ids = set(
            getattr(runtime_context, "selected_attachment_ids", ()) or ()
        )
        allowed_ids.update(
            getattr(runtime_context, "conversation_attachment_ids", ()) or ()
        )
        runtime_state = getattr(runtime, "state", None)
        state_image_ids = (
            runtime_state.get("image_attachment_ids", [])
            if isinstance(runtime_state, dict)
            else []
        )
        if isinstance(state_image_ids, (list, tuple)):
            allowed_ids.update(
                attachment_id
                for attachment_id in state_image_ids
                if isinstance(attachment_id, str) and attachment_id.strip()
            )
        if not allowed_ids:
            return {
                "ok": False,
                "error_code": "image_analysis_not_requested",
            }
        if normalized not in allowed_ids:
            return {
                "ok": False,
                "error_code": "image_attachment_not_in_request",
            }
        try:
            attachment = await asyncio.to_thread(
                attachment_service.get,
                normalized,
            )
            analysis = await asyncio.to_thread(
                attachment_service.repository.get_image_attachment_analysis,
                normalized,
            )
            if analysis is None:
                return {
                    "ok": False,
                    "error_code": "image_analysis_not_ready",
                }
            return {
                "ok": True,
                "attachment_id": attachment.attachment_id,
                "filename": attachment.filename,
                "analysis_status": analysis.status,
                "image_type": analysis.image_type,
                "confidence": analysis.confidence,
                "summary": analysis.summary,
                "ocr_text": analysis.ocr_text,
                "structured_result": analysis.structured_result,
                "error_code": analysis.error_code,
            }
        except Exception:
            return {
                "ok": False,
                "error_code": "image_analysis_read_failed",
            }

    return read_uploaded_image_analysis


def build_generate_image_tool(
    generation_service: SiliconFlowImageGenerationService,
):
    """Build the Main-only tool for generating library-backed images."""

    @tool
    async def generate_image(
        prompt: str,
        runtime: ToolRuntime[ResearchContext],
        negative_prompt: str = "",
        image_size: str = "1024x1024",
        count: int = 1,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Generate an image and return shared-library attachment IDs.

        The prompt should be a concise visual description, not a copied
        research report. The returned IDs can be passed directly to
        request_publication_approval for the current publication.
        """
        thread_id = _thread_id_from_runtime(runtime)
        if thread_id is None:
            return {
                "ok": False,
                "error_code": "missing_thread_id",
                "attachment_ids": [],
            }
        try:
            images = await generation_service.generate(
                thread_id=thread_id,
                prompt=prompt,
                negative_prompt=negative_prompt,
                image_size=image_size,
                count=count,
                seed=seed,
            )
        except ImageGenerationError as error:
            log_event(
                logger,
                logging.WARNING,
                "publishing.tool.image_generation_failed",
                thread_id=thread_id,
                status="failed",
                error_code=error.error_code,
            )
            return {
                "ok": False,
                "error_code": error.error_code,
                "attachment_ids": [],
                "message": str(error),
            }
        except Exception:
            log_event(
                logger,
                logging.ERROR,
                "publishing.tool.image_generation_failed",
                thread_id=thread_id,
                status="failed",
                error_code="image_generation_failed",
            )
            return {
                "ok": False,
                "error_code": "image_generation_failed",
                "attachment_ids": [],
            }

        attachment_ids = tuple(
            item["attachment_id"]
            for item in images
            if isinstance(item.get("attachment_id"), str)
        )
        context = getattr(runtime, "context", None)
        if context is not None:
            try:
                context.generated_attachment_ids = attachment_ids
            except (AttributeError, TypeError):
                # Some isolated tests pass a read-only context object.
                pass

        return {
            "ok": True,
            "mode": "text_to_image",
            "model": generation_service.model_name,
            "attachment_ids": list(attachment_ids),
            "images": images,
            "message": (
                "Images were generated and saved to the shared image library. "
                "Use these attachment_ids for the current publication."
            ),
        }

    return generate_image

# 应用层在启动时注入服务实例，返回一个配置好的工具函数给 Agent 框架
def build_prepare_article_tool(
    publishing_service: PublishingService,
):
    """Build the only publishing tool exposed to the Agent.

    The service is injected by the application layer. The tool can prepare a
    draft, but it deliberately has no approval or publish operation.
    """

    non_retryable_failures: dict[str, dict[str, Any]] = {}

    def call_fingerprint(
        *,
        thread_id: str,
        artifact_id: str,
        title: str | None,
        slug: str | None,
        excerpt: str | None,
        tags: list[str] | None,
        markdown_content: str | None,
    ) -> str:
        payload = json.dumps(
            {
                "thread_id": thread_id,
                "artifact_id": artifact_id,
                "title": title,
                "slug": slug,
                "excerpt": excerpt,
                "tags": tags,
                "markdown_content": markdown_content,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @tool
    async def prepare_article_for_publication(
        artifact_id: str,
        runtime: ToolRuntime[ResearchContext], # Agent 运行时上下文，包含配置信息
        title: str | None = None,
        slug: str | None = None,
        excerpt: str | None = None,
        tags: list[str] | None = None,
        markdown_content: str | None = None,
    ) -> dict[str, Any]:
        """Prepare an Article draft from the current thread's Artifact.

        This tool may create or edit a draft only. It must never approve or
        publish an Article; those actions require an explicit user command.

        ``slug`` must be at most 120 characters and contain only letters,
        numbers, and ``-``. It cannot start or end with ``-``. Do not use
        spaces, dots, underscores, slashes, or URL syntax. Convert versions
        such as ``1.0`` to ``1-0`` (for example,
        ``langchain-1-0-updates``). If the tool returns ``retryable=false``,
        never repeat the same call with identical arguments.
        """
        # 从运行时获取并校验 thread_id
        configurable = runtime.config.get("configurable", {})
        thread_id = configurable.get("thread_id")
        if not isinstance(thread_id, str) or not thread_id.strip():
            return {
                "ok": False,
                "error_code": "missing_thread_id",
                "message": "A thread_id is required to prepare an article.",
            }
        fingerprint = call_fingerprint(
            thread_id=thread_id,
            artifact_id=artifact_id,
            title=title,
            slug=slug,
            excerpt=excerpt,
            tags=tags,
            markdown_content=markdown_content,
        )
        previous_failure = non_retryable_failures.get(fingerprint)
        if previous_failure is not None:
            return {
                **previous_failure,
                "duplicate": True,
                "message": (
                    "This identical non-retryable request already failed. "
                    "Change the invalid argument or ask the user; do not "
                    "repeat this call again."
                ),
            }

        #  创建文章（调用服务）
        try:
            article = await publishing_service.create_article_from_artifact(
                thread_id=thread_id,
                artifact_id=artifact_id,
                title=title,
                slug=slug,
                excerpt=excerpt,
                tags=tags,
            )
            # edit_article 是同步方法（操作 SQLite），但当前工具函数是 async 的
            edit_fields = {
                "title": title,
                "slug": slug,
                "excerpt": excerpt,
                "tags": tags,
                "markdown_content": markdown_content,
            }
            edit_fields = {
                key: value
                for key, value in edit_fields.items()
                if value is not None
            }
            if edit_fields:
                article = await asyncio.to_thread(
                    publishing_service.edit_article,
                    article.article_id,
                    **edit_fields,
                )
            # 异常处理——分类型捕获
        except ArtifactNotFoundError:
            log_event(
                logger,
                logging.WARNING,
                "publishing.tool.failed",
                thread_id=thread_id,
                artifact_id=artifact_id,
                status="failed",
                error_code="artifact_not_found",
            )
            return {
                "ok": False,
                "error_code": "artifact_not_found",
                "message": "The requested Artifact was not found.",
            }
        except ArtifactIntegrityError:
            log_event(
                logger,
                logging.WARNING,
                "publishing.tool.failed",
                thread_id=thread_id,
                artifact_id=artifact_id,
                status="failed",
                error_code="artifact_integrity_failed",
            )
            return {
                "ok": False,
                "error_code": "artifact_integrity_failed",
                "message": "The Artifact integrity check failed.",
            }
        except InvalidSlugError:
            log_event(
                logger,
                logging.WARNING,
                "publishing.tool.failed",
                thread_id=thread_id,
                artifact_id=artifact_id,
                status="failed",
                error_code="invalid_slug",
            )
            result = {
                "ok": False,
                "error_code": "invalid_slug",
                "retryable": False,
                "message": (
                    "The slug is invalid. Use at most 120 characters with "
                    "letters, numbers, and hyphens only; replace version "
                    "dots such as 1.0 with 1-0."
                ),
            }
            non_retryable_failures[fingerprint] = result
            return result
        except PublishingServiceError:
            log_event(
                logger,
                logging.WARNING,
                "publishing.tool.failed",
                thread_id=thread_id,
                artifact_id=artifact_id,
                status="failed",
                error_code="article_preparation_failed",
            )
            result = {
                "ok": False,
                "error_code": "article_preparation_failed",
                "retryable": False,
                "message": "The article could not be prepared.",
            }
            non_retryable_failures[fingerprint] = result
            return result
        except Exception:
            log_event(
                logger,
                logging.ERROR,
                "publishing.tool.failed",
                thread_id=thread_id,
                artifact_id=artifact_id,
                status="failed",
                error_code="article_preparation_failed",
            )
            return {
                "ok": False,
                "error_code": "article_preparation_failed",
                "message": "The article could not be prepared.",
            }

        return {
            "ok": True,
            "article_id": article.article_id,
            "article_version": article.version,
            "status": article.status.value,
            "channel": "local_static_site",
            "approval_required": article.status is ArticleStatus.DRAFT,
            "message": (
                "Article draft is ready for user approval. "
                "Do not approve or publish automatically."
            ),
        }

    return prepare_article_for_publication


def _thread_id_from_runtime(runtime: ToolRuntime[ResearchContext]) -> str | None:
    configurable = runtime.config.get("configurable", {})
    thread_id = configurable.get("thread_id")
    if not isinstance(thread_id, str) or not thread_id.strip():
        return None
    return thread_id


def _runtime_generated_attachment_ids(
    runtime: ToolRuntime[ResearchContext],
) -> tuple[str, ...]:
    """Return exact image IDs produced by the server, never model text."""

    ids: list[str] = []

    def add(values: Any) -> None:
        if not isinstance(values, (list, tuple)):
            return
        for value in values:
            if isinstance(value, str) and value.strip() and value.strip() not in ids:
                ids.append(value.strip())

    context = getattr(runtime, "context", None)
    add(getattr(context, "generated_attachment_ids", ()))

    state = getattr(runtime, "state", None)
    messages = state.get("messages", []) if isinstance(state, dict) else []
    for message in reversed(messages if isinstance(messages, list) else []):
        name = (
            message.get("name")
            if isinstance(message, dict)
            else getattr(message, "name", None)
        )
        if name != "generate_image":
            continue
        content = (
            message.get("content")
            if isinstance(message, dict)
            else getattr(message, "content", None)
        )
        if isinstance(content, dict):
            add(content.get("attachment_ids"))
        elif isinstance(content, str):
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                add(payload.get("attachment_ids"))

    return tuple(ids)


def _repair_generated_attachment_ids(
    attachment_ids: tuple[str, ...],
    generated_ids: tuple[str, ...],
    image_attachment_service: ImageAttachmentService,
) -> tuple[str, ...]:
    """Repair only a near-typo of an exact server-generated ID."""

    available: list[str] = []
    for candidate in generated_ids:
        try:
            image_attachment_service.get(candidate)
        except Exception:
            continue
        if candidate not in available:
            available.append(candidate)

    resolved: list[str] = []
    for requested in attachment_ids:
        try:
            image_attachment_service.get(requested)
            resolved.append(requested)
            continue
        except Exception:
            pass

        scored = sorted(
            (
                SequenceMatcher(None, requested, candidate).ratio(),
                candidate,
            )
            for candidate in available
            if len(candidate) == len(requested)
        )
        if not scored:
            resolved.append(requested)
            continue
        best_score, best_candidate = scored[-1]
        tied = [candidate for score, candidate in scored if score == best_score]
        if best_score >= 0.94 and len(tied) == 1:
            resolved.append(best_candidate)
        else:
            resolved.append(requested)
    return tuple(dict.fromkeys(resolved))


def _article_revision_payload(article) -> dict[str, Any]:
    return {
        "article_id": article.article_id,
        "version": article.version,
        "title": article.title,
        "slug": article.slug,
        "markdown_content": article.markdown_content,
        "excerpt": article.excerpt,
        "tags": list(article.tags),
        "status": article.status.value,
        "updated_at": article.updated_at.isoformat(),
    }


def build_read_article_for_revision_tool(
    publishing_service: PublishingService,
):
    """Build a Main-only read tool for LLM-assisted article revision."""

    @tool
    async def read_article_for_revision(
        article_id: str,
        runtime: ToolRuntime[ResearchContext],
    ) -> dict[str, Any]:
        """Read the current Article snapshot before proposing a revision.

        The tool is scoped to the current thread and never exposes source file
        paths or persistence details.
        """

        thread_id = _thread_id_from_runtime(runtime)
        if thread_id is None:
            return {
                "ok": False,
                "error_code": "missing_thread_id",
                "message": "A thread_id is required to read an article.",
            }

        try:
            article = await asyncio.to_thread(
                publishing_service.get_article,
                article_id,
            )
            if article.source_thread_id != thread_id:
                return {
                    "ok": False,
                    "error_code": "article_not_found",
                    "message": "The article was not found in the current thread.",
                }
        except Exception:
            log_event(
                logger,
                logging.WARNING,
                "publishing.tool.article_read_failed",
                thread_id=thread_id,
                status="failed",
                error_code="article_not_found",
            )
            return {
                "ok": False,
                "error_code": "article_not_found",
                "message": "The article could not be read.",
            }

        return {
            "ok": True,
            "article": _article_revision_payload(article),
            "message": (
                "Use this snapshot to prepare a complete revised draft. "
                "Saving it never approves or publishes the article."
            ),
        }

    return read_article_for_revision


def build_revise_article_tool(
    publishing_service: PublishingService,
):
    """Build a Main-only deterministic Article revision tool."""

    @tool
    async def revise_article_for_publication(
        article_id: str,
        markdown_content: str,
        runtime: ToolRuntime[ResearchContext],
        title: str | None = None,
        slug: str | None = None,
        excerpt: str | None = None,
        tags: list[str] | None = None,
        change_request: str | None = None,
        edit_mode: str = "open_ended",
    ) -> dict[str, Any]:
        """Save an LLM-prepared Article revision without approving or publishing.

        ``markdown_content`` must be the complete revised Markdown snapshot,
        not a partial patch.  The caller should first read the current Article,
        apply either a precise or open-ended user request, and then submit the
        complete result here.
        """

        thread_id = _thread_id_from_runtime(runtime)
        if thread_id is None:
            return {
                "ok": False,
                "error_code": "missing_thread_id",
                "message": "A thread_id is required to revise an article.",
            }
        if edit_mode not in {"precise", "open_ended"}:
            return {
                "ok": False,
                "error_code": "invalid_edit_mode",
                "message": "The edit mode is invalid.",
            }
        if not isinstance(markdown_content, str):
            return {
                "ok": False,
                "error_code": "invalid_markdown_content",
                "message": "The revised Markdown content is invalid.",
            }

        try:
            current = await asyncio.to_thread(
                publishing_service.get_article,
                article_id,
            )
            if current.source_thread_id != thread_id:
                return {
                    "ok": False,
                    "error_code": "article_not_found",
                    "message": "The article was not found in the current thread.",
                }
            revised = await asyncio.to_thread(
                publishing_service.revise_article,
                article_id,
                title=title,
                slug=slug,
                markdown_content=markdown_content,
                excerpt=excerpt,
                tags=tags,
                change_reason=change_request,
            )
        except InvalidStateTransitionError as error:
            error_code = (
                "article_revision_in_flight"
                if "in flight" in str(error)
                else "article_not_revisable"
            )
            log_event(
                logger,
                logging.WARNING,
                "publishing.tool.article_revision_failed",
                thread_id=thread_id,
                status="failed",
                error_code=error_code,
            )
            return {
                "ok": False,
                "error_code": error_code,
                "retryable": False,
                "repairable": False,
                "message": str(error),
            }
        except Exception:
            log_event(
                logger,
                logging.WARNING,
                "publishing.tool.article_revision_failed",
                thread_id=thread_id,
                status="failed",
                error_code="article_revision_failed",
            )
            return {
                "ok": False,
                "error_code": "article_revision_failed",
                "retryable": False,
                "repairable": False,
                "message": "The article revision could not be saved.",
            }

        log_event(
            logger,
            logging.INFO,
            "publishing.tool.article_revised",
            thread_id=thread_id,
            status="completed",
        )
        return {
            "ok": True,
            "article": _article_revision_payload(revised),
            "approval_required": True,
            "published": False,
            "message": (
                "The article revision was saved as a draft. "
                "It must be approved again before publishing."
            ),
        }

    return revise_article_for_publication

# 这个工具是只读的，用于让 Agent 查询当前线程的发布审批状态
def build_publication_approval_status_tool(
    publishing_service: PublishingService,
    hitl_service: HITLService | None = None,
):
    """Build a read-only tool for recovering approvals in one thread."""

    @tool
    # 获取同意审批的状态
    async def get_publication_approval_status(
        runtime: ToolRuntime[ResearchContext],
        include_resolved: bool = False,
    ) -> dict[str, Any]:
        """List durable publication approvals for the current thread.

        This tool is read-only. It may surface a pending or already approved
        request, but it must never approve, reject, resume, or publish it.
        """
        # 提取并校验 thread_id
        configurable = runtime.config.get("configurable", {})
        thread_id = configurable.get("thread_id")
        if not isinstance(thread_id, str) or not thread_id.strip():
            return {
                "ok": False,
                "error_code": "missing_thread_id",
                "message": "A thread_id is required to inspect approvals.",
            }

        try:
            approvals = await asyncio.to_thread(
                publishing_service.list_publication_approvals_for_thread,
                thread_id,
                include_resolved=include_resolved,
            )
            interactions = (
                await asyncio.to_thread(
                    hitl_service.list_recoverable_for_thread,
                    thread_id,
                )
                if hitl_service is not None
                else []
            )
            interactions_by_target = {
                (interaction.target_id, interaction.target_version): interaction
                for interaction in interactions
                if interaction.target_type == "publication_approval"
            }
            # 遍历审批，组装返回数据
            items: list[dict[str, Any]] = []
            for approval in approvals:
                article = await asyncio.to_thread(
                    publishing_service.get_article,
                    approval.article_id,
                )
                publication_state = resolve_publication_hitl_state(
                    approval,
                    hitl_service,
                )
                publication = next(
                    (
                        candidate
                        for candidate in publishing_service.list_publications(
                            approval.article_id,
                        )
                        if candidate.article_version == approval.article_version
                        and candidate.channel is approval.channel
                    ),
                    None,
                )
                interaction = interactions_by_target.get(
                    (approval.approval_id, approval.article_version),
                )
                items.append(
                    {
                        "approval_id": approval.approval_id,
                        "article_id": approval.article_id,
                        "article_title": article.title,
                        "article_slug": article.slug,
                        "article_version": approval.article_version,
                        "channel": approval.channel.value,
                        "status": publication_state.approval_status,
                        "workflow_status": publication_state.workflow_status,
                        "created_at": approval.created_at.isoformat(),
                        "updated_at": approval.updated_at.isoformat(),
                        "decided_at": (
                            approval.decided_at.isoformat()
                            if approval.decided_at is not None
                            else None
                        ),
                        "decision_actor": approval.decision_actor,
                        "decision_reason": approval.decision_reason,
                        "interaction_id": publication_state.interaction_id,
                        "interaction_action": publication_state.action,
                        "interaction_status": publication_state.interaction_status,
                        "state_source": publication_state.state_source,
                        "requires_user_confirmation": (
                            publication_state.requires_user_confirmation
                        ),
                        "publication_id": (
                            publication.publication_id
                            if publication is not None
                            else None
                        ),
                        "publication_status": (
                            publication.status.value
                            if publication is not None
                            else None
                        ),
                        "publication_error_code": (
                            publication.error_code
                            if publication is not None
                            else None
                        ),
                    }
                )
        except Exception:
            log_event(
                logger,
                logging.ERROR,
                "publishing.tool.approval_status_failed",
                thread_id=thread_id,
                status="failed",
                error_code="approval_status_failed",
            )
            return {
                "ok": False,
                "error_code": "approval_status_failed",
                "message": "Publication approval status is unavailable.",
            }

        log_event(
            logger,
            logging.INFO,
            "publishing.tool.approval_status_read",
            thread_id=thread_id,
            status="completed",
        )
        recovery_cards = []
        for item in items:
            if not item["interaction_id"] or item["interaction_action"] not in {
                "approve",
                "reject",
                "resume",
            }:
                continue

            publication_status = item["publication_status"]
            can_continue_publish = (
                item["status"] == "approved"
                and item["interaction_status"] in {"approved", "failed"}
                and publication_status in {
                    None,
                    "failed",
                    "delivery_unknown",
                }
            )
            recovery_cards.append(
                {
                    "action": (
                        "resume"
                        if can_continue_publish
                        else item["interaction_action"]
                    ),
                    "resolution_status": "resolved",
                    "target": {
                        "approval_id": item["approval_id"],
                        "article_id": item["article_id"],
                        "article_title": item["article_title"],
                        "article_slug": item["article_slug"],
                        "article_version": item["article_version"],
                        "channel": item["channel"],
                        "approval_status": item["status"],
                        "workflow_status": item["workflow_status"],
                        "publication_status": publication_status,
                        "publication_id": item["publication_id"],
                        "publication_error_code": item[
                            "publication_error_code"
                        ],
                    },
                    "candidates": [],
                    "requires_user_confirmation": (
                        item["requires_user_confirmation"]
                        or can_continue_publish
                    ),
                    "interaction_id": item["interaction_id"],
                    "interaction_status": item["interaction_status"],
                }
            )

        return {
            "ok": True,
            "approvals": items,
            "count": len(items),
            "recovery_cards": recovery_cards,
            "message": (
                "No unresolved publication approvals were found."
                if not items
                else "Publication approval status loaded."
            ),
        }

    return get_publication_approval_status


def build_resolve_publication_intent_tool(
    publishing_service: PublishingService,
    hitl_service: HITLService | None = None,
    enable_native_interrupt: bool = False,
):
    """Build a read-only tool that resolves a language intent to candidates."""
    # 在工厂内部创建 PublishingIntentResolver 实例解析器
    resolver = PublishingIntentResolver(publishing_service, hitl_service)

    @tool
    async def resolve_publication_intent(
        action: str,
        runtime: ToolRuntime[ResearchContext],
        target_hint: str | None = None, # 可选，用户描述的目标
        channel: str | None = None,
        decision_reason: str | None = None,
    ) -> dict[str, Any]:
        """Resolve a publishing intent without executing it.

        ``action`` may be status, approve, reject, resume, or none.  ``channel``
        is an optional platform extracted from the user's language. The result
        only identifies a candidate or reports ambiguity. A later deterministic
        application command is required for any state change.
        """
        # 提取并校验 thread_id
        configurable = runtime.config.get("configurable", {})
        thread_id = configurable.get("thread_id")
        if not isinstance(thread_id, str) or not thread_id.strip():
            return {
                "ok": False,
                "error_code": "missing_thread_id",
                "message": "A thread_id is required to resolve intent.",
            }
        # 用户语言 → 结构化意图 → 安全目标映射
        try:
            intent = PublishingIntent(
                action=action,
                target_hint=target_hint,
                channel=channel,
                decision_reason=decision_reason,
            )
            resolution = resolver.resolve(
                intent,
                thread_id=thread_id,
            )
        except PublishingIntentError:
            log_event(
                logger,
                logging.WARNING,
                "publishing.tool.intent_rejected",
                thread_id=thread_id,
                status="rejected",
                error_code="invalid_publishing_intent",
            )
            return {
                "ok": False,
                "error_code": "invalid_publishing_intent",
                "retryable": False,
                "message": "The publishing intent is invalid.",
            }
        except Exception:
            log_event(
                logger,
                logging.ERROR,
                "publishing.tool.intent_resolution_failed",
                thread_id=thread_id,
                status="failed",
                error_code="intent_resolution_failed",
            )
            return {
                "ok": False,
                "error_code": "intent_resolution_failed",
                "message": "The publishing intent could not be resolved.",
            }
        # DTO 转换
        def target_payload(target):
            if target is None:
                return None
            return {
                "approval_id": target.approval_id,
                "article_id": target.article_id,
                "article_title": target.article_title,
                "article_slug": target.article_slug,
                "article_version": target.article_version,
                "channel": target.channel.value,
                "approval_status": (
                    target.approval_status.value
                    if target.approval_status is not None
                    else None
                ),
                "publication_status": (
                    target.publication_status.value
                    if target.publication_status is not None
                    else None
                ),
                "attachment_ids": list(
                    getattr(target, "attachment_ids", ()) or ()
                ),
                "publication_id": target.publication_id,
                "publication_error_code": target.publication_error_code,
            }

        result = {
            "ok": True,
            "action": intent.action.value,
            "channel": (
                intent.channel.value if intent.channel is not None else None
            ),
            "resolution_status": resolution.status.value,
            "target": target_payload(resolution.target),
            "candidates": [
                target_payload(candidate)
                for candidate in resolution.candidates
            ],
            "requires_user_confirmation": resolution.is_resolved,
            "attachment_ids": list(
                getattr(
                    getattr(runtime, "context", None),
                    "publication_attachment_ids",
                    (),
                )
                or getattr(
                    getattr(runtime, "context", None),
                    "selected_attachment_ids",
                    (),
                )
            ),
            "message": (
                "A unique publishing target was resolved; show a confirmation card."
                if resolution.is_resolved
                else "The publishing intent was not uniquely resolved."
            ),
        }
        if not resolution.is_resolved:
            result["retryable"] = False
            result["error_code"] = (
                "publication_target_not_found"
                if resolution.status.value == "not_found"
                else "publication_target_ambiguous"
            )

        if (
            hitl_service is not None
            and resolution.is_resolved
            and intent.action.value in {"approve", "reject"}
        ):
            target = resolution.target
            if target is None:
                return {
                    "ok": False,
                    "error_code": "hitl_interaction_unavailable",
                    "message": "The approval interaction could not be created.",
                }

            tool_call_id = getattr(runtime, "tool_call_id", None)
            if (
                enable_native_interrupt
                or not isinstance(tool_call_id, str)
                or not tool_call_id.strip()
            ):
                tool_call_id = (
                    f"publishing-intent:{thread_id}:"
                    f"{intent.action.value}:{target.approval_id}:"
                    f"{target.article_version}"
                )

            try:
                interaction = await asyncio.to_thread(
                    hitl_service.create_or_get_interaction,
                    thread_id=thread_id,
                    run_id=tool_call_id,
                    action=HITLAction(intent.action.value),
                    target_type="publication_approval",
                    target_id=target.approval_id,
                    target_version=target.article_version,
                )
            except HITLThreadBusyError as error:
                blocking = error.interaction
                log_event(
                    logger,
                    logging.INFO,
                    "publishing.tool.hitl_blocked",
                    thread_id=thread_id,
                    status="waiting",
                    error_code="hitl_pending",
                )
                return {
                    "ok": False,
                    "error_code": "hitl_pending",
                    "retryable": False,
                    "requires_user_confirmation": False,
                    "blocking_interaction_id": blocking.interaction_id,
                    "message": (
                        "An earlier HITL interaction is still unresolved. "
                        "Complete it before creating another HITL."
                    ),
                }
            except Exception:
                log_event(
                    logger,
                    logging.ERROR,
                    "publishing.tool.hitl_interaction_failed",
                    thread_id=thread_id,
                    status="failed",
                    error_code="hitl_interaction_unavailable",
                )
                return {
                    "ok": False,
                    "error_code": "hitl_interaction_unavailable",
                    "message": "The approval interaction could not be created.",
                }

            result["interaction_id"] = interaction.interaction_id
            result["interaction_status"] = interaction.status.value

            if (
                enable_native_interrupt
                and interaction.status is HITLStatus.PENDING
            ):
                resume_value = interrupt(
                    {
                        "kind": "publication_approval",
                        **result,
                    }
                )
                if isinstance(resume_value, dict):
                    result["resumed"] = True
                    decision = resume_value.get("decision")
                    result["resume_decision"] = (
                        decision if isinstance(decision, str) else None
                    )
                    if decision in {
                        HITLStatus.APPROVED.value,
                        HITLStatus.REJECTED.value,
                    }:
                        result["requires_user_confirmation"] = False
                        result["target"] = {
                            **result["target"],
                            "approval_status": decision,
                        }
                        result["message"] = (
                            "HITL approval was recorded. Do not resolve the "
                            "same approval again; wait for an explicit publish action."
                            if decision == HITLStatus.APPROVED.value
                            else "HITL rejection was recorded. Do not resolve the "
                            "same approval again; wait for a revision or a new request."
                        )

        log_event(
            logger,
            logging.INFO,
            "publishing.tool.intent_resolved",
            thread_id=thread_id,
            status=resolution.status.value,
        )
        return result

    return resolve_publication_intent


def build_request_publication_approval_tool(
    publishing_service: PublishingService,
    hitl_service: HITLService | None = None,
    enable_native_interrupt: bool = False,
    image_attachment_service: ImageAttachmentService | None = None,
):
    """Create a durable publication approval request without publishing."""

    resolver = PublishingIntentResolver(publishing_service, hitl_service)

    @tool
    async def request_publication_approval(
        runtime: ToolRuntime[ResearchContext],
        target_hint: str | None = None,
        channel: str | None = None,
        attachment_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Request human approval for one exact Article version and channel.

        This tool may create or reuse an approval request, but it never
        approves, rejects, resumes, or publishes the Article. A draft may be
        selected because the HITL decision pins its content hash; the Article
        becomes approved only after the user approves the resulting card.
        For ``wechat_official_account``, the title must be at most 32
        characters; for ``xiaohongshu``, the title must be at most 20
        characters; for ``douyin``, the title must be at most 30 characters.
        When the user explicitly selected images in the current request,
        pass their attachment IDs in ``attachment_ids``.  Those IDs are
        pinned to the approval and are used for the eventual publication.
        A channel validation error is non-retryable until the Article is
        revised. For a known length error, the response includes
        ``repairable=true``, the affected field, the maximum length, and the
        required ``revise_article_for_publication`` action. The model must
        revise before trying approval again; it must not repeat the same
        request unchanged.
        """

        thread_id = _thread_id_from_runtime(runtime)
        if thread_id is None:
            return {
                "ok": False,
                "error_code": "missing_thread_id",
                "message": "A thread_id is required to request publication approval.",
            }

        selected_attachment_ids = tuple(
            dict.fromkeys(
                attachment_id.strip()
                for attachment_id in (
                    attachment_ids
                    if attachment_ids is not None
                    else getattr(
                        getattr(runtime, "context", None),
                        "selected_attachment_ids",
                        (),
                    )
                )
                if isinstance(attachment_id, str) and attachment_id.strip()
            )
        )
        attachment_selection_confirmed = bool(
            getattr(
                getattr(runtime, "context", None),
                "attachment_selection_confirmed",
                False,
            )
        ) or attachment_ids is not None

        try:
            intent = PublishingIntent(
                action="request_publication",
                target_hint=target_hint,
                channel=channel,
            )
            resolution = resolver.resolve(intent, thread_id=thread_id)
        except PublishingIntentError as error:
            error_code = (
                "publication_channel_required"
                if "channel" in str(error)
                else "invalid_publishing_intent"
            )
            return {
                "ok": False,
                "error_code": error_code,
                "retryable": False,
                "message": "Please specify which publication channel to use.",
            }
        except Exception:
            log_event(
                logger,
                logging.ERROR,
                "publishing.tool.approval_request_resolution_failed",
                thread_id=thread_id,
                status="failed",
                error_code="intent_resolution_failed",
            )
            return {
                "ok": False,
                "error_code": "intent_resolution_failed",
                "message": "The publication target could not be resolved.",
            }

        def target_payload(target):
            if target is None:
                return None
            return {
                "approval_id": target.approval_id,
                "article_id": target.article_id,
                "article_title": target.article_title,
                "article_slug": target.article_slug,
                "article_version": target.article_version,
                "channel": target.channel.value,
                "approval_status": (
                    target.approval_status.value
                    if target.approval_status is not None
                    else None
                ),
                "publication_status": (
                    target.publication_status.value
                    if target.publication_status is not None
                    else None
                ),
            }

        result = {
            "ok": True,
            "action": intent.action.value,
            "channel": intent.channel.value,
            "resolution_status": resolution.status.value,
            "target": target_payload(resolution.target),
            "candidates": [
                target_payload(candidate)
                for candidate in resolution.candidates
            ],
            "requires_user_confirmation": False,
            "attachment_ids": list(selected_attachment_ids),
        }
        if not resolution.is_resolved or resolution.target is None:
            result["retryable"] = False
            result["error_code"] = (
                "publication_target_not_found"
                if resolution.status.value == "not_found"
                else "publication_target_ambiguous"
            )
            result["message"] = (
                "The publication target was not uniquely resolved."
            )
            return result

        target = resolution.target

        if image_attachment_service is not None and selected_attachment_ids:
            selected_attachment_ids = _repair_generated_attachment_ids(
                selected_attachment_ids,
                _runtime_generated_attachment_ids(runtime),
                image_attachment_service,
            )
            result["attachment_ids"] = list(selected_attachment_ids)
            try:
                for attachment_id in selected_attachment_ids:
                    await asyncio.to_thread(
                        image_attachment_service.get,
                        attachment_id,
                    )
            except Exception:
                return {
                    "ok": False,
                    "error_code": "publication_attachment_not_found",
                    "retryable": False,
                    "requires_user_confirmation": False,
                    "message": (
                        "One or more selected publication images are no longer available."
                    ),
                }

        # Image upload and publication are deliberately separate actions.  On
        # the first publication request, pause at the conversation-level image
        # selector. The next user turn carries the selected IDs back into the
        # agent context, after which this tool creates the real HITL approval.
        if (
            image_attachment_service is not None
            and target.channel.value in {
                "wechat_official_account",
                "xiaohongshu",
                "douyin",
            }
            and not attachment_selection_confirmed
        ):
            result["attachment_selection_required"] = True
            result["attachments"] = await asyncio.to_thread(
                _image_library_preview,
                image_attachment_service,
            )
            result["message"] = (
                "Ask the user to choose images in the conversation card. "
                "Do not create publication approval until the selection is confirmed."
            )
            return result

        try:
            approval = await asyncio.to_thread(
                publishing_service.request_publication_approval,
                target.article_id,
                channel=target.channel,
                actor="agent",
                attachment_ids=selected_attachment_ids,
            )
        except PublishingServiceError as error:
            error_code = getattr(
                error,
                "error_code",
                "publication_approval_request_failed",
            )
            return {
                "ok": False,
                "error_code": error_code,
                "retryable": False,
                **_publication_repair_metadata(error_code),
                "target": target_payload(target),
                "channel": target.channel.value,
                "message": str(error) or (
                    "The publication approval request could not be created."
                ),
            }
        except Exception:
            return {
                "ok": False,
                "error_code": "publication_approval_request_failed",
                "retryable": False,
                "repairable": False,
                "target": target_payload(target),
                "channel": target.channel.value,
                "message": "The publication approval request could not be created.",
            }

        result["target"] = {
            **target_payload(target),
            "approval_id": approval.approval_id,
            "approval_status": approval.status.value,
            "attachment_ids": list(approval.attachment_ids),
        }
        # The durable approval is authoritative.  A replayed approval may
        # carry an older selection, so return that snapshot rather than the
        # current mutable conversation selection.
        result["attachment_ids"] = list(approval.attachment_ids)
        result["approval_required"] = True
        result["requires_user_confirmation"] = True
        result["message"] = (
            "A publication approval request was created; show a confirmation card."
        )

        if hitl_service is not None:
            tool_call_id = getattr(runtime, "tool_call_id", None)
            if (
                enable_native_interrupt
                or not isinstance(tool_call_id, str)
                or not tool_call_id.strip()
            ):
                tool_call_id = (
                    f"publication-request:{thread_id}:"
                    f"{approval.approval_id}:{approval.article_version}"
                )
            try:
                interaction = await asyncio.to_thread(
                    hitl_service.create_or_get_interaction,
                    thread_id=thread_id,
                    run_id=tool_call_id,
                    action=HITLAction.APPROVE,
                    target_type="publication_approval",
                    target_id=approval.approval_id,
                    target_version=approval.article_version,
                )
            except HITLThreadBusyError as error:
                blocking = error.interaction
                log_event(
                    logger,
                    logging.INFO,
                    "publishing.tool.hitl_blocked",
                    thread_id=thread_id,
                    status="waiting",
                    error_code="hitl_pending",
                )
                return {
                    "ok": False,
                    "error_code": "hitl_pending",
                    "retryable": False,
                    "requires_user_confirmation": False,
                    "blocking_interaction_id": blocking.interaction_id,
                    "message": (
                        "An earlier HITL interaction is still unresolved. "
                        "Complete it before creating another HITL."
                    ),
                }
            result["interaction_id"] = interaction.interaction_id
            result["interaction_status"] = interaction.status.value
            if (
                enable_native_interrupt
                and interaction.status is HITLStatus.PENDING
            ):
                resume_value = interrupt({
                    "kind": "publication_approval",
                    **result,
                })
                if isinstance(resume_value, dict):
                    decision = resume_value.get("decision")
                    result["resumed"] = True
                    result["resume_decision"] = (
                        decision if isinstance(decision, str) else None
                    )
                    if decision == HITLStatus.APPROVED.value:
                        result["target"] = {
                            **result["target"],
                            "approval_status": HITLStatus.APPROVED.value,
                        }
                        result["approval_required"] = False
                        result["requires_user_confirmation"] = False
                        result["message"] = (
                            "HITL approval was recorded. Do not request approval "
                            "again; wait for an explicit publish action."
                        )
                    elif decision == HITLStatus.REJECTED.value:
                        result["target"] = {
                            **result["target"],
                            "approval_status": HITLStatus.REJECTED.value,
                        }
                        result["approval_required"] = False
                        result["requires_user_confirmation"] = False
                        result["message"] = (
                            "HITL rejection was recorded. Do not request approval "
                            "again; wait for a revision or a new request."
                        )

        log_event(
            logger,
            logging.INFO,
            "publishing.tool.approval_requested",
            thread_id=thread_id,
            status="waiting_approval",
        )
        return result

    return request_publication_approval
