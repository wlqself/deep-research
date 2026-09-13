"""Asynchronous understanding for user-uploaded images.

The first adapter uses an OpenAI-compatible vision endpoint.  OCR-specific
engines can be added later without changing the attachment or HTTP layers.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from .models import ImageAttachmentAnalysis, utc_now


logger = logging.getLogger("deep_research.image_analysis")

IMAGE_TYPES = {
    "photo",
    "document",
    "screenshot",
    "question",
    "table_or_chart",
    "unknown",
}


class ImageAnalysisService:
    """Schedule and persist idempotent image analysis jobs."""

    def __init__(
        self,
        repository: Any,
        *,
        model_name: str | None,
        api_key: str | None,
        base_url: str | None,
        enabled: bool = True,
        timeout_seconds: int = 60,
    ) -> None:
        self.repository = repository
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = (
            base_url or "https://api.openai.com/v1"
        ).rstrip("/")
        self.enabled = enabled
        self.timeout_seconds = timeout_seconds
        self._tasks: set[asyncio.Task[Any]] = set()
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, attachment_id: str) -> asyncio.Lock:
        lock = self._locks.get(attachment_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[attachment_id] = lock
        return lock

    def schedule(self, attachment_id: str) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Persistence remains useful for synchronous callers and the
            # application handler schedules jobs from its active loop.
            return
        task = loop.create_task(
            self.analyze(attachment_id),
            name=f"image-analysis:{attachment_id}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def enqueue(self, attachment_id: str) -> None:
        """Create a visible pending record before starting background work."""
        self.ensure_pending(attachment_id)
        self.schedule(attachment_id)

    def ensure_pending(self, attachment_id: str) -> None:
        """Persist the queue state without touching the event loop."""
        if self.repository.get_image_attachment_analysis(attachment_id) is None:
            now = utc_now()
            self.repository.save_image_attachment_analysis(
                ImageAttachmentAnalysis(
                    analysis_id=uuid4().hex,
                    attachment_id=attachment_id,
                    status="pending",
                    analysis_model=self.model_name,
                    created_at=now,
                    updated_at=now,
                )
            )

    async def close(self) -> None:
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    async def analyze(
        self,
        attachment_id: str,
        *,
        force: bool = False,
        focus: str | None = None,
    ) -> None:
        async with self._lock_for(attachment_id):
            existing = await asyncio.to_thread(
                self.repository.get_image_attachment_analysis,
                attachment_id,
            )
            if not force and existing is not None and existing.status == "completed":
                return
            await self._analyze_unlocked(attachment_id, focus=focus)

    async def analyze_now(
        self,
        attachment_id: str,
        *,
        force: bool = False,
        focus: str | None = None,
    ) -> ImageAttachmentAnalysis | None:
        """Run an on-demand analysis for an Agent tool call."""
        self.ensure_pending(attachment_id)
        await self.analyze(attachment_id, force=force, focus=focus)
        return await asyncio.to_thread(
            self.repository.get_image_attachment_analysis,
            attachment_id,
        )

    async def _analyze_unlocked(
        self,
        attachment_id: str,
        *,
        focus: str | None = None,
    ) -> None:
        try:
            attachment = await asyncio.to_thread(
                self.repository.get_image_attachment,
                attachment_id,
            )
            await self._save_status(
                attachment_id,
                status="analyzing",
                error_code=None,
            )

            if not self.enabled or not self.model_name or not self.api_key:
                await self._save_status(
                    attachment_id,
                    status="skipped",
                    error_code="image_analysis_not_configured",
                )
                return

            content = await asyncio.to_thread(
                lambda: Path(attachment.storage_path).read_bytes()
            )
            result = await self._request_vision_analysis(
                content,
                attachment.content_type,
                attachment.filename,
                focus=focus,
            )
            await asyncio.to_thread(
                self.repository.save_image_attachment_analysis,
                ImageAttachmentAnalysis(
                    analysis_id=uuid4().hex,
                    attachment_id=attachment_id,
                    status="completed",
                    image_type=result["image_type"],
                    confidence=result["confidence"],
                    summary=result["summary"],
                    ocr_text=result["ocr_text"],
                    structured_result=json.dumps(
                        result,
                        ensure_ascii=False,
                    ),
                    analysis_model=self.model_name,
                    updated_at=utc_now(),
                ),
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "image analysis failed",
                extra={"attachment_id": attachment_id},
            )
            try:
                await self._save_status(
                    attachment_id,
                    status="failed",
                    error_code="image_analysis_failed",
                )
            except Exception:
                logger.exception("failed to persist image analysis failure")

    async def _save_status(
        self,
        attachment_id: str,
        *,
        status: str,
        error_code: str | None,
    ) -> None:
        existing = await asyncio.to_thread(
            self.repository.get_image_attachment_analysis,
            attachment_id,
        )
        now = utc_now()
        analysis = ImageAttachmentAnalysis(
            analysis_id=existing.analysis_id if existing else uuid4().hex,
            attachment_id=attachment_id,
            status=status,
            image_type=existing.image_type if existing else None,
            confidence=existing.confidence if existing else None,
            summary=existing.summary if existing else None,
            ocr_text=existing.ocr_text if existing else None,
            structured_result=existing.structured_result if existing else None,
            analysis_model=self.model_name,
            analysis_version=existing.analysis_version if existing else "v1",
            error_code=error_code,
            created_at=existing.created_at if existing else now,
            updated_at=now,
        )
        await asyncio.to_thread(
            self.repository.save_image_attachment_analysis,
            analysis,
        )

    async def _request_vision_analysis(
        self,
        content: bytes,
        content_type: str,
        filename: str,
        *,
        focus: str | None = None,
    ) -> dict[str, Any]:
        encoded = base64.b64encode(content).decode("ascii")
        prompt = (
            "分析这张用户上传的图片，并严格只返回 JSON，不要 Markdown。"
            "image_type 只能是 photo、document、screenshot、question、"
            "table_or_chart、unknown 之一。"
            "如果图片含有文字，ocr_text 提取可读文字；如果是题目，保留题干、"
            "公式和选项；如果是表格或图表，描述其结构和关键数据。"
            "photo 用 summary 描述主体、场景和用途。"
            "JSON 字段必须为：image_type、confidence、summary、ocr_text、"
            "key_points。confidence 为 0 到 1 的数字，key_points 为字符串数组。"
            f"文件名：{filename}。"
        )
        if isinstance(focus, str) and focus.strip():
            prompt += (
                "用户正在复核此前的图片理解，请重点重新核对以下问题，并以当前图片为准："
                f"{focus.strip()[:1000]}。"
            )
        payload = {
            "model": self.model_name,
            "stream": False,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{content_type};base64,{encoded}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            body = response.json()

        content_value = (
            body.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        if isinstance(content_value, list):
            content_value = "".join(
                item.get("text", "")
                for item in content_value
                if isinstance(item, dict)
            )
        if not isinstance(content_value, str) or not content_value.strip():
            raise ValueError("image_analysis_empty_response")

        parsed = self._parse_json(content_value)
        image_type = parsed.get("image_type")
        if image_type not in IMAGE_TYPES:
            image_type = "unknown"
        confidence = parsed.get("confidence", 0)
        try:
            confidence = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence = 0.0
        summary = parsed.get("summary")
        ocr_text = parsed.get("ocr_text")
        key_points = parsed.get("key_points", [])
        return {
            "image_type": image_type,
            "confidence": confidence,
            "summary": str(summary).strip()[:4000]
            if summary is not None
            else "",
            "ocr_text": str(ocr_text).strip()[:20000]
            if ocr_text is not None
            else "",
            "key_points": [
                str(item).strip()[:500]
                for item in key_points
                if isinstance(item, str) and item.strip()
            ][:20],
        }

    @staticmethod
    def _parse_json(value: str) -> dict[str, Any]:
        candidate = value.strip()
        if candidate.startswith("```"):
            candidate = candidate.strip("`").strip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start < 0 or end <= start:
                raise ValueError("image_analysis_invalid_json")
            parsed = json.loads(candidate[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("image_analysis_invalid_json")
        return parsed


__all__ = ["ImageAnalysisService"]
