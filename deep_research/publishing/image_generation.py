"""Provider-neutral image generation backed by SiliconFlow.

The service deliberately turns remote generation results into ordinary image
attachments immediately.  Publishing and the image library therefore only
need to understand ``attachment_id`` and never need to know which provider
created the file.
"""

from __future__ import annotations

import base64
import asyncio
import logging
import mimetypes
from typing import Any
from uuid import uuid4

import httpx

from .attachments import ImageAttachmentService


logger = logging.getLogger("deep_research.image_generation")

_CONTENT_TYPES = {"image/jpeg", "image/png", "image/gif"}
_MAX_REMOTE_IMAGE_BYTES = 10 * 1024 * 1024


class ImageGenerationError(RuntimeError):
    """A safe, user-facing image generation failure."""

    def __init__(self, error_code: str, message: str = "") -> None:
        super().__init__(message or error_code)
        self.error_code = error_code


class SiliconFlowImageGenerationService:
    """Generate images and persist them in the shared attachment library."""

    def __init__(
        self,
        attachment_service: ImageAttachmentService,
        *,
        model_name: str | None,
        api_key: str | None,
        base_url: str | None,
        enabled: bool = True,
        timeout_seconds: int = 180,
        prompt_max_chars: int = 800,
        max_images: int = 4,
    ) -> None:
        self.attachment_service = attachment_service
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = (base_url or "https://api.siliconflow.cn/v1").rstrip("/")
        self.enabled = enabled
        self.timeout_seconds = timeout_seconds
        self.prompt_max_chars = prompt_max_chars
        self.max_images = max_images

    async def generate(
        self,
        *,
        thread_id: str,
        prompt: str,
        negative_prompt: str = "",
        image_size: str = "1024x1024",
        count: int = 1,
        seed: int | None = None,
    ) -> list[dict[str, Any]]:
        self._validate(prompt, negative_prompt, image_size, count)
        if not self.enabled or not self.model_name or not self.api_key:
            raise ImageGenerationError(
                "image_generation_not_configured",
                "Image generation is not configured.",
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        generated: list[dict[str, Any]] = []
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            # One request per image keeps the adapter compatible with both
            # current and older SiliconFlow batch semantics.
            for index in range(count):
                payload: dict[str, Any] = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "image_size": image_size,
                }
                if negative_prompt.strip():
                    payload["negative_prompt"] = negative_prompt.strip()
                if seed is not None:
                    payload["seed"] = seed + index
                try:
                    response = await client.post(
                        f"{self.base_url}/images/generations",
                        headers=headers,
                        json=payload,
                    )
                    response.raise_for_status()
                    data = response.json()
                except httpx.TimeoutException as error:
                    raise ImageGenerationError(
                        "image_generation_timeout",
                        "The image generation service timed out.",
                    ) from error
                except httpx.HTTPStatusError as error:
                    logger.warning(
                        "image generation provider returned HTTP %s",
                        error.response.status_code,
                    )
                    raise ImageGenerationError(
                        "image_generation_provider_error",
                        "The image generation service rejected the request.",
                    ) from error
                except (httpx.HTTPError, ValueError, TypeError) as error:
                    raise ImageGenerationError(
                        "image_generation_provider_error",
                        "The image generation service could not be reached.",
                    ) from error

                item = self._first_image(data)
                content, content_type = await self._download_image(client, item)
                attachment = await self._save_attachment(
                    thread_id=thread_id,
                    content=content,
                    content_type=content_type,
                    index=index,
                )
                generated.append(
                    {
                        "attachment_id": attachment.attachment_id,
                        "filename": attachment.filename,
                        "content_type": attachment.content_type,
                        "size_bytes": attachment.size_bytes,
                        "seed": data.get("seed"),
                    }
                )
        return generated

    def _validate(
        self,
        prompt: str,
        negative_prompt: str,
        image_size: str,
        count: int,
    ) -> None:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ImageGenerationError("image_generation_prompt_required")
        if len(prompt) > self.prompt_max_chars:
            raise ImageGenerationError("image_generation_prompt_too_long")
        if not isinstance(negative_prompt, str):
            raise ImageGenerationError("image_generation_negative_prompt_invalid")
        if len(negative_prompt) > self.prompt_max_chars:
            raise ImageGenerationError("image_generation_negative_prompt_too_long")
        if (
            not isinstance(image_size, str)
            or not image_size.strip()
            or "x" not in image_size.lower()
        ):
            raise ImageGenerationError("image_generation_image_size_invalid")
        try:
            width, height = (int(value) for value in image_size.lower().split("x", 1))
        except ValueError as error:
            raise ImageGenerationError("image_generation_image_size_invalid") from error
        if width < 256 or height < 256 or width > 2048 or height > 2048:
            raise ImageGenerationError("image_generation_image_size_invalid")
        if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= self.max_images:
            raise ImageGenerationError("image_generation_count_invalid")

    @staticmethod
    def _first_image(data: Any) -> dict[str, Any]:
        images = data.get("images") if isinstance(data, dict) else None
        if not isinstance(images, list) or not images or not isinstance(images[0], dict):
            raise ImageGenerationError("image_generation_empty_result")
        item = images[0]
        if not item.get("url") and not item.get("b64_json"):
            raise ImageGenerationError("image_generation_empty_result")
        return item

    async def _download_image(
        self,
        client: httpx.AsyncClient,
        item: dict[str, Any],
    ) -> tuple[bytes, str]:
        b64_json = item.get("b64_json")
        if isinstance(b64_json, str) and b64_json:
            try:
                content = base64.b64decode(b64_json, validate=True)
            except (ValueError, TypeError) as error:
                raise ImageGenerationError("image_generation_result_invalid") from error
            content_type = "image/png"
        else:
            url = item.get("url")
            if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                raise ImageGenerationError("image_generation_result_invalid")
            try:
                response = await client.get(url)
                response.raise_for_status()
            except (httpx.HTTPError, ValueError) as error:
                raise ImageGenerationError("image_generation_download_failed") from error
            content = response.content
            content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
            if content_type not in _CONTENT_TYPES:
                content_type = mimetypes.guess_type(url)[0] or "image/png"
        if not content or len(content) > _MAX_REMOTE_IMAGE_BYTES:
            raise ImageGenerationError("image_generation_result_too_large")
        if content_type not in _CONTENT_TYPES:
            raise ImageGenerationError("image_generation_result_type_unsupported")
        return content, content_type

    async def _save_attachment(
        self,
        *,
        thread_id: str,
        content: bytes,
        content_type: str,
        index: int,
    ):
        suffix = {"image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif"}[content_type]
        return await asyncio.to_thread(
            self.attachment_service.save,
            thread_id=thread_id,
            filename=f"generated-{uuid4().hex[:12]}-{index + 1}{suffix}",
            content_type=content_type,
            content=content,
        )


__all__ = ["ImageGenerationError", "SiliconFlowImageGenerationService"]
