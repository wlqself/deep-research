"""PublicationPublisher implementation for Xiaohongshu."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from ...models import Article, ArticleStatus, PublicationChannel, PublicationStatus
from ...inline_images import (
    extract_inline_image_references,
    ordered_unique_attachment_ids,
)
from ...service_support import (
    PublicationDeliveryUnknownError,
    PublicationExecutionError,
    PublicationPollingError,
    PublicationResult,
    PublicationValidationError,
)
from .client import (
    XiaohongshuApiError,
    XiaohongshuClient,
    XiaohongshuDeliveryUnknownError,
)
from .formatter import format_xiaohongshu_content


class _XiaohongshuClient(Protocol):
    def publish_note(self, *, title: str, content: str, images: list[str], tags: list[str]):
        ...

    def get_publication_status(self, external_id: str):
        ...


class XiaohongshuPublisher:
    """Adapt an approved Article to a local Xiaohongshu browser service."""

    channel = PublicationChannel.XIAOHONGSHU

    def __init__(
        self,
        client: _XiaohongshuClient | XiaohongshuClient,
        *,
        image_paths_resolver: Callable[[Article, tuple[str, ...]], list[str]] | None = None,
        max_images: int = 9,
        title_max_length: int = 20,
        content_max_length: int = 1000,
    ) -> None:
        if max_images < 1:
            raise ValueError("max_images must be positive")
        if content_max_length < 1:
            raise ValueError("content_max_length must be positive")
        self._client = client
        self._image_paths_resolver = image_paths_resolver
        self._max_images = max_images
        self._title_max_length = title_max_length
        self._content_max_length = content_max_length

    def publish(
        self,
        article: Article,
        *,
        attachment_ids: tuple[str, ...] = (),
    ) -> PublicationResult:
        if article.status not in {ArticleStatus.PUBLISHING, ArticleStatus.PUBLISHED}:
            raise PublicationExecutionError("article_not_publishing")
        content = format_xiaohongshu_content(
            article,
            max_title_length=self._title_max_length,
            max_content_length=self._content_max_length,
        )
        try:
            inline_references = extract_inline_image_references(
                article.markdown_content,
            )
            resolved_attachment_ids = ordered_unique_attachment_ids(
                inline_references,
                attachment_ids,
            )
            image_paths = (
                self._image_paths_resolver(article, resolved_attachment_ids)
                if self._image_paths_resolver is not None
                else []
            )
            if not isinstance(image_paths, list) or len(image_paths) > self._max_images:
                raise PublicationValidationError(
                    "xiaohongshu_image_count_invalid",
                    "Xiaohongshu image count is outside the configured limit",
                )
            normalized_paths = []
            for image_path in image_paths:
                path = Path(image_path)
                if not path.is_file():
                    raise PublicationValidationError(
                        "xiaohongshu_image_not_found",
                        "Xiaohongshu image file does not exist",
                    )
                normalized_paths.append(str(path.resolve()))
            if not normalized_paths:
                raise PublicationValidationError(
                    "xiaohongshu_image_required",
                    "Xiaohongshu requires at least one image",
                )
            receipt = self._client.publish_note(
                title=content.title,
                content=content.content,
                images=normalized_paths,
                tags=list(content.tags),
            )
        except PublicationValidationError:
            raise
        except XiaohongshuDeliveryUnknownError as error:
            raise PublicationDeliveryUnknownError(
                error.error_code,
                external_id=error.external_id,
            ) from error
        except XiaohongshuApiError as error:
            raise PublicationExecutionError(error.error_code) from error
        except Exception as error:
            raise PublicationExecutionError("xiaohongshu_publisher_failed") from error

        return PublicationResult(
            status=receipt.status,
            external_id=receipt.external_id,
            public_url=receipt.public_url,
        )

    def get_publication_status(self, external_id: str) -> PublicationResult:
        try:
            receipt = self._client.get_publication_status(external_id)
        except XiaohongshuApiError as error:
            if error.error_code in {
                "xiaohongshu_service_unreachable",
                "xiaohongshu_http_error",
            }:
                raise PublicationPollingError(error.error_code) from error
            raise PublicationExecutionError(error.error_code) from error
        except Exception as error:
            raise PublicationPollingError("xiaohongshu_status_failed") from error
        return PublicationResult(
            status=receipt.status,
            external_id=receipt.external_id,
            public_url=receipt.public_url,
        )


__all__ = ["XiaohongshuPublisher"]
