"""PublicationPublisher implementation for Douyin image-text posts."""

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
from .client import DouyinApiError, DouyinClient, DouyinDeliveryUnknownError
from .formatter import format_douyin_content


class _DouyinClient(Protocol):
    def publish_image_note(
        self,
        *,
        title: str,
        content: str,
        images: list[str],
        tags: list[str],
    ) -> object:
        ...

    def get_publication_status(self, external_id: str) -> object:
        ...


class DouyinPublisher:
    """Adapt an approved Article to a local Douyin browser service."""

    channel = PublicationChannel.DOUYIN

    def __init__(
        self,
        client: _DouyinClient | DouyinClient,
        *,
        image_paths_resolver: Callable[[Article, tuple[str, ...]], list[str]] | None = None,
        max_images: int = 30,
        title_max_length: int = 30,
    ) -> None:
        if max_images < 1:
            raise ValueError("max_images must be positive")
        self._client = client
        self._image_paths_resolver = image_paths_resolver
        self._max_images = max_images
        self._title_max_length = title_max_length

    def publish(
        self,
        article: Article,
        *,
        attachment_ids: tuple[str, ...] = (),
    ) -> PublicationResult:
        if article.status not in {ArticleStatus.PUBLISHING, ArticleStatus.PUBLISHED}:
            raise PublicationExecutionError("article_not_publishing")
        content = format_douyin_content(
            article,
            max_title_length=self._title_max_length,
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
                    "douyin_image_count_invalid",
                    "Douyin image count is outside the configured limit",
                )
            normalized_paths = []
            for image_path in image_paths:
                path = Path(image_path)
                if not path.is_file():
                    raise PublicationValidationError(
                        "douyin_image_not_found",
                        "Douyin image file does not exist",
                    )
                normalized_paths.append(str(path.resolve()))
            if not normalized_paths:
                raise PublicationValidationError(
                    "douyin_image_required",
                    "Douyin image-text posts require at least one image",
                )
            receipt = self._client.publish_image_note(
                title=content.title,
                content=content.content,
                images=normalized_paths,
                tags=list(content.tags),
            )
        except PublicationValidationError as error:
            raise PublicationExecutionError(error.error_code) from error
        except DouyinDeliveryUnknownError as error:
            raise PublicationDeliveryUnknownError(
                error.error_code,
                external_id=error.external_id,
            ) from error
        except DouyinApiError as error:
            raise PublicationExecutionError(error.error_code) from error
        except Exception as error:
            raise PublicationExecutionError("douyin_publisher_failed") from error

        return PublicationResult(
            status=receipt.status,
            external_id=receipt.external_id,
            public_url=receipt.public_url,
        )

    def get_publication_status(self, external_id: str) -> PublicationResult:
        try:
            receipt = self._client.get_publication_status(external_id)
        except DouyinApiError as error:
            if error.error_code in {
                "douyin_service_unreachable",
                "douyin_http_error",
            }:
                raise PublicationPollingError(error.error_code) from error
            raise PublicationExecutionError(error.error_code) from error
        except Exception as error:
            raise PublicationPollingError("douyin_status_failed") from error
        return PublicationResult(
            status=receipt.status,
            external_id=receipt.external_id,
            public_url=receipt.public_url,
        )


__all__ = ["DouyinPublisher"]
