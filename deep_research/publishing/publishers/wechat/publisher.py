"""WeChat Official Account publisher adapter."""

from __future__ import annotations

from collections.abc import Callable
from html import escape
from pathlib import Path
from typing import Protocol

from ...models import Article, ArticleStatus, PublicationChannel, PublicationStatus
from ...inline_images import extract_inline_image_references
from ...renderers.wechat import WECHAT_THEMES, render_wechat_html
from ...service_support import (
    PublicationDeliveryUnknownError,
    PublicationExecutionError,
    PublicationPollingError,
    PublicationResult,
    PublicationValidationError,
)
from .api import (
    WeChatApiError,
    WeChatDeliveryUnknownError,
    WeChatOfficialAccountClient,
    WeChatTokenError,
    WeChatPublishStatus,
)
from .drafts import WeChatDraftValidationError, build_wechat_news_draft


class _WeChatPublisherClient(Protocol):
    def add_draft(self, payload: dict[str, object]) -> str:
        ...

    def submit_draft(self, media_id: str) -> str:
        ...

    def upload_cover(
        self,
        content: bytes,
        *,
        filename: str,
        content_type: str = "image/jpeg",
    ) -> str:
        ...

    def upload_inline_image(
        self,
        content: bytes,
        *,
        filename: str,
        content_type: str = "image/jpeg",
    ) -> str:
        ...


class WeChatOfficialAccountPublisher:
    """Turn one approved Article version into a WeChat publication request."""

    channel = PublicationChannel.WECHAT_OFFICIAL_ACCOUNT
    _FAILED_STATUS_CODES = {
        2: "wechat_original_declaration_failed",
        3: "wechat_publish_failed",
        4: "wechat_audit_rejected",
        5: "wechat_article_deleted",
        6: "wechat_account_banned",
    }

    def __init__(
        self,
        client: _WeChatPublisherClient | WeChatOfficialAccountClient,
        *,
        thumb_media_id: str | None,
        cover_media_id_resolver: Callable[[], str | None] | None = None,
        attachment_resolver: Callable[
            [Article, tuple[str, ...]], list[tuple[str, str]]
        ] | None = None,
        author: str | None = None,
        publish_mode: str = "draft",
        theme: str = "minimal",
        content_source_url: str | None = None,
    ) -> None:
        if publish_mode not in {"draft", "publish"}:
            raise ValueError("publish_mode must be 'draft' or 'publish'")
        if theme not in WECHAT_THEMES:
            raise ValueError("unsupported WeChat theme")
        if (
            (not isinstance(thumb_media_id, str) or not thumb_media_id.strip())
            and cover_media_id_resolver is None
            and attachment_resolver is None
        ):
            raise ValueError("a WeChat cover source must be configured")

        self._client = client
        self._thumb_media_id = (
            thumb_media_id.strip() if isinstance(thumb_media_id, str) else None
        )
        self._cover_media_id_resolver = cover_media_id_resolver
        self._attachment_resolver = attachment_resolver
        self._author = author
        self._publish_mode = publish_mode
        self._theme = theme
        self._content_source_url = content_source_url

    def publish(
        self,
        article: Article,
        *,
        attachment_ids: tuple[str, ...] = (),
    ) -> PublicationResult:
        if article.status not in {ArticleStatus.PUBLISHING, ArticleStatus.PUBLISHED}:
            raise PublicationExecutionError("article_not_publishing")

        media_id: str | None = None
        try:
            inline_references = extract_inline_image_references(
                article.markdown_content,
            )
            selected_attachments = self._resolve_attachments(
                article,
                attachment_ids,
            )
            inline_urls: dict[str, str] = {}
            if inline_references:
                inline_ids: list[str] = []
                seen_inline_ids: set[str] = set()
                for reference in inline_references:
                    if reference.attachment_id not in seen_inline_ids:
                        seen_inline_ids.add(reference.attachment_id)
                        inline_ids.append(reference.attachment_id)
                inline_attachments = self._resolve_attachments(
                    article,
                    tuple(inline_ids),
                )
                for attachment_id, (image_path, content_type) in zip(
                    inline_ids,
                    inline_attachments,
                    strict=True,
                ):
                    inline_urls[attachment_id] = self._client.upload_inline_image(
                        Path(image_path).read_bytes(),
                        filename=Path(image_path).name,
                        content_type=content_type,
                    )

            def render_inline_image(attachment_id: str, alt: str) -> str:
                image_url = inline_urls.get(attachment_id)
                if not image_url:
                    raise PublicationExecutionError("wechat_image_not_found")
                return (
                    '<img src="'
                    + escape(image_url, quote=True)
                    + '" alt="'
                    + escape(alt, quote=True)
                    + '" />'
                )

            content_html = render_wechat_html(
                article.markdown_content,
                theme=self._theme,
                image_renderer=render_inline_image if inline_references else None,
            )
            if selected_attachments:
                cover_path, cover_type = selected_attachments[0]
                media_id = self._client.upload_cover(
                    Path(cover_path).read_bytes(),
                    filename=Path(cover_path).name,
                    content_type=cover_type,
                )
                trailing_urls = []
                inline_ids = set(inline_urls)
                for index, (image_path, content_type) in enumerate(
                    selected_attachments[1:],
                    start=1,
                ):
                    selected_id = (
                        attachment_ids[index]
                        if index < len(attachment_ids)
                        else None
                    )
                    if selected_id in inline_ids:
                        continue
                    trailing_urls.append(
                        self._client.upload_inline_image(
                            Path(image_path).read_bytes(),
                            filename=Path(image_path).name,
                            content_type=content_type,
                        )
                    )
                content_html = self._append_inline_images(content_html, trailing_urls)
            else:
                media_id = (
                    self._cover_media_id_resolver()
                    if self._cover_media_id_resolver is not None
                    else self._thumb_media_id
                )

            if not isinstance(media_id, str) or not media_id.strip():
                raise PublicationExecutionError("wechat_cover_not_configured")
            payload = build_wechat_news_draft(
                article,
                content_html=content_html,
                thumb_media_id=media_id,
                author=self._author,
                content_source_url=self._content_source_url,
            )
            external_id = self._client.add_draft(payload)
            if self._publish_mode == "publish":
                external_id = self._client.submit_draft(external_id)
        except WeChatDraftValidationError as error:
            raise PublicationExecutionError(error.error_code) from error
        except WeChatDeliveryUnknownError as error:
            raise PublicationDeliveryUnknownError(
                error.error_code,
                external_id=media_id,
            ) from error
        except WeChatApiError as error:
            raise PublicationExecutionError(error.error_code) from error
        except WeChatTokenError as error:
            raise PublicationExecutionError(error.error_code) from error
        except PublicationValidationError as error:
            raise PublicationExecutionError(error.error_code) from error
        except PublicationExecutionError:
            raise
        except (OSError, ValueError) as error:
            raise PublicationExecutionError("wechat_image_invalid") from error
        except Exception as error:
            raise PublicationExecutionError("wechat_publisher_failed") from error

        return PublicationResult(
            status=(
                PublicationStatus.DRAFTED
                if self._publish_mode == "draft"
                else PublicationStatus.PUBLISHING
            ),
            external_id=external_id,
        )

    def _resolve_attachments(
        self,
        article: Article,
        attachment_ids: tuple[str, ...],
    ) -> list[tuple[str, str]]:
        if not attachment_ids:
            return []
        if self._attachment_resolver is None:
            raise PublicationExecutionError("wechat_image_not_configured")
        attachments = self._attachment_resolver(article, attachment_ids)
        if len(attachments) != len(attachment_ids):
            raise PublicationExecutionError("wechat_image_not_found")
        return attachments

    @staticmethod
    def _append_inline_images(content_html: str, image_urls: list[str]) -> str:
        if not image_urls:
            return content_html
        image_html = "".join(
            '<p><img src="'
            + escape(url, quote=True)
            + '" style="max-width:100%;height:auto;" /></p>'
            for url in image_urls
        )
        return content_html + image_html

    @property
    def client(self):
        return self._client

    @property
    def publish_mode(self) -> str:
        return self._publish_mode

    def get_publication_status(self, external_id: str) -> PublicationResult:
        if not isinstance(external_id, str) or not external_id.strip():
            raise PublicationExecutionError("wechat_publish_id_invalid")
        if self._publish_mode == "draft":
            return PublicationResult(
                status=PublicationStatus.DRAFTED,
                external_id=external_id.strip(),
            )
        try:
            status = self._client.get_publish_status(external_id.strip())
        except WeChatApiError as error:
            if error.error_code == "wechat_publish_status_request_failed":
                raise PublicationPollingError(error.error_code) from error
            raise PublicationExecutionError(error.error_code) from error
        except WeChatTokenError as error:
            if error.error_code == "wechat_token_request_failed":
                raise PublicationPollingError(error.error_code) from error
            raise PublicationExecutionError(error.error_code) from error
        except Exception as error:
            raise PublicationPollingError("wechat_publish_status_failed") from error
        return self._map_publish_status(status)

    def _map_publish_status(self, status: WeChatPublishStatus) -> PublicationResult:
        if status.publish_status == 0:
            return PublicationResult(
                status=PublicationStatus.PUBLISHED,
                external_id=status.article_id or status.publish_id,
                public_url=status.article_url,
            )
        if status.publish_status == 1:
            return PublicationResult(
                status=PublicationStatus.PUBLISHING,
                external_id=status.publish_id,
            )
        raise PublicationExecutionError(
            self._FAILED_STATUS_CODES.get(
                status.publish_status,
                "wechat_publish_status_unknown",
            )
        )


__all__ = ["WeChatOfficialAccountPublisher"]
