"""Upload one WeChat cover image and print its permanent media ID.

This is a one-time operator tool. It only uploads a ``thumb`` material; it
does not create a draft or publish an article.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from ..publishing.publishers.wechat_api import (
    WeChatAccessTokenProvider,
    WeChatApiError,
    WeChatCredentials,
    WeChatOfficialAccountClient,
    WeChatTokenError,
)
from ..publishing.repository import PublishingRepository
from ..publishing.wechat_covers import WeChatCoverService


class WeChatMediaSettings(BaseSettings):
    """Load only the credentials needed by this operator command."""

    wechat_app_id: str | None = None
    wechat_app_secret: str | None = None
    publishing_db_path: str = "deep_research_publishing.sqlite"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        str_strip_whitespace=True,
    )


_CONTENT_TYPES = {
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
}
_MAX_COVER_BYTES = 64 * 1024


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Upload one permanent WeChat thumb material and print its media_id. "
            "This command does not create or publish an article."
        ),
    )
    parser.add_argument(
        "--file",
        required=True,
        type=Path,
        help="Local JPG, PNG, or GIF cover image to upload.",
    )
    return parser


def upload_cover(path: Path) -> str:
    """Upload ``path`` using .env credentials and return its media ID."""

    if not path.is_file():
        raise ValueError("cover_file_not_found")

    content_type = _CONTENT_TYPES.get(path.suffix.lower())
    if content_type is None:
        raise ValueError("cover_file_type_unsupported")

    content = path.read_bytes()
    if not content:
        raise ValueError("cover_file_empty")
    if len(content) > _MAX_COVER_BYTES:
        raise ValueError("cover_file_too_large")

    config = WeChatMediaSettings()
    if not config.wechat_app_id or not config.wechat_app_secret:
        raise ValueError("wechat_credentials_missing")

    credentials = WeChatCredentials(
        app_id=config.wechat_app_id,
        app_secret=config.wechat_app_secret,
    )
    token_provider = WeChatAccessTokenProvider(credentials)
    client = WeChatOfficialAccountClient(token_provider)
    repository = PublishingRepository(config.publishing_db_path)
    repository.initialize()
    try:
        asset = WeChatCoverService(repository, client).upload_cover(
            content,
            filename=path.name,
            content_type=content_type,
            make_active=True,
        )
        return asset.remote_media_id
    finally:
        repository.close()


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        media_id = upload_cover(args.file)
    except (OSError, ValueError) as error:
        print(f"wechat_cover_upload_failed: {error}", file=sys.stderr)
        return 2
    except (WeChatApiError, WeChatTokenError) as error:
        details = error.error_code
        if (
            isinstance(error, (WeChatApiError, WeChatTokenError))
            and error.provider_code is not None
        ):
            details += f" provider_errcode={error.provider_code}"
        print(
            f"wechat_cover_upload_failed: {details}",
            file=sys.stderr,
        )
        return 3

    # Keep stdout copy-friendly and never print credentials or access tokens.
    print(media_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "upload_cover"]
