"""Versioned WeChat cover-material management."""

from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import uuid4

from .models import WeChatCoverAsset, utc_now


class WeChatCoverService:
    """Upload, cache, rotate, and resolve permanent WeChat cover materials."""

    def __init__(self, repository, client=None) -> None:
        self.repository = repository
        self.client = client

    def active_media_id(self, fallback: str | None = None) -> str | None:
        asset = self.repository.get_active_wechat_cover_asset()
        if asset is not None:
            return asset.remote_media_id
        return fallback.strip() if isinstance(fallback, str) and fallback.strip() else None

    def register_uploaded_cover(
        self,
        *,
        content_sha256: str,
        remote_media_id: str,
        make_active: bool = True,
    ) -> WeChatCoverAsset:
        asset = WeChatCoverAsset(
            asset_id=uuid4().hex,
            content_sha256=content_sha256,
            remote_media_id=remote_media_id,
            is_active=make_active,
            last_verified_at=utc_now(),
        )
        return self.repository.save_wechat_cover_asset(asset, make_active=make_active)

    def upload_cover(
        self,
        content: bytes,
        *,
        filename: str,
        content_type: str = "image/jpeg",
        make_active: bool = True,
    ) -> WeChatCoverAsset:
        if self.client is None:
            raise RuntimeError("wechat_cover_client_not_configured")
        digest = hashlib.sha256(content).hexdigest()
        existing = self.repository.get_wechat_cover_asset_by_hash(digest)
        if existing is not None:
            if make_active and not existing.is_active:
                return self.repository.set_active_wechat_cover_asset(existing.asset_id)
            return existing
        remote_media_id = self.client.upload_cover(
            content,
            filename=filename,
            content_type=content_type,
        )
        return self.register_uploaded_cover(
            content_sha256=digest,
            remote_media_id=remote_media_id,
            make_active=make_active,
        )

    def upload_cover_file(
        self,
        path: str | Path,
        *,
        content_type: str = "image/jpeg",
        make_active: bool = True,
    ) -> WeChatCoverAsset:
        file_path = Path(path)
        return self.upload_cover(
            file_path.read_bytes(),
            filename=file_path.name,
            content_type=content_type,
            make_active=make_active,
        )

    def set_active(self, asset_id: str) -> WeChatCoverAsset:
        return self.repository.set_active_wechat_cover_asset(asset_id)


__all__ = ["WeChatCoverService"]
