"""Safe local storage for user-uploaded image attachments."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from uuid import uuid4

from .models import ImageAttachment

_ALLOWED = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
}
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


class ImageAttachmentService:
    def __init__(self, repository, root: str | Path, *, max_bytes: int = 10 * 1024 * 1024):
        self.repository = repository
        self.root = Path(root)
        self.max_bytes = max_bytes

    def save(self, *, thread_id: str, filename: str, content_type: str, content: bytes) -> ImageAttachment:
        if content_type not in _ALLOWED:
            raise ValueError("image_attachment_type_unsupported")
        if not content:
            raise ValueError("image_attachment_empty")
        if len(content) > self.max_bytes:
            raise ValueError("image_attachment_too_large")
        safe_name = _SAFE_NAME.sub("-", Path(filename).name).strip(".-") or "image"
        suffix = _ALLOWED[content_type]
        if not safe_name.lower().endswith(suffix):
            safe_name += suffix
        digest = hashlib.sha256(content).hexdigest()
        attachment_id = uuid4().hex
        directory = self.root / thread_id
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{attachment_id}{suffix}"
        target.write_bytes(content)
        attachment = ImageAttachment(
            attachment_id=attachment_id,
            thread_id=thread_id,
            filename=safe_name,
            storage_path=str(target.resolve()),
            content_type=content_type,
            size_bytes=len(content),
            content_sha256=digest,
        )
        try:
            stored = self.repository.create_image_attachment(attachment)
            if stored.attachment_id != attachment.attachment_id:
                # The repository found an identical existing image. Do not
                # leave the temporary duplicate file on disk.
                target.unlink(missing_ok=True)
            return stored
        except Exception:
            target.unlink(missing_ok=True)
            raise

    def get(self, attachment_id: str) -> ImageAttachment:
        attachment = self.repository.get_image_attachment(attachment_id)
        if not Path(attachment.storage_path).is_file():
            raise ValueError("image_attachment_file_missing")
        return attachment

    def get_for_thread(self, attachment_id: str, thread_id: str) -> ImageAttachment:
        """Backward-compatible alias; attachments are now globally reusable."""

        return self.get(attachment_id)

    def delete(self, attachment_id: str) -> ImageAttachment:
        attachment = self.repository.get_image_attachment(attachment_id)
        self.repository.deactivate_wechat_cover_asset_by_hash(
            attachment.content_sha256
        )
        deleted = self.repository.delete_image_attachment(attachment_id)
        Path(deleted.storage_path).unlink(missing_ok=True)
        return deleted


__all__ = ["ImageAttachmentService"]
