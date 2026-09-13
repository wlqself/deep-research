"""Local, deterministic publisher for Markdown article snapshots."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path

from ..inline_images import (
    extract_inline_image_references,
    replace_inline_attachment_urls,
)
from ..models import Article, ArticleStatus, PublicationChannel
from ..service import PublicationResult, validate_slug

# 目录来自可信的应用配置（而非 HTTP 请求），版本化的输出路径确保同一 slug 的新版本不会覆盖旧版本；
class LocalStaticPublisher:
    """Write one publishing version into the configured local site directory.

    The directory is supplied by trusted application configuration, never by an
    HTTP request. Versioned output paths preserve previously published content
    when a later Article revision uses the same slug.
    """

    channel = PublicationChannel.LOCAL_STATIC_SITE

    # 初始化站点目录，类型转换和空值校验
    def __init__(
        self,
        site_dir: str | Path,
        *,
        attachment_resolver: Callable[
            [Article, tuple[str, ...]], list[tuple[str, str]]
        ] | None = None,
    ) -> None:
        if isinstance(site_dir, Path):
            configured_dir = site_dir
        elif isinstance(site_dir, str) and site_dir.strip():
            configured_dir = Path(site_dir)
        else:
            raise ValueError("site_dir must not be empty")

        self._site_dir = configured_dir
        self._attachment_resolver = attachment_resolver
    # 核心发布逻辑
    def publish(self, article: Article) -> PublicationResult:
        """Atomically write the approved Article version as Markdown."""
        # 文章状态必须是 PUBLISHING
        if article.status is not ArticleStatus.PUBLISHING:
            raise ValueError("article must be in publishing state")
        # 对文章的 slug 再做一次校验
        safe_slug = validate_slug(article.slug)
        # 构造版本化的相对路径
        relative_path = Path(
            "articles",
            safe_slug,
            f"v{article.version}",
            "index.md",
        )
        target_dir = self._site_dir / relative_path.parent
        target_path = self._site_dir / relative_path
        # 路径遍历防护，防止恶意 slug 写入站点目录之外的文件系统位置
        root = self._site_dir.resolve()
        resolved_target = target_path.resolve()
        try:
            resolved_target.relative_to(root)
        except ValueError as error:
            raise ValueError("article output path is outside site directory") from error

        target_dir.mkdir(parents=True, exist_ok=True)
        content = article.markdown_content
        inline_references = extract_inline_image_references(content)
        if inline_references:
            inline_ids = tuple(dict.fromkeys(
                reference.attachment_id for reference in inline_references
            ))
            if self._attachment_resolver is None:
                raise ValueError("local inline image attachments are not configured")
            attachments = self._attachment_resolver(article, inline_ids)
            if len(attachments) != len(inline_ids):
                raise ValueError("local inline image attachment does not exist")
            image_dir = target_dir / "images"
            image_dir.mkdir(parents=True, exist_ok=True)
            urls_by_id: dict[str, str] = {}
            for attachment_id, (image_path, _content_type) in zip(
                inline_ids,
                attachments,
                strict=True,
            ):
                source = Path(image_path)
                if not source.is_file():
                    raise ValueError("local inline image file does not exist")
                suffix = source.suffix.lower() or ".img"
                target_name = f"{attachment_id}{suffix}"
                target_image = image_dir / target_name
                target_image.write_bytes(source.read_bytes())
                urls_by_id[attachment_id] = f"images/{target_name}"
            content = replace_inline_attachment_urls(content, urls_by_id)

        #在目标目录内创建一个临时文件，这里是为了进行原子替换
        temporary_path: Path | None = None
        try:
            file_descriptor, temporary_name = tempfile.mkstemp(
                dir=target_dir,
                prefix=".index-",
                suffix=".tmp",
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(
                file_descriptor,
                mode="w",
                encoding="utf-8",
                newline="",
            ) as temporary_file:
                temporary_file.write(content)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            # 用 os.replace（POSIX 的 rename 系统调用）将临时文件原子性地替换为目标文件
            os.replace(temporary_path, target_path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

        return PublicationResult(
            public_url="/" + relative_path.as_posix(),
            external_id=(
                f"local_static_site:{article.article_id}:v{article.version}"
            ),
        )

