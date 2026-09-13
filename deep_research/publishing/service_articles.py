"""Article drafting and editorial workflow operations."""

from __future__ import annotations

import logging
from uuid import uuid4

from .approvals import ApprovalAction
from .artifacts import (
    ArtifactIntegrityError,
    ArtifactNotFoundError,
)
from .formatting import normalize_article_markdown
from .models import (
    Article,
    ArticleStatus,
    InvalidStateTransitionError,
    PublicationChannel,
    PublicationStatus,
)
from .repository import DuplicateRecordError
from .service_support import (
    ArticleImportConflictError,
    normalized_tags,
    excerpt_from_markdown,
    slugify_title,
    title_from_filename,
    validate_slug,
)


class ArticleWorkflowMixin:
    """Implement Article creation, editing, and editorial approval."""

    async def create_article_from_artifact(
        self,
        *,
        thread_id: str,
        artifact_id: str,
        title: str | None = None,
        slug: str | None = None,
        excerpt: str | None = None,
        tags: list[str] | None = None,
    ) -> Article:
        """Copy one verified Artifact into an independent Article draft."""

        try:
            snapshot = await self.artifact_service.read(
                thread_id=thread_id,
                artifact_id=artifact_id,
            )
        except ArtifactNotFoundError:
            self._record_event(
                event="publishing.article.import_failed",
                status="failed",
                thread_id=thread_id,
                artifact_id=artifact_id,
                error_code="artifact_not_found",
                level=logging.WARNING,
            )
            raise
        except ArtifactIntegrityError:
            self._record_event(
                event="publishing.article.import_failed",
                status="failed",
                thread_id=thread_id,
                artifact_id=artifact_id,
                error_code="artifact_integrity_failed",
                level=logging.WARNING,
            )
            raise

        existing = next(
            (
                article
                for article in self.repository.list_articles()
                if article.source_thread_id == thread_id
                and article.source_artifact_id == artifact_id
            ),
            None,
        )
        if existing is not None:
            self._record_event(
                event="publishing.article.import_replayed",
                status="replayed",
                article=existing,
            )
            return existing

        normalized_title = (
            title or title_from_filename(snapshot.filename)
        ).strip()
        if not normalized_title:
            raise ArticleImportConflictError("article title must not be empty")

        requested_slug = (
            slugify_title(normalized_title)
            if slug is None
            else validate_slug(slug)
        )
        unique_slug = self._unique_slug(requested_slug)
        normalized_excerpt = (
            excerpt_from_markdown(snapshot.markdown_content)
            if excerpt is None
            else excerpt.strip()
        )
        article = Article(
            article_id=uuid4().hex,
            source_thread_id=thread_id,
            source_artifact_id=artifact_id,
            source_artifact_sha256=snapshot.sha256,
            title=normalized_title,
            slug=unique_slug,
            markdown_content=normalize_article_markdown(
                snapshot.markdown_content
            ),
            excerpt=normalized_excerpt,
            tags=normalized_tags(tags),
        )

        try:
            created = self.repository.create_article(article)
        except DuplicateRecordError as error:
            existing = next(
                (
                    candidate
                    for candidate in self.repository.list_articles()
                    if candidate.source_thread_id == thread_id
                    and candidate.source_artifact_id == artifact_id
                ),
                None,
            )
            if existing is not None:
                self._record_event(
                    event="publishing.article.import_replayed",
                    status="replayed",
                    article=existing,
                )
                return existing
            self._record_event(
                event="publishing.article.import_failed",
                status="failed",
                article=article,
                error_code="article_import_conflict",
                level=logging.WARNING,
            )
            raise ArticleImportConflictError(
                "article could not be created with a unique slug"
            ) from error

        self._record_event(
            event="publishing.article.imported",
            status="completed",
            article=created,
        )
        return created

    def list_articles(self) -> list[Article]:
        """List current Article records."""

        return self.repository.list_articles()

    def get_article(self, article_id: str) -> Article:
        """Get one current Article record."""

        return self.repository.get_article(article_id)

    def edit_article(
        self,
        article_id: str,
        *,
        title: str | None = None,
        slug: str | None = None,
        markdown_content: str | None = None,
        excerpt: str | None = None,
        tags: list[str] | None = None,
    ) -> Article:
        """Edit a draft Article through the domain state guard."""

        article = self.repository.get_article(article_id)
        if markdown_content is not None:
            markdown_content = normalize_article_markdown(markdown_content)
        if slug is not None:
            slug = validate_slug(slug)
            if slug != article.slug:
                slug = self._unique_slug(
                    slug,
                    excluding_article_id=article_id,
                )

        try:
            article.edit(
                title=title,
                slug=slug,
                markdown_content=markdown_content,
                excerpt=excerpt,
                tags=normalized_tags(tags) if tags is not None else None,
            )
            updated = self.repository.update_article(article)
        except InvalidStateTransitionError:
            self._record_event(
                event="publishing.article.edit_failed",
                status="failed",
                article=article,
                error_code="article_not_editable",
                level=logging.WARNING,
            )
            raise
        except Exception:
            self._record_event(
                event="publishing.article.edit_failed",
                status="failed",
                article=article,
                error_code="article_edit_failed",
                level=logging.WARNING,
            )
            raise

        self._record_event(
            event="publishing.article.edited",
            status="completed",
            article=updated,
        )
        return updated

    def revise_article(
        self,
        article_id: str,
        *,
        title: str | None = None,
        slug: str | None = None,
        markdown_content: str | None = None,
        excerpt: str | None = None,
        tags: list[str] | None = None,
        change_reason: str | None = None,
    ) -> Article:
        """Edit a draft or create a new draft revision for a reviewed article.

        Drafts are edited through the existing guarded editor.  Approved and
        published versions are immutable snapshots, so they first produce a
        new draft revision and leave all earlier approval/publication records
        attached to their original version.
        """

        article = self.repository.get_article(article_id)
        if markdown_content is not None:
            markdown_content = normalize_article_markdown(markdown_content)
        if article.status is ArticleStatus.DRAFT:
            return self.edit_article(
                article_id,
                title=title,
                slug=slug,
                markdown_content=markdown_content,
                excerpt=excerpt,
                tags=tags,
            )

        if article.status not in {
            ArticleStatus.APPROVED,
            ArticleStatus.PUBLISHED,
            ArticleStatus.FAILED,
            ArticleStatus.PUBLISHING,
        }:
            self._record_event(
                event="publishing.article.revision_failed",
                status="failed",
                article=article,
                error_code="article_not_revisable",
                level=logging.WARNING,
            )
            raise InvalidStateTransitionError(
                "only draft, approved, published, or failed articles can be revised"
            )

        if article.status is ArticleStatus.PUBLISHING:
            # The Article status is shared by the version, while publication
            # status is per channel.  A timed-out/ambiguous channel may be
            # revised, but an actually running publication must finish first
            # so its immutable content snapshot cannot be changed underneath
            # it.
            in_flight = any(
                publication.article_version == article.version
                and publication.status is PublicationStatus.PUBLISHING
                for publication in self.repository.list_publications(
                    article.article_id,
                )
            )
            if in_flight:
                self._record_event(
                    event="publishing.article.revision_failed",
                    status="failed",
                    article=article,
                    error_code="article_revision_in_flight",
                    level=logging.WARNING,
                )
                raise InvalidStateTransitionError(
                    "the current article version still has a publication in flight"
                )

        if article.status is ArticleStatus.APPROVED:
            pending_approval = self.repository.get_pending_approval_request(
                action=ApprovalAction.PUBLISH,
                article_id=article.article_id,
                article_version=article.version,
                channel=PublicationChannel.LOCAL_STATIC_SITE,
            )
            if pending_approval is not None:
                self.reject_publication_request(
                    pending_approval.approval_id,
                    decision_actor="user",
                    decision_reason=change_reason,
                )

        normalized_slug = None
        if slug is not None:
            normalized_slug = validate_slug(slug)
            if normalized_slug != article.slug:
                normalized_slug = self._unique_slug(
                    normalized_slug,
                    excluding_article_id=article_id,
                )

        revision = article.create_revision(
            title=title,
            slug=normalized_slug,
            markdown_content=markdown_content,
            excerpt=excerpt,
            tags=normalized_tags(tags) if tags is not None else None,
        )
        try:
            created = self.repository.create_revision(
                revision,
                previous_version=article.version,
            )
        except Exception:
            self._record_event(
                event="publishing.article.revision_failed",
                status="failed",
                article=article,
                error_code="article_revision_failed",
                level=logging.WARNING,
            )
            raise

        self._record_event(
            event="publishing.article.revision_created",
            status="completed",
            article=created,
            actor="agent",
            summary="文章已创建新的待修改版本",
        )
        return created

    def approve_article(self, article_id: str) -> Article:
        """Approve the editorial Article draft."""

        article = self.repository.get_article(article_id)
        try:
            article.approve()
            approved = self.repository.update_article(article)
        except InvalidStateTransitionError:
            self._record_event(
                event="publishing.article.approve_failed",
                status="failed",
                article=article,
                error_code="article_not_approvable",
                level=logging.WARNING,
            )
            raise

        self._record_event(
            event="publishing.article.approved",
            status="completed",
            article=approved,
            actor="user",
            summary="用户批准发布文章",
        )
        return approved

    def _unique_slug(
        self,
        requested_slug: str,
        *,
        excluding_article_id: str | None = None,
    ) -> str:
        """Return a unique current Article slug."""

        existing_slugs = {
            article.slug
            for article in self.repository.list_articles()
            if article.article_id != excluding_article_id
        }
        if requested_slug not in existing_slugs:
            return requested_slug

        suffix = 2
        while True:
            suffix_text = f"-{suffix}"
            base = requested_slug[: 120 - len(suffix_text)].rstrip("-")
            candidate = f"{base}{suffix_text}"
            if candidate not in existing_slugs:
                return candidate
            suffix += 1


__all__ = ["ArticleWorkflowMixin"]
