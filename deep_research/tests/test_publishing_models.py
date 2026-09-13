import unittest
from datetime import datetime, timezone

from deep_research.publishing.models import (
    Article,
    ArticleStatus,
    ArtifactSnapshot,
    InvalidDomainValueError,
    InvalidStateTransitionError,
    Publication,
    PublicationChannel,
    PublicationStatus,
)


SHA256 = "a" * 64
TIMESTAMP = datetime(2026, 1, 1, tzinfo=timezone.utc)


def build_article(
    *,
    status: ArticleStatus = ArticleStatus.DRAFT,
    version: int = 1,
) -> Article:
    return Article(
        article_id="article-1",
        source_thread_id="thread-1",
        source_artifact_id="artifact-1",
        source_artifact_sha256=SHA256,
        title="Research article",
        slug="research-article",
        markdown_content="# Article",
        excerpt="An excerpt",
        tags=["research"],
        status=status,
        version=version,
        created_at=TIMESTAMP,
        updated_at=TIMESTAMP,
    )


class ArticleModelTests(unittest.TestCase):
    def test_draft_can_be_edited(self):
        article = build_article()

        article.edit(
            title="Updated title",
            slug="updated-title",
            markdown_content="# Updated",
            excerpt="Updated excerpt",
            tags=["one", "two"],
            now=TIMESTAMP,
        )

        self.assertEqual(article.title, "Updated title")
        self.assertEqual(article.slug, "updated-title")
        self.assertEqual(article.markdown_content, "# Updated")
        self.assertEqual(article.tags, ["one", "two"])

    def test_article_state_machine_accepts_valid_transitions(self):
        article = build_article()

        article.approve(now=TIMESTAMP)
        self.assertEqual(article.status, ArticleStatus.APPROVED)
        self.assertEqual(article.approved_at, TIMESTAMP)

        article.start_publishing(now=TIMESTAMP)
        self.assertEqual(article.status, ArticleStatus.PUBLISHING)

        article.mark_published(now=TIMESTAMP)
        self.assertEqual(article.status, ArticleStatus.PUBLISHED)
        self.assertEqual(article.published_at, TIMESTAMP)

    def test_failed_article_can_be_retried(self):
        article = build_article()
        article.approve(now=TIMESTAMP)
        article.start_publishing(now=TIMESTAMP)
        article.mark_failed(now=TIMESTAMP)

        self.assertEqual(article.status, ArticleStatus.FAILED)

        article.start_publishing(now=TIMESTAMP)
        self.assertEqual(article.status, ArticleStatus.PUBLISHING)

    def test_invalid_article_transitions_are_rejected(self):
        article = build_article()

        with self.assertRaises(InvalidStateTransitionError):
            article.start_publishing(now=TIMESTAMP)

        article.approve(now=TIMESTAMP)
        with self.assertRaises(InvalidStateTransitionError):
            article.edit(title="must fail")

        article.start_publishing(now=TIMESTAMP)
        with self.assertRaises(InvalidStateTransitionError):
            article.approve(now=TIMESTAMP)

    def test_published_article_is_not_modified_when_creating_revision(self):
        article = build_article(status=ArticleStatus.PUBLISHED)
        article.published_at = TIMESTAMP

        revision = article.create_revision(
            title="Revision",
            markdown_content="# Revision",
            now=TIMESTAMP,
        )

        self.assertEqual(article.status, ArticleStatus.PUBLISHED)
        self.assertEqual(article.version, 1)
        self.assertEqual(article.title, "Research article")
        self.assertEqual(revision.status, ArticleStatus.DRAFT)
        self.assertEqual(revision.version, 2)
        self.assertEqual(revision.title, "Revision")

    def test_published_article_cannot_be_edited_directly(self):
        article = build_article(status=ArticleStatus.PUBLISHED)

        with self.assertRaises(InvalidStateTransitionError):
            article.edit(title="must not overwrite published version")

    def test_artifact_snapshot_requires_valid_hash_and_size(self):
        snapshot = ArtifactSnapshot(
            source_thread_id="thread-1",
            source_artifact_id="artifact-1",
            workspace_path="/final/report.md",
            filename="report.md",
            size_bytes=len("# Report".encode("utf-8")),
            sha256=SHA256,
            markdown_content="# Report",
        )
        self.assertEqual(snapshot.source_thread_id, "thread-1")

        with self.assertRaises(InvalidDomainValueError):
            ArtifactSnapshot(
                source_thread_id="thread-1",
                source_artifact_id="artifact-1",
                workspace_path="/final/report.md",
                filename="report.md",
                size_bytes=-1,
                sha256=SHA256,
                markdown_content="# Report",
            )

        with self.assertRaises(InvalidDomainValueError):
            ArtifactSnapshot(
                source_thread_id="thread-1",
                source_artifact_id="artifact-1",
                workspace_path="/final/report.md",
                filename="report.md",
                size_bytes=8,
                sha256="not-a-sha256",
                markdown_content="# Report",
            )

    def test_publication_state_machine(self):
        publication = Publication(
            publication_id="publication-1",
            article_id="article-1",
            article_version=1,
            channel=PublicationChannel.LOCAL_STATIC_SITE,
            status=PublicationStatus.PUBLISHING,
            idempotency_key="request-1",
            content_sha256=SHA256,
            attempt_count=1,
            created_at=TIMESTAMP,
            updated_at=TIMESTAMP,
        )

        publication.mark_published(
            public_url="/articles/research-article/",
            now=TIMESTAMP,
        )

        self.assertEqual(publication.status, PublicationStatus.PUBLISHED)
        self.assertEqual(
            publication.public_url,
            "/articles/research-article/",
        )

        with self.assertRaises(InvalidStateTransitionError):
            publication.retry(now=TIMESTAMP)

    def test_delivery_unknown_requires_explicit_retry_transition(self):
        publication = Publication(
            publication_id="publication-unknown",
            article_id="article-1",
            article_version=1,
            channel=PublicationChannel.WECHAT_OFFICIAL_ACCOUNT,
            status=PublicationStatus.PUBLISHING,
            idempotency_key="request-unknown",
            content_sha256=SHA256,
            attempt_count=1,
            created_at=TIMESTAMP,
            updated_at=TIMESTAMP,
        )

        publication.mark_delivery_unknown(
            error_code="wechat_publish_request_failed",
            now=TIMESTAMP,
        )
        publication.retry_delivery_unknown(now=TIMESTAMP)

        self.assertEqual(publication.status, PublicationStatus.PUBLISHING)
        self.assertEqual(publication.attempt_count, 2)
        self.assertIsNone(publication.error_code)


if __name__ == "__main__":
    unittest.main()
