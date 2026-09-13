import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from deep_research.publishing.models import (
    Article,
    ArticleStatus,
    Publication,
    PublicationChannel,
    PublicationStatus,
)
from deep_research.publishing.approvals import (
    ApprovalAction,
    ApprovalRequest,
)
from deep_research.publishing.repository import (
    DuplicateRecordError,
    IdempotencyConflictError,
    PublishingRepository,
    RecordNotFoundError,
    RepositoryClosedError,
)


SHA256 = "b" * 64
TIMESTAMP = datetime(2026, 1, 1, tzinfo=timezone.utc)


def build_article(
    *,
    article_id="article-1",
    slug="research-article",
    status=ArticleStatus.DRAFT,
    version=1,
):
    return Article(
        article_id=article_id,
        source_thread_id="thread-1",
        source_artifact_id="artifact-1",
        source_artifact_sha256=SHA256,
        title="Research article",
        slug=slug,
        markdown_content="# Article",
        excerpt="An excerpt",
        tags=["research"],
        status=status,
        version=version,
        created_at=TIMESTAMP,
        updated_at=TIMESTAMP,
        published_at=TIMESTAMP if status is ArticleStatus.PUBLISHED else None,
    )


def build_publication(
    *,
    publication_id="publication-1",
    idempotency_key="request-1",
    content_sha256=SHA256,
):
    return Publication(
        publication_id=publication_id,
        article_id="article-1",
        article_version=1,
        channel=PublicationChannel.LOCAL_STATIC_SITE,
        status=PublicationStatus.PUBLISHING,
        idempotency_key=idempotency_key,
        content_sha256=content_sha256,
        attempt_count=1,
        created_at=TIMESTAMP,
        updated_at=TIMESTAMP,
    )


class PublishingRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "publishing.sqlite"
        self.repository = PublishingRepository(self.db_path)
        self.repository.initialize()

    def tearDown(self):
        self.repository.close()
        self.temp_dir.cleanup()

    def test_initialize_is_repeatable_and_data_survives_reopen(self):
        article = build_article()
        self.repository.create_article(article)
        self.repository.initialize()
        self.repository.close()

        reopened = PublishingRepository(self.db_path)
        reopened.initialize()
        try:
            restored = reopened.get_article("article-1")
        finally:
            reopened.close()

        self.assertEqual(restored.article_id, "article-1")
        self.assertEqual(restored.markdown_content, "# Article")

    def test_approval_attachment_snapshot_survives_reopen(self):
        article = build_article()
        self.repository.create_article(article)
        approval = ApprovalRequest(
            approval_id="approval-with-images",
            action=ApprovalAction.PUBLISH,
            article_id=article.article_id,
            article_version=article.version,
            channel=PublicationChannel.LOCAL_STATIC_SITE,
            content_sha256=SHA256,
            attachment_ids=("image-a", "image-b"),
        )
        self.repository.create_approval_request(approval)
        self.repository.close()

        reopened = PublishingRepository(self.db_path)
        reopened.initialize()
        try:
            restored = reopened.get_approval_request(approval.approval_id)
        finally:
            reopened.close()

        self.assertEqual(restored.attachment_ids, ("image-a", "image-b"))

    def test_reopen_migrates_legacy_local_only_channel_constraints(self):
        article = build_article()
        self.repository.create_article(article)
        publication = build_publication()
        self.repository.create_publication(publication)
        approval = ApprovalRequest(
            approval_id="approval-legacy",
            action=ApprovalAction.PUBLISH,
            article_id=article.article_id,
            article_version=article.version,
            channel=PublicationChannel.LOCAL_STATIC_SITE,
            content_sha256=SHA256,
        )
        self.repository.create_approval_request(approval)
        self.repository.close()

        connection = sqlite3.connect(self.db_path)
        try:
            connection.execute("PRAGMA foreign_keys = OFF")
            connection.execute(
                'ALTER TABLE publications RENAME TO publications__legacy'
            )
            connection.execute("DROP INDEX idx_publications_article")
            connection.execute(
                """
                CREATE TABLE publications (
                    publication_id TEXT PRIMARY KEY,
                    article_id TEXT NOT NULL,
                    article_version INTEGER NOT NULL,
                    channel TEXT NOT NULL CHECK (
                        channel IN ('local_static_site')
                    ),
                    status TEXT NOT NULL CHECK (
                        status IN ('publishing','published','failed')
                    ),
                    idempotency_key TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    external_id TEXT,
                    public_url TEXT,
                    attempt_count INTEGER NOT NULL CHECK (attempt_count >= 0),
                    error_code TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    published_at TEXT,
                    UNIQUE (channel, idempotency_key),
                    FOREIGN KEY (article_id, article_version)
                        REFERENCES articles (article_id, version)
                )
                """
            )
            connection.execute(
                """
                INSERT INTO publications
                SELECT * FROM publications__legacy
                """
            )
            connection.execute("DROP TABLE publications__legacy")
            connection.execute(
                """
                CREATE INDEX idx_publications_article
                ON publications (article_id, article_version, created_at DESC)
                """
            )

            connection.execute(
                'ALTER TABLE approval_requests RENAME TO approval_requests__legacy'
            )
            connection.execute("DROP INDEX uq_approval_pending_target")
            connection.execute("DROP INDEX idx_approval_requests_article")
            connection.execute(
                """
                CREATE TABLE approval_requests (
                    approval_id TEXT PRIMARY KEY,
                    action TEXT NOT NULL CHECK (action IN ('publish')),
                    article_id TEXT NOT NULL,
                    article_version INTEGER NOT NULL CHECK (article_version >= 1),
                    channel TEXT NOT NULL CHECK (
                        channel IN ('local_static_site')
                    ),
                    content_sha256 TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('pending','approved','rejected','expired')
                    ),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    decided_at TEXT,
                    decision_actor TEXT,
                    decision_reason TEXT,
                    UNIQUE (approval_id),
                    FOREIGN KEY (article_id, article_version)
                        REFERENCES articles (article_id, version)
                )
                """
            )
            connection.execute(
                """
                INSERT INTO approval_requests
                SELECT * FROM approval_requests__legacy
                """
            )
            connection.execute("DROP TABLE approval_requests__legacy")
            connection.execute(
                """
                CREATE UNIQUE INDEX uq_approval_pending_target
                ON approval_requests (action, article_id, article_version, channel)
                WHERE status = 'pending'
                """
            )
            connection.execute(
                """
                CREATE INDEX idx_approval_requests_article
                ON approval_requests (article_id, created_at DESC)
                """
            )
            connection.commit()
        finally:
            connection.close()

        reopened = PublishingRepository(self.db_path)
        reopened.initialize()
        try:
            self.assertEqual(
                reopened.get_publication("publication-1").channel,
                PublicationChannel.LOCAL_STATIC_SITE,
            )
            self.assertEqual(
                reopened.get_approval_request("approval-legacy").channel,
                PublicationChannel.LOCAL_STATIC_SITE,
            )
            wechat_approval = ApprovalRequest(
                approval_id="approval-wechat",
                action=ApprovalAction.PUBLISH,
                article_id=article.article_id,
                article_version=article.version,
                channel=PublicationChannel.WECHAT_OFFICIAL_ACCOUNT,
                content_sha256=SHA256,
            )
            reopened.create_approval_request(wechat_approval)
            self.assertEqual(
                reopened.get_approval_request("approval-wechat").channel,
                PublicationChannel.WECHAT_OFFICIAL_ACCOUNT,
            )
        finally:
            reopened.close()

    def test_article_can_be_updated_and_listed(self):
        article = build_article()
        self.repository.create_article(article)

        article.title = "Updated title"
        article.updated_at = TIMESTAMP
        updated = self.repository.update_article(article)

        self.assertEqual(updated.title, "Updated title")
        self.assertEqual(self.repository.list_articles()[0].title, "Updated title")

    def test_current_slug_is_unique(self):
        self.repository.create_article(build_article())

        with self.assertRaises(DuplicateRecordError):
            self.repository.create_article(
                build_article(
                    article_id="article-2",
                    slug="research-article",
                )
            )

    def test_revision_preserves_previous_published_version(self):
        original = build_article(status=ArticleStatus.PUBLISHED)
        self.repository.create_article(original)

        revision = original.create_revision(
            title="New version",
            markdown_content="# New version",
            now=TIMESTAMP,
        )
        self.repository.create_revision(
            revision,
            previous_version=original.version,
        )

        historical = self.repository.get_article(
            "article-1",
            version=1,
        )
        current = self.repository.get_article("article-1")

        self.assertEqual(historical.status, ArticleStatus.PUBLISHED)
        self.assertEqual(historical.title, "Research article")
        self.assertEqual(current.version, 2)
        self.assertEqual(current.status, ArticleStatus.DRAFT)
        self.assertEqual(current.title, "New version")

    def test_failed_revision_does_not_hide_previous_current_version(self):
        original = build_article(status=ArticleStatus.PUBLISHED)
        self.repository.create_article(original)
        conflicting_revision = original.create_revision(now=TIMESTAMP)

        self.repository.create_article(
            build_article(
                article_id="article-2",
                slug="other-article",
            )
        )
        conflicting_revision.slug = "other-article"

        with self.assertRaises(DuplicateRecordError):
            self.repository.create_revision(
                conflicting_revision,
                previous_version=1,
            )

        current = self.repository.get_article("article-1")
        self.assertEqual(current.version, 1)
        self.assertTrue(current.status is ArticleStatus.PUBLISHED)

    def test_publication_is_idempotent_by_channel_and_key(self):
        self.repository.create_article(build_article())
        publication = build_publication()
        self.repository.create_publication(publication)

        same_request = build_publication(
            publication_id="publication-2",
        )
        with self.assertRaises(DuplicateRecordError):
            self.repository.create_publication(same_request)

        conflicting_request = build_publication(
            publication_id="publication-3",
            content_sha256="c" * 64,
        )
        with self.assertRaises(IdempotencyConflictError):
            self.repository.create_publication(conflicting_request)

        restored = self.repository.get_publication_by_idempotency_key(
            channel=PublicationChannel.LOCAL_STATIC_SITE,
            idempotency_key="request-1",
        )
        self.assertIsNotNone(restored)
        self.assertEqual(restored.publication_id, "publication-1")

    def test_publication_requires_existing_article_version(self):
        with self.assertRaises(DuplicateRecordError):
            self.repository.create_publication(build_publication())

    def test_closed_repository_rejects_operations(self):
        self.repository.close()

        with self.assertRaises(RepositoryClosedError):
            self.repository.list_articles()

        with self.assertRaises(RepositoryClosedError):
            self.repository.get_article("article-1")

    def test_missing_article_is_reported(self):
        with self.assertRaises(RecordNotFoundError):
            self.repository.get_article("missing")


if __name__ == "__main__":
    unittest.main()
