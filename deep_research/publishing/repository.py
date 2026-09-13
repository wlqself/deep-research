"""SQLite repository for publishing records.

The repository is intentionally independent from LangGraph state.  It owns
only publishing records and never stores credentials or channel tokens.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from .approvals import (
    ApprovalAction,
    ApprovalRequest,
    ApprovalStatus,
)
from .models import (
    Article,
    ArticleStatus,
    Publication,
    PublicationChannel,
    PublicationStatus,
    WeChatCoverAsset,
    ImageAttachment,
    ImageAttachmentAnalysis,
    utc_now,
)

# 仓库异常体系
class PublishingRepositoryError(Exception):
    """Base exception for repository failures."""


class RepositoryClosedError(PublishingRepositoryError):
    """Raised when an operation is attempted after repository close."""


class DuplicateRecordError(PublishingRepositoryError):
    """Raised when a record would violate a repository uniqueness rule."""


class RecordNotFoundError(PublishingRepositoryError):
    """Raised when a requested publishing record does not exist."""


class IdempotencyConflictError(PublishingRepositoryError):
    """Raised when an idempotency key is reused for a different request."""

# 枚举值元组
_ARTICLE_STATUS_VALUES = tuple(status.value for status in ArticleStatus)
_PUBLICATION_STATUS_VALUES = tuple(
    status.value for status in PublicationStatus
)
_PUBLICATION_CHANNEL_VALUES = tuple(
    channel.value for channel in PublicationChannel
)
_APPROVAL_ACTION_VALUES = tuple(
    action.value for action in ApprovalAction
)
_APPROVAL_STATUS_VALUES = tuple(
    status.value for status in ApprovalStatus
)

# 时间戳序列化，把 datetime 转为 ISO 8601 字符串，便于存入 SQLite
def _timestamp_to_storage(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None

# 把存储的 ISO 字符串解析回 datetime
def _timestamp_from_storage(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value is not None else None

# 把 Article 领域对象的所有字段转换为数据库 INSERT/UPDATE 语句所需的参数元组；
def _article_to_parameters(article: Article) -> tuple[object, ...]:
    return (
        article.article_id,
        article.version,
        article.source_thread_id,
        article.source_artifact_id,
        article.source_artifact_sha256,
        article.title,
        article.slug,
        article.markdown_content,
        article.excerpt,
        json.dumps(article.tags, ensure_ascii=False),
        article.status.value,
        _timestamp_to_storage(article.created_at),
        _timestamp_to_storage(article.updated_at),
        _timestamp_to_storage(article.approved_at),
        _timestamp_to_storage(article.published_at),
    )

# 这是一个 ORM 风格的映射函数：把 SQLite 查询返回的一行记录（sqlite3.Row）还原为 Article 领域对象
def _row_to_article(row: sqlite3.Row) -> Article:
    return Article(
        article_id=row["article_id"],
        source_thread_id=row["source_thread_id"],
        source_artifact_id=row["source_artifact_id"],
        source_artifact_sha256=row["source_artifact_sha256"],
        title=row["title"],
        slug=row["slug"],
        markdown_content=row["markdown_content"],
        excerpt=row["excerpt"],
        tags=json.loads(row["tags_json"]),
        status=ArticleStatus(row["status"]),
        version=row["version"],
        created_at=_timestamp_from_storage(row["created_at"]),
        updated_at=_timestamp_from_storage(row["updated_at"]),
        approved_at=_timestamp_from_storage(row["approved_at"]),
        published_at=_timestamp_from_storage(row["published_at"]),
    )

# 把 Publication 领域对象序列化为数据库 INSERT/UPDATE 所需的参数元组；
def _publication_to_parameters(
    publication: Publication,
) -> tuple[object, ...]:
    return (
        publication.publication_id,
        publication.article_id,
        publication.article_version,
        publication.channel.value,
        publication.status.value,
        publication.idempotency_key,
        publication.content_sha256,
        publication.external_id,
        publication.public_url,
        publication.attempt_count,
        publication.error_code,
        _timestamp_to_storage(publication.created_at),
        _timestamp_to_storage(publication.updated_at),
        _timestamp_to_storage(publication.published_at),
    )

# 把 SQLite 查询返回的 sqlite3.Row 还原为 Publication 领域对象。
def _row_to_publication(
    row: sqlite3.Row,
    *,
    attachment_ids: tuple[str, ...] = (),
) -> Publication:
    return Publication(
        publication_id=row["publication_id"],
        article_id=row["article_id"],
        article_version=row["article_version"],
        channel=PublicationChannel(row["channel"]),
        status=PublicationStatus(row["status"]),
        idempotency_key=row["idempotency_key"],
        content_sha256=row["content_sha256"],
        external_id=row["external_id"],
        public_url=row["public_url"],
        attachment_ids=attachment_ids,
        attempt_count=row["attempt_count"],
        error_code=row["error_code"],
        created_at=_timestamp_from_storage(row["created_at"]),
        updated_at=_timestamp_from_storage(row["updated_at"]),
        published_at=_timestamp_from_storage(row["published_at"]),
    )


def _row_to_wechat_cover_asset(row: sqlite3.Row) -> WeChatCoverAsset:
    return WeChatCoverAsset(
        asset_id=row["asset_id"],
        content_sha256=row["content_sha256"],
        remote_media_id=row["remote_media_id"],
        is_active=bool(row["is_active"]),
        created_at=_timestamp_from_storage(row["created_at"]),
        last_verified_at=_timestamp_from_storage(row["last_verified_at"]),
    )


def _row_to_image_attachment(row: sqlite3.Row) -> ImageAttachment:
    return ImageAttachment(
        attachment_id=row["attachment_id"],
        thread_id=row["thread_id"],
        filename=row["filename"],
        storage_path=row["storage_path"],
        content_type=row["content_type"],
        size_bytes=row["size_bytes"],
        content_sha256=row["content_sha256"],
        created_at=_timestamp_from_storage(row["created_at"]),
    )


def _row_to_image_attachment_analysis(
    row: sqlite3.Row,
) -> ImageAttachmentAnalysis:
    return ImageAttachmentAnalysis(
        analysis_id=row["analysis_id"],
        attachment_id=row["attachment_id"],
        status=row["status"],
        image_type=row["image_type"],
        confidence=row["confidence"],
        summary=row["summary"],
        ocr_text=row["ocr_text"],
        structured_result=row["structured_result"],
        analysis_model=row["analysis_model"],
        analysis_version=row["analysis_version"],
        error_code=row["error_code"],
        created_at=_timestamp_from_storage(row["created_at"]) or utc_now(),
        updated_at=_timestamp_from_storage(row["updated_at"]) or utc_now(),
    )


def _approval_to_parameters(
    request: ApprovalRequest,
) -> tuple[object, ...]:
    return (
        request.approval_id,
        request.action.value,
        request.article_id,
        request.article_version,
        request.channel.value,
        request.content_sha256,
        request.status.value,
        _timestamp_to_storage(request.created_at),
        _timestamp_to_storage(request.updated_at),
        _timestamp_to_storage(request.decided_at),
        request.decision_actor,
        request.decision_reason,
    )


def _row_to_approval(
    row: sqlite3.Row,
    *,
    attachment_ids: tuple[str, ...] = (),
) -> ApprovalRequest:
    return ApprovalRequest(
        approval_id=row["approval_id"],
        action=ApprovalAction(row["action"]),
        article_id=row["article_id"],
        article_version=row["article_version"],
        channel=PublicationChannel(row["channel"]),
        content_sha256=row["content_sha256"],
        attachment_ids=attachment_ids,
        status=ApprovalStatus(row["status"]),
        created_at=_timestamp_from_storage(row["created_at"]),
        updated_at=_timestamp_from_storage(row["updated_at"]),
        decided_at=_timestamp_from_storage(row["decided_at"]),
        decision_actor=row["decision_actor"],
        decision_reason=row["decision_reason"],
    )


def _channel_constraint_needs_migration(
    connection: sqlite3.Connection,
    table_name: str,
) -> bool:
    row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    if row is None or not isinstance(row[0], str):
        return False
    required = ("wechat_official_account", "xiaohongshu", "douyin")
    if table_name == "publications":
        required += ("drafted", "delivery_unknown")
    return any(token not in row[0] for token in required)


def _drop_indexes_for_table(
    connection: sqlite3.Connection,
    table_name: str,
) -> None:
    index_rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = ?",
        (table_name,),
    ).fetchall()
    for row in index_rows:
        index_name = row[0]
        if not isinstance(index_name, str) or index_name.startswith(
            "sqlite_autoindex"
        ):
            continue
        quoted_name = index_name.replace('"', '""')
        connection.execute(f'DROP INDEX "{quoted_name}"')


def _migrate_channel_constraints(
    connection: sqlite3.Connection,
) -> None:
    """Rebuild legacy channel tables whose CHECK omitted WeChat."""

    channel_values = ",".join(
        repr(value) for value in _PUBLICATION_CHANNEL_VALUES
    )
    table_specs = {
        "publications": {
            "columns": f"""
                publication_id TEXT PRIMARY KEY,
                article_id TEXT NOT NULL,
                article_version INTEGER NOT NULL,
                channel TEXT NOT NULL CHECK (channel IN ({channel_values})),
                status TEXT NOT NULL CHECK (
                    status IN ({','.join(repr(value) for value in _PUBLICATION_STATUS_VALUES)})
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
            """,
            "indexes": (
                "CREATE INDEX IF NOT EXISTS idx_publications_article "
                "ON publications (article_id, article_version, created_at DESC)",
            ),
        },
        "approval_requests": {
            "columns": f"""
                approval_id TEXT PRIMARY KEY,
                action TEXT NOT NULL CHECK (
                    action IN ({','.join(repr(value) for value in _APPROVAL_ACTION_VALUES)})
                ),
                article_id TEXT NOT NULL,
                article_version INTEGER NOT NULL CHECK (article_version >= 1),
                channel TEXT NOT NULL CHECK (channel IN ({channel_values})),
                content_sha256 TEXT NOT NULL,
                status TEXT NOT NULL CHECK (
                    status IN ({','.join(repr(value) for value in _APPROVAL_STATUS_VALUES)})
                ),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                decided_at TEXT,
                decision_actor TEXT,
                decision_reason TEXT,
                UNIQUE (approval_id),
                FOREIGN KEY (article_id, article_version)
                    REFERENCES articles (article_id, version)
            """,
            "indexes": (
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_approval_pending_target "
                "ON approval_requests (action, article_id, article_version, channel) "
                "WHERE status = 'pending'",
                "CREATE INDEX IF NOT EXISTS idx_approval_requests_article "
                "ON approval_requests (article_id, created_at DESC)",
            ),
        },
    }

    for table_name, spec in table_specs.items():
        if not _channel_constraint_needs_migration(
            connection,
            table_name,
        ):
            continue

        legacy_name = f"{table_name}__legacy_channels"
        connection.execute(
            f'ALTER TABLE "{table_name}" RENAME TO "{legacy_name}"'
        )
        _drop_indexes_for_table(connection, legacy_name)
        connection.execute(
            f'CREATE TABLE "{table_name}" ({spec["columns"]})'
        )

        columns = (
            "publication_id, article_id, article_version, channel, status, "
            "idempotency_key, content_sha256, external_id, public_url, "
            "attempt_count, error_code, created_at, updated_at, published_at"
            if table_name == "publications"
            else
            "approval_id, action, article_id, article_version, channel, "
            "content_sha256, status, created_at, updated_at, decided_at, "
            "decision_actor, decision_reason"
        )
        connection.execute(
            f'INSERT INTO "{table_name}" ({columns}) '
            f'SELECT {columns} FROM "{legacy_name}"'
        )
        connection.execute(f'DROP TABLE "{legacy_name}"')
        for index_sql in spec["indexes"]:
            connection.execute(index_sql)

# 发布领域的持久化层核心，负责 Article 和 Publication 的 CRUD 操作
class PublishingRepository:
    """Thread-safe, explicitly opened SQLite repository."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._connection: sqlite3.Connection | None = None
        self._lock = threading.RLock()
    # 幂等初始化
    def initialize(self) -> None:
        """Open the database and create the schema if it is absent."""
        # 加锁保证线程安全，如果 _connection 已经存在，说明已经初始化过了，直接返回（幂等操作）
        with self._lock:
            if self._connection is not None:
                return

            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(
                self.db_path,
                timeout=30,
                check_same_thread=False, # 允许连接在不同线程间使用（配合 _lock 保证安全）；
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            # 建表（articles），用 executescript 一次性执行所有 DDL
            try:
                connection.executescript(
                    f"""
                    CREATE TABLE IF NOT EXISTS articles (
                        article_id TEXT NOT NULL,
                        version INTEGER NOT NULL CHECK (version >= 1),
                        source_thread_id TEXT NOT NULL,
                        source_artifact_id TEXT NOT NULL,
                        source_artifact_sha256 TEXT NOT NULL,
                        title TEXT NOT NULL,
                        slug TEXT NOT NULL,
                        markdown_content TEXT NOT NULL,
                        excerpt TEXT NOT NULL,
                        tags_json TEXT NOT NULL,
                        status TEXT NOT NULL CHECK (
                            status IN ({','.join(repr(value) for value in _ARTICLE_STATUS_VALUES)})
                        ),
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        approved_at TEXT,
                        published_at TEXT,
                        is_current INTEGER NOT NULL DEFAULT 1 CHECK (
                            is_current IN (0, 1)
                        ),
                        PRIMARY KEY (article_id, version)
                    );

                    CREATE UNIQUE INDEX IF NOT EXISTS
                        uq_articles_current_slug
                    ON articles (slug)
                    WHERE is_current = 1;

                    CREATE TABLE IF NOT EXISTS publications (
                        publication_id TEXT PRIMARY KEY,
                        article_id TEXT NOT NULL,
                        article_version INTEGER NOT NULL,
                        channel TEXT NOT NULL CHECK (
                            channel IN ({','.join(repr(value) for value in _PUBLICATION_CHANNEL_VALUES)})
                        ),
                        status TEXT NOT NULL CHECK (
                            status IN ({','.join(repr(value) for value in _PUBLICATION_STATUS_VALUES)})
                        ),
                        idempotency_key TEXT NOT NULL,
                        content_sha256 TEXT NOT NULL,
                        external_id TEXT,
                        public_url TEXT,
                        attempt_count INTEGER NOT NULL CHECK (
                            attempt_count >= 0
                        ),
                        error_code TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        published_at TEXT,
                        UNIQUE (channel, idempotency_key),
                        FOREIGN KEY (article_id, article_version)
                            REFERENCES articles (article_id, version)
                    );

                    CREATE INDEX IF NOT EXISTS
                        idx_articles_current_updated_at
                    ON articles (is_current, updated_at DESC);

                    CREATE INDEX IF NOT EXISTS
                        idx_publications_article
                    ON publications (article_id, article_version, created_at DESC);

                    CREATE TABLE IF NOT EXISTS publication_attachments (
                        publication_id TEXT NOT NULL,
                        attachment_id TEXT NOT NULL,
                        position INTEGER NOT NULL CHECK (position >= 0),
                        PRIMARY KEY (publication_id, attachment_id),
                        UNIQUE (publication_id, position)
                    );

                    CREATE TABLE IF NOT EXISTS approval_requests (
                        approval_id TEXT PRIMARY KEY,
                        action TEXT NOT NULL CHECK (
                            action IN ({','.join(repr(value) for value in _APPROVAL_ACTION_VALUES)})
                        ),
                        article_id TEXT NOT NULL,
                        article_version INTEGER NOT NULL CHECK (
                            article_version >= 1
                        ),
                        channel TEXT NOT NULL CHECK (
                            channel IN ({','.join(repr(value) for value in _PUBLICATION_CHANNEL_VALUES)})
                        ),
                        content_sha256 TEXT NOT NULL,
                        status TEXT NOT NULL CHECK (
                            status IN ({','.join(repr(value) for value in _APPROVAL_STATUS_VALUES)})
                        ),
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        decided_at TEXT,
                        decision_actor TEXT,
                        decision_reason TEXT,
                        UNIQUE (approval_id),
                        FOREIGN KEY (article_id, article_version)
                            REFERENCES articles (article_id, version)
                    );

                    CREATE UNIQUE INDEX IF NOT EXISTS
                        uq_approval_pending_target
                    ON approval_requests (
                        action,
                        article_id,
                        article_version,
                        channel
                    )
                    WHERE status = 'pending';

                    CREATE INDEX IF NOT EXISTS
                        idx_approval_requests_article
                    ON approval_requests (article_id, created_at DESC);

                    CREATE TABLE IF NOT EXISTS approval_request_attachments (
                        approval_id TEXT NOT NULL,
                        attachment_id TEXT NOT NULL,
                        position INTEGER NOT NULL CHECK (position >= 0),
                        PRIMARY KEY (approval_id, attachment_id),
                        UNIQUE (approval_id, position),
                        FOREIGN KEY (approval_id)
                            REFERENCES approval_requests (approval_id)
                            ON DELETE CASCADE
                    );

                    CREATE TABLE IF NOT EXISTS wechat_cover_assets (
                        asset_id TEXT PRIMARY KEY,
                        content_sha256 TEXT NOT NULL UNIQUE,
                        remote_media_id TEXT NOT NULL,
                        is_active INTEGER NOT NULL DEFAULT 0 CHECK (is_active IN (0, 1)),
                        created_at TEXT NOT NULL,
                        last_verified_at TEXT
                    );

                    CREATE UNIQUE INDEX IF NOT EXISTS
                        uq_wechat_cover_active
                    ON wechat_cover_assets (is_active)
                    WHERE is_active = 1;

                    CREATE TABLE IF NOT EXISTS image_attachments (
                        attachment_id TEXT PRIMARY KEY,
                        thread_id TEXT NOT NULL,
                        filename TEXT NOT NULL,
                        storage_path TEXT NOT NULL,
                        content_type TEXT NOT NULL,
                        size_bytes INTEGER NOT NULL CHECK (size_bytes > 0),
                        content_sha256 TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_image_attachments_thread
                    ON image_attachments (thread_id, created_at DESC);

                    CREATE TABLE IF NOT EXISTS image_attachment_analysis (
                        analysis_id TEXT PRIMARY KEY,
                        attachment_id TEXT NOT NULL UNIQUE,
                        status TEXT NOT NULL,
                        image_type TEXT,
                        confidence REAL,
                        summary TEXT,
                        ocr_text TEXT,
                        structured_result TEXT,
                        analysis_model TEXT,
                        analysis_version TEXT NOT NULL,
                        error_code TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY (attachment_id)
                            REFERENCES image_attachments (attachment_id)
                            ON DELETE CASCADE
                    );
                    CREATE INDEX IF NOT EXISTS idx_image_analysis_status
                    ON image_attachment_analysis (status, updated_at DESC);

                    CREATE TABLE IF NOT EXISTS audit_events (
                        event_id TEXT PRIMARY KEY,
                        action TEXT NOT NULL,
                        status TEXT NOT NULL,
                        actor TEXT NOT NULL,
                        article_id TEXT,
                        publication_id TEXT,
                        channel TEXT,
                        error_code TEXT,
                        occurred_at TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS
                        idx_audit_events_article
                    ON audit_events (article_id, occurred_at DESC);
                    """
                )
                _migrate_channel_constraints(connection)
                # 提交与异常处理
                connection.commit()
            except Exception:
                connection.close()
                raise

            self._connection = connection
    # 关闭仓库
    def close(self) -> None:
        with self._lock:
            if self._connection is None:
                return

            self._connection.close()
            self._connection = None

    @contextmanager
    # 事务上下文管理器
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._require_connection()
        with self._lock:
            try:
                # 立即启动一个事务并获取预留锁
                connection.execute("BEGIN IMMEDIATE")
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise
    # 连接守卫
    def _require_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RepositoryClosedError("publishing repository is not open")
        return self._connection
    # 插入文章
    def create_article(self, article: Article) -> Article:
        with self._transaction() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO articles (
                        article_id,
                        version,
                        source_thread_id,
                        source_artifact_id,
                        source_artifact_sha256,
                        title,
                        slug,
                        markdown_content,
                        excerpt,
                        tags_json,
                        status,
                        created_at,
                        updated_at,
                        approved_at,
                        published_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    _article_to_parameters(article),
                )
            except sqlite3.IntegrityError as error:
                raise DuplicateRecordError(
                    "article id, version, or slug already exists"
                ) from error

        return article
    # 创建修订版本
    def create_revision(
        self,
        article: Article,
        *,
        previous_version: int,
    ) -> Article:
        # 新文章的 version 必须严格大于 previous_version
        if article.version <= previous_version:
            raise DuplicateRecordError(
                "article revision must have a newer version"
            )

        with self._transaction() as connection:
            # 在事务中，先查询 previous_version 是否存在且 is_current = 1
            previous = connection.execute(
                """
                SELECT article_id
                FROM articles
                WHERE article_id = ? AND version = ? AND is_current = 1
                """,
                (article.article_id, previous_version),
            ).fetchone()
            # 如果查不到（说明版本不存在或已不是当前版本），抛 RecordNotFoundError
            if previous is None:
                raise RecordNotFoundError("current article version not found")

            connection.execute(
                """
                UPDATE articles
                SET is_current = 0
                WHERE article_id = ? AND version = ?
                """,
                (article.article_id, previous_version),
            )
            # 整个操作（查旧→更新旧→插入新）在同一个事务中，要么全部成功，要么全部回滚，保证数据一致性
            try:
                connection.execute(
                    """
                    INSERT INTO articles (
                        article_id,
                        version,
                        source_thread_id,
                        source_artifact_id,
                        source_artifact_sha256,
                        title,
                        slug,
                        markdown_content,
                        excerpt,
                        tags_json,
                        status,
                        created_at,
                        updated_at,
                        approved_at,
                        published_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    _article_to_parameters(article),
                )
            except sqlite3.IntegrityError as error:
                raise DuplicateRecordError(
                    "article revision or current slug already exists"
                ) from error

        return article
    # 更新当前版本文章
    def update_article(self, article: Article) -> Article:
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE articles
                SET source_thread_id = ?,
                    source_artifact_id = ?,
                    source_artifact_sha256 = ?,
                    title = ?,
                    slug = ?,
                    markdown_content = ?,
                    excerpt = ?,
                    tags_json = ?,
                    status = ?,
                    updated_at = ?,
                    approved_at = ?,
                    published_at = ?
                WHERE article_id = ? AND version = ? AND is_current = 1
                """,
                (
                    article.source_thread_id,
                    article.source_artifact_id,
                    article.source_artifact_sha256,
                    article.title,
                    article.slug,
                    article.markdown_content,
                    article.excerpt,
                    json.dumps(article.tags, ensure_ascii=False),
                    article.status.value,
                    _timestamp_to_storage(article.updated_at),
                    _timestamp_to_storage(article.approved_at),
                    _timestamp_to_storage(article.published_at),
                    article.article_id,
                    article.version,
                ),
            )
            if cursor.rowcount != 1:
                raise RecordNotFoundError("current article version not found")

        return article
    # 查询文章，指定 version 查特定版本，或 version=None 查当前版本
    def get_article(
        self,
        article_id: str,
        *,
        version: int | None = None,
    ) -> Article:
        connection = self._require_connection()
        with self._lock:
            if version is None:
                row = connection.execute(
                    """
                    SELECT *
                    FROM articles
                    WHERE article_id = ? AND is_current = 1
                    """,
                    (article_id,),
                ).fetchone()
            else:
                row = connection.execute(
                    """
                    SELECT *
                    FROM articles
                    WHERE article_id = ? AND version = ?
                    """,
                    (article_id, version),
                ).fetchone()

        if row is None:
            raise RecordNotFoundError("article not found")
        return _row_to_article(row)
    # 列出所有当前文章，返回所有当前版本的文章列表，按更新时间倒序排列
    def list_articles(self) -> list[Article]:
        connection = self._require_connection()
        with self._lock:
            rows = connection.execute(
                """
                SELECT *
                FROM articles
                WHERE is_current = 1
                ORDER BY updated_at DESC, article_id ASC
                """
            ).fetchall()
        return [_row_to_article(row) for row in rows]

    @staticmethod
    def _replace_publication_attachments(
        connection: sqlite3.Connection,
        publication: Publication,
    ) -> None:
        connection.execute(
            "DELETE FROM publication_attachments WHERE publication_id = ?",
            (publication.publication_id,),
        )
        if not publication.attachment_ids:
            return
        connection.executemany(
            """
            INSERT INTO publication_attachments (
                publication_id, attachment_id, position
            ) VALUES (?, ?, ?)
            """,
            [
                (publication.publication_id, attachment_id, position)
                for position, attachment_id in enumerate(
                    publication.attachment_ids
                )
            ],
        )

    def _get_publication_attachment_ids(
        self,
        publication_id: str,
    ) -> tuple[str, ...]:
        connection = self._require_connection()
        rows = connection.execute(
            """
            SELECT attachment_id
            FROM publication_attachments
            WHERE publication_id = ?
            ORDER BY position ASC
            """,
            (publication_id,),
        ).fetchall()
        return tuple(row["attachment_id"] for row in rows)

    def _get_approval_attachment_ids(
        self,
        approval_id: str,
    ) -> tuple[str, ...]:
        connection = self._require_connection()
        rows = connection.execute(
            """
            SELECT attachment_id
            FROM approval_request_attachments
            WHERE approval_id = ?
            ORDER BY position ASC
            """,
            (approval_id,),
        ).fetchall()
        return tuple(row["attachment_id"] for row in rows)

    # 创建发布记录（含幂等冲突处理）
    def create_publication(self, publication: Publication) -> Publication:
        with self._transaction() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO publications (
                        publication_id,
                        article_id,
                        article_version,
                        channel,
                        status,
                        idempotency_key,
                        content_sha256,
                        external_id,
                        public_url,
                        attempt_count,
                        error_code,
                        created_at,
                        updated_at,
                        published_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    _publication_to_parameters(publication),
                )
                self._replace_publication_attachments(connection, publication)
            except sqlite3.IntegrityError as error:
                existing = connection.execute(
                    """
                    SELECT publication_id, content_sha256
                    FROM publications
                    WHERE channel = ? AND idempotency_key = ?
                    """,
                    (
                        publication.channel.value,
                        publication.idempotency_key,
                    ),
                ).fetchone()

                if existing is not None:
                    # 如果已存在的记录 content_sha256 和本次不同 → 说明同一个幂等键被用于不同内容，这是客户端错误，抛 IdempotencyConflictError；
                    if existing["content_sha256"] != publication.content_sha256:
                        raise IdempotencyConflictError(
                            "idempotency key is already used for different content"
                        ) from error
                    # 如果 content_sha256 相同 → 说明是重复提交同一请求，抛 DuplicateRecordError
                    raise DuplicateRecordError(
                        "publication already exists"
                    ) from error
                raise DuplicateRecordError(
                    "publication or article reference already exists"
                ) from error

        return publication
    # 更新发布记录
    def update_publication(self, publication: Publication) -> Publication:
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE publications
                SET status = ?,
                    external_id = ?,
                    public_url = ?,
                    attempt_count = ?,
                    error_code = ?,
                    updated_at = ?,
                    published_at = ?
                WHERE publication_id = ?
                """,
                (
                    publication.status.value,
                    publication.external_id,
                    publication.public_url,
                    publication.attempt_count,
                    publication.error_code,
                    _timestamp_to_storage(publication.updated_at),
                    _timestamp_to_storage(publication.published_at),
                    publication.publication_id,
                ),
            )
            # 检查 rowcount != 1 确保记录存在，否则抛 RecordNotFoundError
            if cursor.rowcount != 1:
                raise RecordNotFoundError("publication not found")
            self._replace_publication_attachments(connection, publication)

        return publication
    # 按 ID 查询发布记录
    def get_publication(self, publication_id: str) -> Publication:
        connection = self._require_connection()
        with self._lock:
            row = connection.execute(
                """
                SELECT *
                FROM publications
                WHERE publication_id = ?
                """,
                (publication_id,),
            ).fetchone()

        if row is None:
            raise RecordNotFoundError("publication not found")
        return _row_to_publication(
            row,
            attachment_ids=self._get_publication_attachment_ids(
                publication_id
            ),
        )
    # 按幂等键查询,这是幂等机制的核心查询方法，用于检查请求是否已经处理过
    def get_publication_by_idempotency_key(
        self,
        *,
        channel: PublicationChannel,
        idempotency_key: str,
    ) -> Publication | None:
        connection = self._require_connection()
        with self._lock:
            row = connection.execute(
                """
                SELECT *
                FROM publications
                WHERE channel = ? AND idempotency_key = ?
                """,
                (channel.value, idempotency_key),
            ).fetchone()
        return (
            _row_to_publication(
                row,
                attachment_ids=self._get_publication_attachment_ids(
                    row["publication_id"]
                ),
            )
            if row is not None
            else None
        )
    # 按文章列出发布记录
    def list_publications(self, article_id: str) -> list[Publication]:
        connection = self._require_connection()
        with self._lock:
            rows = connection.execute(
                """
                SELECT *
                FROM publications
                WHERE article_id = ?
                ORDER BY created_at DESC, publication_id ASC
                """,
                (article_id,),
            ).fetchall()
        return [
            _row_to_publication(
                row,
                attachment_ids=self._get_publication_attachment_ids(
                    row["publication_id"]
                ),
            )
            for row in rows
        ]

    def get_wechat_cover_asset_by_hash(
        self,
        content_sha256: str,
    ) -> WeChatCoverAsset | None:
        with self._lock:
            connection = self._require_connection()
            row = connection.execute(
                """
                SELECT asset_id, content_sha256, remote_media_id, is_active,
                       created_at, last_verified_at
                FROM wechat_cover_assets
                WHERE content_sha256 = ?
                """,
                (content_sha256.lower(),),
            ).fetchone()
        return _row_to_wechat_cover_asset(row) if row is not None else None

    def create_image_attachment(self, attachment: ImageAttachment) -> ImageAttachment:
        with self._transaction() as connection:
            existing = connection.execute(
                """
                SELECT attachment_id, thread_id, filename, storage_path,
                       content_type, size_bytes, content_sha256, created_at
                FROM image_attachments
                WHERE content_sha256 = ?
                ORDER BY created_at ASC, attachment_id ASC
                LIMIT 1
                """,
                (attachment.content_sha256.lower(),),
            ).fetchone()
            if existing is not None:
                # Content hashes are the application-level idempotency key.
                # Returning the canonical row also makes concurrent uploads
                # converge because this lookup runs inside the write lock.
                return _row_to_image_attachment(existing)
            try:
                connection.execute(
                    """
                    INSERT INTO image_attachments (
                        attachment_id, thread_id, filename, storage_path,
                        content_type, size_bytes, content_sha256, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        attachment.attachment_id,
                        attachment.thread_id,
                        attachment.filename,
                        attachment.storage_path,
                        attachment.content_type,
                        attachment.size_bytes,
                        attachment.content_sha256,
                        _timestamp_to_storage(attachment.created_at),
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise DuplicateRecordError("image attachment already exists") from error
        return attachment

    def get_image_attachment(self, attachment_id: str) -> ImageAttachment:
        with self._lock:
            row = self._require_connection().execute(
                """
                SELECT attachment_id, thread_id, filename, storage_path,
                       content_type, size_bytes, content_sha256, created_at
                FROM image_attachments WHERE attachment_id = ?
                """,
                (attachment_id,),
            ).fetchone()
        if row is None:
            raise RecordNotFoundError("image attachment not found")
        return _row_to_image_attachment(row)

    def list_image_attachments(
        self,
        thread_id: str | None = None,
    ) -> list[ImageAttachment]:
        with self._lock:
            connection = self._require_connection()
            if thread_id is None:
                rows = connection.execute(
                    """
                    SELECT attachment_id, thread_id, filename, storage_path,
                           content_type, size_bytes, content_sha256, created_at
                    FROM image_attachments
                    ORDER BY created_at ASC, attachment_id ASC
                    """
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT attachment_id, thread_id, filename, storage_path,
                           content_type, size_bytes, content_sha256, created_at
                    FROM image_attachments WHERE thread_id = ?
                    ORDER BY created_at ASC, attachment_id ASC
                    """,
                    (thread_id,),
                ).fetchall()
        attachments: list[ImageAttachment] = []
        seen_hashes: set[str] = set()
        for row in rows:
            content_hash = str(row["content_sha256"]).lower()
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            attachments.append(_row_to_image_attachment(row))
        # Keep the library's usual newest-first presentation while selecting
        # the same earliest row that create_image_attachment uses as canonical.
        attachments.reverse()
        return attachments

    def delete_image_attachment(self, attachment_id: str) -> ImageAttachment:
        attachment = self.get_image_attachment(attachment_id)
        with self._transaction() as connection:
            connection.execute(
                "DELETE FROM image_attachment_analysis WHERE attachment_id = ?",
                (attachment_id,),
            )
            deleted = connection.execute(
                "DELETE FROM image_attachments WHERE attachment_id = ?",
                (attachment_id,),
            )
            if deleted.rowcount != 1:
                raise RecordNotFoundError("image attachment not found")
        return attachment

    def get_image_attachment_analysis(
        self,
        attachment_id: str,
    ) -> ImageAttachmentAnalysis | None:
        with self._lock:
            row = self._require_connection().execute(
                """
                SELECT analysis_id, attachment_id, status, image_type,
                       confidence, summary, ocr_text, structured_result,
                       analysis_model, analysis_version, error_code,
                       created_at, updated_at
                FROM image_attachment_analysis
                WHERE attachment_id = ?
                """,
                (attachment_id,),
            ).fetchone()
        return (
            _row_to_image_attachment_analysis(row)
            if row is not None
            else None
        )

    def save_image_attachment_analysis(
        self,
        analysis: ImageAttachmentAnalysis,
    ) -> ImageAttachmentAnalysis:
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO image_attachment_analysis (
                    analysis_id, attachment_id, status, image_type,
                    confidence, summary, ocr_text, structured_result,
                    analysis_model, analysis_version, error_code,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(attachment_id) DO UPDATE SET
                    analysis_id = excluded.analysis_id,
                    status = excluded.status,
                    image_type = excluded.image_type,
                    confidence = excluded.confidence,
                    summary = excluded.summary,
                    ocr_text = excluded.ocr_text,
                    structured_result = excluded.structured_result,
                    analysis_model = excluded.analysis_model,
                    analysis_version = excluded.analysis_version,
                    error_code = excluded.error_code,
                    updated_at = excluded.updated_at
                """,
                (
                    analysis.analysis_id,
                    analysis.attachment_id,
                    analysis.status,
                    analysis.image_type,
                    analysis.confidence,
                    analysis.summary,
                    analysis.ocr_text,
                    analysis.structured_result,
                    analysis.analysis_model,
                    analysis.analysis_version,
                    analysis.error_code,
                    _timestamp_to_storage(analysis.created_at),
                    _timestamp_to_storage(analysis.updated_at),
                ),
            )
        return analysis

    def deactivate_wechat_cover_asset_by_hash(
        self,
        content_sha256: str,
    ) -> None:
        with self._transaction() as connection:
            connection.execute(
                """
                UPDATE wechat_cover_assets
                SET is_active = 0
                WHERE content_sha256 = ? AND is_active = 1
                """,
                (content_sha256.lower(),),
            )

    def get_active_wechat_cover_asset(self) -> WeChatCoverAsset | None:
        with self._lock:
            connection = self._require_connection()
            row = connection.execute(
                """
                SELECT asset_id, content_sha256, remote_media_id, is_active,
                       created_at, last_verified_at
                FROM wechat_cover_assets
                WHERE is_active = 1
                LIMIT 1
                """
            ).fetchone()
        return _row_to_wechat_cover_asset(row) if row is not None else None

    def list_wechat_cover_assets(self) -> list[WeChatCoverAsset]:
        with self._lock:
            connection = self._require_connection()
            rows = connection.execute(
                """
                SELECT asset_id, content_sha256, remote_media_id, is_active,
                       created_at, last_verified_at
                FROM wechat_cover_assets
                ORDER BY is_active DESC, created_at DESC, asset_id ASC
                """
            ).fetchall()
        return [_row_to_wechat_cover_asset(row) for row in rows]

    def save_wechat_cover_asset(
        self,
        asset: WeChatCoverAsset,
        *,
        make_active: bool = False,
    ) -> WeChatCoverAsset:
        with self._transaction() as connection:
            if make_active:
                connection.execute(
                    "UPDATE wechat_cover_assets SET is_active = 0 WHERE is_active = 1"
                )
            try:
                connection.execute(
                    """
                    INSERT INTO wechat_cover_assets (
                        asset_id, content_sha256, remote_media_id, is_active,
                        created_at, last_verified_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(content_sha256) DO UPDATE SET
                        remote_media_id = excluded.remote_media_id,
                        is_active = CASE
                            WHEN excluded.is_active = 1 THEN 1
                            ELSE wechat_cover_assets.is_active
                        END,
                        last_verified_at = excluded.last_verified_at
                    """,
                    (
                        asset.asset_id,
                        asset.content_sha256,
                        asset.remote_media_id,
                        1 if make_active or asset.is_active else 0,
                        _timestamp_to_storage(asset.created_at),
                        _timestamp_to_storage(asset.last_verified_at),
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise DuplicateRecordError("wechat cover asset already exists") from error
        return self.get_wechat_cover_asset_by_hash(asset.content_sha256) or asset

    def set_active_wechat_cover_asset(self, asset_id: str) -> WeChatCoverAsset:
        with self._lock:
            connection = self._require_connection()
            exists = connection.execute(
                "SELECT 1 FROM wechat_cover_assets WHERE asset_id = ?",
                (asset_id,),
            ).fetchone()
        if exists is None:
            raise RecordNotFoundError("wechat cover asset not found")
        with self._transaction() as connection:
            connection.execute(
                "UPDATE wechat_cover_assets SET is_active = 0 WHERE is_active = 1"
            )
            updated = connection.execute(
                "UPDATE wechat_cover_assets SET is_active = 1, last_verified_at = ? WHERE asset_id = ?",
                (_timestamp_to_storage(utc_now()), asset_id),
            )
            if updated.rowcount != 1:
                raise RecordNotFoundError("wechat cover asset not found")
        asset = next(
            (item for item in self.list_wechat_cover_assets() if item.asset_id == asset_id),
            None,
        )
        if asset is None:
            raise RecordNotFoundError("wechat cover asset not found")
        return asset

    def list_incomplete_publications(
        self,
        *,
        channel: PublicationChannel = PublicationChannel.WECHAT_OFFICIAL_ACCOUNT,
        status: PublicationStatus = PublicationStatus.PUBLISHING,
    ) -> list[Publication]:
        """Return durable external tasks eligible for controlled recovery."""
        with self._lock:
            connection = self._require_connection()
            rows = connection.execute(
                """
                SELECT publication_id, article_id, article_version, channel,
                       status, idempotency_key, content_sha256, external_id,
                       public_url, attempt_count, error_code, created_at,
                       updated_at, published_at
                FROM publications
                WHERE channel = ? AND status = ?
                  AND external_id IS NOT NULL AND TRIM(external_id) <> ''
                ORDER BY updated_at ASC, publication_id ASC
                """,
                (channel.value, status.value),
            ).fetchall()
        return [
            _row_to_publication(
                row,
                attachment_ids=self._get_publication_attachment_ids(
                    row["publication_id"]
                ),
            )
            for row in rows
        ]

    # Explicit alias used by recovery callers; both names share one protocol.
    def list_resumable_publications(self) -> list[Publication]:
        return self.list_incomplete_publications()

    def create_approval_request(
        self,
        request: ApprovalRequest,
    ) -> ApprovalRequest:
        with self._transaction() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO approval_requests (
                        approval_id,
                        action,
                        article_id,
                        article_version,
                        channel,
                        content_sha256,
                        status,
                        created_at,
                        updated_at,
                        decided_at,
                        decision_actor,
                        decision_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    _approval_to_parameters(request),
                )
                if request.attachment_ids:
                    connection.executemany(
                        """
                        INSERT INTO approval_request_attachments (
                            approval_id, attachment_id, position
                        ) VALUES (?, ?, ?)
                        """,
                        [
                            (request.approval_id, attachment_id, position)
                            for position, attachment_id in enumerate(
                                request.attachment_ids
                            )
                        ],
                    )
            except sqlite3.IntegrityError as error:
                raise DuplicateRecordError(
                    "approval request or pending target already exists"
                ) from error

        return request

    def update_approval_request(
        self,
        request: ApprovalRequest,
    ) -> ApprovalRequest:
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE approval_requests
                SET status = ?,
                    updated_at = ?,
                    decided_at = ?,
                    decision_actor = ?,
                    decision_reason = ?
                WHERE approval_id = ?
                """,
                (
                    request.status.value,
                    _timestamp_to_storage(request.updated_at),
                    _timestamp_to_storage(request.decided_at),
                    request.decision_actor,
                    request.decision_reason,
                    request.approval_id,
                ),
            )
            if cursor.rowcount != 1:
                raise RecordNotFoundError("approval request not found")

        return request

    def get_approval_request(
        self,
        approval_id: str,
    ) -> ApprovalRequest:
        connection = self._require_connection()
        with self._lock:
            row = connection.execute(
                """
                SELECT *
                FROM approval_requests
                WHERE approval_id = ?
                """,
                (approval_id,),
            ).fetchone()

        if row is None:
            raise RecordNotFoundError("approval request not found")
        return _row_to_approval(
            row,
            attachment_ids=self._get_approval_attachment_ids(
                row["approval_id"]
            ),
        )

    def get_pending_approval_request(
        self,
        *,
        action: ApprovalAction,
        article_id: str,
        article_version: int,
        channel: PublicationChannel,
    ) -> ApprovalRequest | None:
        connection = self._require_connection()
        with self._lock:
            row = connection.execute(
                """
                SELECT *
                FROM approval_requests
                WHERE action = ?
                  AND article_id = ?
                  AND article_version = ?
                  AND channel = ?
                  AND status = ?
                """,
                (
                    action.value,
                    article_id,
                    article_version,
                    channel.value,
                    ApprovalStatus.PENDING.value,
                ),
            ).fetchone()

        return (
            _row_to_approval(
                row,
                attachment_ids=self._get_approval_attachment_ids(
                    row["approval_id"]
                ),
            )
            if row is not None
            else None
        )

    def list_approval_requests(
        self,
        *,
        article_id: str | None = None,
        status: ApprovalStatus | None = None,
        limit: int = 100,
    ) -> list[ApprovalRequest]:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be a positive integer")

        connection = self._require_connection()
        with self._lock:
            clauses = []
            parameters: list[object] = []

            if article_id is not None:
                clauses.append("article_id = ?")
                parameters.append(article_id)

            if status is not None:
                clauses.append("status = ?")
                parameters.append(status.value)

            where_clause = (
                "WHERE " + " AND ".join(clauses)
                if clauses
                else ""
            )
            parameters.append(limit)
            rows = connection.execute(
                f"""
                SELECT *
                FROM approval_requests
                {where_clause}
                ORDER BY created_at DESC, approval_id ASC
                LIMIT ?
                """,
                parameters,
            ).fetchall()

        return [
            _row_to_approval(
                row,
                attachment_ids=self._get_approval_attachment_ids(
                    row["approval_id"]
                ),
            )
            for row in rows
        ]

    def list_approval_requests_for_thread(
        self,
        thread_id: str,
        *,
        statuses: tuple[ApprovalStatus, ...] | None = None,
        limit: int = 100,
    ) -> list[ApprovalRequest]:
        """List approvals belonging to Articles sourced from one thread."""

        if not isinstance(thread_id, str) or not thread_id.strip():
            raise ValueError("thread_id must not be empty")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be a positive integer")
        if statuses is not None and not statuses:
            return []

        connection = self._require_connection()
        with self._lock:
            clauses = ["a.source_thread_id = ?"]
            parameters: list[object] = [thread_id]
            if statuses is not None:
                clauses.append(
                    "ar.status IN ("
                    + ",".join("?" for _ in statuses)
                    + ")"
                )
                parameters.extend(status.value for status in statuses)
            parameters.append(limit)
            rows = connection.execute(
                f"""
                SELECT ar.*
                FROM approval_requests AS ar
                INNER JOIN articles AS a
                    ON a.article_id = ar.article_id
                   AND a.version = ar.article_version
                WHERE {' AND '.join(clauses)}
                ORDER BY ar.created_at DESC, ar.approval_id ASC
                LIMIT ?
                """,
                parameters,
            ).fetchall()

        return [
            _row_to_approval(
                row,
                attachment_ids=self._get_approval_attachment_ids(
                    row["approval_id"]
                ),
            )
            for row in rows
        ]

    def record_audit_event(
        self,
        *,
        action: str,
        status: str,
        actor: str = "application",
        article_id: str | None = None,
        publication_id: str | None = None,
        channel: PublicationChannel | None = None,
        error_code: str | None = None,
        occurred_at: datetime | None = None,
    ) -> str:
        """Persist safe business metadata for user-visible audit history."""

        event_id = uuid4().hex
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO audit_events (
                    event_id,
                    action,
                    status,
                    actor,
                    article_id,
                    publication_id,
                    channel,
                    error_code,
                    occurred_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    action,
                    status,
                    actor,
                    article_id,
                    publication_id,
                    channel.value if channel is not None else None,
                    error_code,
                    _timestamp_to_storage(occurred_at),
                ),
            )
        return event_id

    def list_audit_events(
        self,
        *,
        article_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, object]]:
        """Return safe audit fields for the publishing management UI."""

        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be a positive integer")

        connection = self._require_connection()
        with self._lock:
            if article_id is None:
                rows = connection.execute(
                    """
                    SELECT event_id, action, status, actor, article_id,
                           publication_id, channel, error_code, occurred_at
                    FROM audit_events
                    ORDER BY occurred_at DESC, event_id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT event_id, action, status, actor, article_id,
                           publication_id, channel, error_code, occurred_at
                    FROM audit_events
                    WHERE article_id = ?
                    ORDER BY occurred_at DESC, event_id DESC
                    LIMIT ?
                    """,
                    (article_id, limit),
                ).fetchall()

        return [
            {
                "event_id": row["event_id"],
                "action": row["action"],
                "status": row["status"],
                "actor": row["actor"],
                "article_id": row["article_id"],
                "publication_id": row["publication_id"],
                "channel": row["channel"],
                "error_code": row["error_code"],
                "occurred_at": row["occurred_at"],
            }
            for row in rows
        ]


__all__ = [
    "ApprovalRequest",
    "ApprovalStatus",
    "DuplicateRecordError",
    "IdempotencyConflictError",
    "PublishingRepository",
    "PublishingRepositoryError",
    "RecordNotFoundError",
    "RepositoryClosedError",
]
